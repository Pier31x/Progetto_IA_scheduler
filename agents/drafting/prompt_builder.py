"""
agents/drafting/prompt_builder.py

Costruzione dei prompt per il Drafting Agent basato su LLM.

Scelta progettuale
-----------------------------------
L'LLM NON tenta più, da solo, di produrre o riparare uno schedule
fattibile: propone solo assegnazioni guidate dalle preferenze, che il
Drafting Agent valida e "blocca" (pin) in un modello CP-SAT — è il
solver OR-Tools a garantire il completamento fattibile del resto,
sfruttando i vincoli hard già implementati in `solver/constraints.py`.

Questo sostituisce i tre prompt precedenti (drafting / repair /
refinement) con un unico prompt (`build_more_assignments_prompt`),
usato sia per la prima proposta sia per le proposte successive: cambia
solo quali worker sono "a fuoco" (tutti, per un draft da zero; solo
alcuni, per un refinement/repair mirato) e quali assegnazioni sono già
bloccate (da mostrare come contesto, per evitare duplicati).
"""

from __future__ import annotations

import json
from typing import List, Optional, Set

from input.model_draft_parser import ModelDraft
from models.worker import Worker
from models.schedule import Schedule
from agents.verification_agent import VerificationReport


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _worker_description(worker: Worker) -> str:
    """Restituisce una descrizione testuale del lavoratore."""

    preferred = (
        ", ".join(worker.preferred_shifts)
        if worker.preferred_shifts
        else "none"
    )

    avoid = (
        ", ".join(worker.avoid_shifts)
        if worker.avoid_shifts
        else "none"
    )

    return (
        f"- {worker.worker_id}\n"
        f"  role: {worker.role}\n"
        f"  preferred shifts: {preferred}\n"
        f"  avoid shifts: {avoid}\n"
        f"  night tolerance: {worker.night_tolerance}\n"
    )


def _serialize_assignments(assignments) -> str:
    """Serializza una lista di Assignment nella struttura JSON standard."""

    payload = [
        {
            "worker": a.worker_id,
            "day": a.day.isoformat(),
            "shift": a.shift_type,
        }
        for a in assignments
    ]

    return json.dumps({"assignments": payload}, indent=2, ensure_ascii=False)


def extract_involved_workers(
    workers: List[Worker],
    report: VerificationReport,
) -> Set[str]:
    """
    Individua quali worker_id compaiono nei messaggi di violazione di un
    `VerificationReport` — usato dal Drafting Agent per restringere il
    "perimetro libero" su cui l'LLM propone assegnazioni durante un
    repair mirato.

    Le violazioni di copertura non nominano un worker specifico (sono
    per definizione un'assenza), quindi non contribuiscono a questo
    insieme.

    Se nessun worker specifico viene identificato (caso raro: solo
    violazioni di copertura), il fallback è l'intero roster — evita di
    restituire un insieme vuoto che bloccherebbe qualunque riparazione.
    """

    ids = {w.worker_id for w in workers}
    involved: Set[str] = set()

    for v in report.all_violations:
        for wid in ids:
            if v.startswith(f"{wid}:"):
                involved.add(wid)
                break

    if not involved:
        involved = set(ids)

    return involved


# ──────────────────────────────────────────────────────────────────────
# Prompt unico
# ──────────────────────────────────────────────────────────────────────

def build_more_assignments_prompt(
    workers: List[Worker],
    draft: ModelDraft,
    locked_assignments: List,
    focus_worker_ids: Optional[Set[str]] = None,
    fairness_target: Optional[str] = None,
) -> str:
    """
    Prompt unico per proporre assegnazioni guidate dalle preferenze.

    Usato sia per la prima proposta (`focus_worker_ids=None` → tutti i
    worker, `locked_assignments=[]`) sia per proposte successive/mirate
    (`focus_worker_ids` ristretto ai worker coinvolti in un repair o al
    worker target di un refinement).

    L'LLM NON deve garantire la fattibilità: propone solo assegnazioni
    plausibili in base alle preferenze. Il Drafting Agent le valida
    contro i vincoli hard prima di bloccarle, e il solver OR-Tools
    completa e garantisce il resto — quindi qui non c'è alcuna
    richiesta di "correggere violazioni" o "ricopiare tutto".
    """

    if focus_worker_ids is not None:
        relevant_workers = [w for w in workers if w.worker_id in focus_worker_ids]
        scope_note = (
            f"Focus ONLY on these {len(relevant_workers)} workers — "
            f"the other {len(workers) - len(relevant_workers)} are handled "
            f"separately, do not mention them."
        )
    else:
        relevant_workers = workers
        scope_note = "Consider all workers below."

    workers_description = "\n".join(
        _worker_description(w) for w in relevant_workers
    )

    relevant_ids = {w.worker_id for w in relevant_workers}
    locked_for_relevant = [
        a for a in locked_assignments if a.worker_id in relevant_ids
    ]
    locked_json = _serialize_assignments(locked_for_relevant)

    coverage_lines = []
    for c in draft.coverage:
        if c.min_specialized == 0:
            coverage_lines.append(f"{c.shift_type}: at least {c.min_standard} workers")
        else:
            coverage_lines.append(
                f"{c.shift_type}: {c.min_standard} standard + "
                f"{c.min_specialized} specialized"
            )
    coverage_section = "\n".join(coverage_lines)

    fairness_section = ""
    if fairness_target is not None:
        fairness_section = f"""
==========================================================
FAIRNESS PRIORITY
==========================================================

Worker {fairness_target} currently has the lowest satisfaction. Give
special priority to THEIR preferred shifts and avoided shifts when
proposing assignments for them.
"""

    return f"""
You are a hospital scheduling assistant proposing PREFERENCE-GUIDED
shift assignments. You do NOT need to produce a complete or feasible
schedule — a separate constraint solver will validate and complete
whatever you propose, guaranteeing all hard constraints are respected.
Your job is only to suggest assignments that a worker would clearly
want, based on their stated preferences.

{scope_note}

==========================================================
PLANNING HORIZON
==========================================================

From {draft.start_date} to {draft.end_date}

Shift types: {", ".join(draft.shift_names)}

Coverage requirements (for context only, the solver enforces these):
{coverage_section}

==========================================================
WORKERS
==========================================================

{workers_description}

==========================================================
ALREADY PROPOSED (do not repeat these worker+day combinations)
==========================================================

{locked_json}
{fairness_section}
==========================================================
YOUR TASK
==========================================================

- Propose ADDITIONAL assignments for the workers above, on days not already listed for them.
- Aim to build a realistic monthly calendar skeleton for each worker:
  1. Assign at most ONE shift per day to each worker (NEVER double-shift them on the same day).
  2. Maximize their 'preferred_shifts' and protect their 'preferred_days_off' (keep those days empty).
  3. NEVER assign a shift listed in 'avoid_shifts'.
  4. Fill the remaining active days with neutral/standard shifts to ensure a dense schedule, without exceeding an average of 4-5 working days per week.
- Aim for a high-quality proposal. A separate solver will do the final mathematical verification of exact hours and legally required rest periods, so focus your attention on maximizing worker satisfaction.

Return ONLY valid JSON in this format:


{{
  "assignments": [
    {{
      "worker":"W01",
      "day":"YYYY-MM-DD",
      "shift":"morning"
    }}
  ]
}}

Do not explain your reasoning. Output only the JSON object.
""".strip()