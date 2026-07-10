"""
agents/refinement_agent.py

Stage 4 — Refinement Agent.

Ad ogni iterazione:

    1. individua il worker meno soddisfatto nella schedule corrente;
    2. chiede al Drafting Agent (Solver o LLM) di generare una nuova
       proposta mirata a migliorare quel worker, PARTENDO dalla
       schedule corrente (mai da zero, mai da "{}");
    3. se la proposta non è fattibile, la ripara passando il
       `VerificationReport` prodotto dal Verification Agent, sempre
       insieme alla schedule su cui si è verificata la violazione;
    4. accetta la proposta solo se migliora il worker target senza
       peggiorare il minimo globale di soddisfazione.

Il Drafting Agent basato su LLM richiede esplicitamente una
`current_schedule` reale ogni volta che gli si chiede un refinement o
un repair: qui la passiamo sempre, così non finisce mai per ricevere
"{}" come stato precedente. Per rimanere compatibili anche con un
Drafting Agent che non supporta questo parametro (es. quello OR-Tools
puro), lo passiamo solo se la sua `solve()` lo accetta.
"""

import inspect
import logging
from typing import Optional

from models.schedule import Schedule
from input.model_draft_parser import ModelDraft
from agents.drafting.base import DraftingAgent
from agents.verification_agent import verify, evaluate_fairness, VerificationReport
from solver.fairness import is_fairness_improvement

logger = logging.getLogger(__name__)
MAX_REFINEMENT_ITERATIONS = 30


def _solve_supports(param_name: str, drafting_agent: DraftingAgent) -> bool:
    """
    Verifica se `drafting_agent.solve()` accetta un parametro con questo
    nome, per poter passare `current_schedule` solo agli agenti che lo
    supportano (es. LLMDraftingAgent) senza rompere quelli che non lo
    conoscono (es. un Drafting Agent OR-Tools puro).
    """
    try:
        sig = inspect.signature(drafting_agent.solve)
        return param_name in sig.parameters
    except (TypeError, ValueError):
        return False


def refine(
    initial_schedule: Schedule,
    drafting_agent: DraftingAgent,
    draft: ModelDraft,
    max_iterations: int = MAX_REFINEMENT_ITERATIONS,
) -> Schedule:

    current = initial_schedule
    workers = initial_schedule.workers

    if not current.satisfaction_scores:
        evaluate_fairness(current)

    logger.info(
        f"[Refinement Agent] Avviato. Min iniziale: {current.min_satisfaction():.3f}"
    )

    supports_current_schedule = _solve_supports("current_schedule", drafting_agent)

    for iteration in range(1, max_iterations + 1):

        old_scores = dict(current.satisfaction_scores)
        target_id = current.least_satisfied_worker()
        old_global_min = min(old_scores.values())

        logger.info(
            f"[Iter {iteration}] Target: {target_id} "
            f"(score={old_scores[target_id]:.3f}, global_min={old_global_min:.3f})"
        )

        # ── STEP 1: proposta di refinement ──────────────────────────
        solve_kwargs = dict(
            workers=workers,
            draft=draft,
            least_satisfied_id=target_id,
            boost_factor=5,
        )
        if supports_current_schedule:
            solve_kwargs["current_schedule"] = current

        new_schedule = drafting_agent.solve(**solve_kwargs)

        if new_schedule is None:
            logger.warning(f"[Iter {iteration}] Nessuna soluzione trovata. Stop.")
            break

        # ── STEP 2: verifica ─────────────────────────────────────────
        report: VerificationReport = verify(new_schedule, draft)

        # ── STEP 3: repair loop ──────────────────────────────────────
        if not report.feasible:
            logger.warning(
                f"[Iter {iteration}] {report.total_violations} violazioni. Retry repair."
            )

            repair_kwargs = dict(
                workers=workers,
                draft=draft,
                least_satisfied_id=None,
                violation_feedback=report,
                boost_factor=1,
            )
            if supports_current_schedule:
                repair_kwargs["current_schedule"] = new_schedule

            new_schedule = drafting_agent.solve(**repair_kwargs)

            if new_schedule is None:
                logger.warning(f"[Iter {iteration}] Repair fallito.")
                break

            report = verify(new_schedule, draft)

            if not report.feasible:
                logger.warning(
                    f"[Iter {iteration}] Repair non risolutivo. "
                    f"Violazioni rimanenti: {report.total_violations}"
                )
                break

            logger.info(f"[Iter {iteration}] Repair riuscito. Schedule fattibile.")

        # ── STEP 4: valutazione fairness ─────────────────────────────
        evaluate_fairness(new_schedule)
        new_scores = new_schedule.satisfaction_scores
        new_global_min = min(new_scores.values())

        # ── STEP 5: criterio di accettazione ─────────────────────────
        if is_fairness_improvement(old_scores, new_scores, target_id):

            """if new_global_min < old_global_min - 1e-4:
                logger.info(
                    f"[Iter {iteration}] Rifiutato: min peggiorato "
                    f"({old_global_min:.3f} → {new_global_min:.3f})"
                )
                break"""

            logger.info(
                f"[Iter {iteration}] Accettato: target {target_id} "
                f"({old_scores[target_id]:.3f} → {new_scores[target_id]:.3f}) | "
                f"min {old_global_min:.3f} → {new_global_min:.3f}"
            )

            current = new_schedule

            if new_global_min <= old_global_min + 1e-4:
                logger.info(
                    f"[Iter {iteration}] Convergenza raggiunta."
                )
                break

        else:
            logger.info(
                f"[Iter {iteration}] Nessun miglioramento per target {target_id}."
            )
            break

    logger.info(
        f"[Refinement Agent] Completato. Min finale: {current.min_satisfaction():.3f}"
    )

    return current