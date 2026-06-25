"""
solver/model_builder.py

Costruisce il modello CP-SAT a partire dal ModelDraft e dai Worker.

Scelta progettuale: il ModelDraft è ora la singola fonte di verità
per tutti i parametri strutturali. Il model_builder non conosce
hardcoded nessun vincolo — delega tutto a constraints.py, che a sua
volta legge i parametri dal draft. Questo rende il sistema
completamente guidato dal file di input istituzionale.
"""

from datetime import date, timedelta
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

from models.worker import Worker
from input.model_draft_parser import ModelDraft
from solver.constraints import (
    Vars,
    add_one_shift_per_day,
    add_no_consecutive_shifts,
    add_rest_after_night,
    add_max_weekly_hours,
    add_monthly_shift_count,
    add_coverage,
)


def _generate_days(draft: ModelDraft) -> List[date]:
    days = []
    current = draft.start_date
    while current <= draft.end_date:
        days.append(current)
        current += timedelta(days=1)
    return days


def build_model(
    workers: List[Worker],
    draft: ModelDraft,
) -> Tuple[cp_model.CpModel, Vars, List[date]]:
    """
    Costruisce il modello CP-SAT con variabili e vincoli hard.

    Args:
        workers: lista dei lavoratori con preferenze
        draft: model draft istituzionale (turni, vincoli, copertura)

    Returns:
        (model, x, days)
    """
    model = cp_model.CpModel()
    days = _generate_days(draft)
    shift_types = draft.shift_names

    # Variabili di decisione: x[worker_id, day, shift_type] ∈ {0, 1}
    x: Vars = {}
    for worker in workers:
        for day in days:
            for s in shift_types:
                x[(worker.worker_id, day, s)] = model.new_bool_var(
                    f"x_{worker.worker_id}_{day.isoformat()}_{s}"
                )

    # Applicazione vincoli hard (tutti parametrizzati dal draft)
    add_one_shift_per_day(model, x, workers, draft)
    add_no_consecutive_shifts(model, x, workers, draft)
    add_rest_after_night(model, x, workers, draft)
    add_max_weekly_hours(model, x, workers, draft)
    add_monthly_shift_count(model, x, workers, draft)
    add_coverage(model, x, workers, draft)

    return model, x, days


def export_model_summary(
    workers: List[Worker],
    draft: ModelDraft,
    x: Vars,
    days: List[date],
    filepath: str,
) -> None:
    """
    Esporta una descrizione leggibile del modello CP-SAT parziale.

    Questo è uno dei deliverable richiesti dalla traccia:
    "The input partial OR-tools sat cp_model".

    Il file prodotto descrive: variabili, vincoli hard in forma
    matematica e parametri dell'obiettivo — senza la soluzione,
    che viene aggiunta dopo la risoluzione.

    Scelta progettuale: esportiamo una rappresentazione testuale
    strutturata piuttosto che il bytecode interno di OR-Tools,
    perché è più leggibile e verificabile dagli esaminatori.
    """
    shift_types = draft.shift_names
    weights = draft.shift_weights
    durations = draft.shift_durations
    n_days = len(days)
    n_workers = len(workers)
    n_vars = len(x)

    lines = [
        "=" * 70,
        "SMARTSCHEDULER — OR-Tools CP-SAT Model (Partial)",
        "=" * 70,
        "",
        "[DECISION VARIABLES]",
        f"  Type: BoolVar  (0 = not assigned, 1 = assigned)",
        f"  Schema: x[worker_id, date, shift_type]",
        f"  Total: {n_vars} variables",
        f"    Workers  : {n_workers}",
        f"    Days     : {n_days}  ({draft.start_date} to {draft.end_date})",
        f"    Shifts   : {shift_types}",
        "",
        "[HARD CONSTRAINTS]",
        "",
        "  C1 — One shift per day:",
        f"    ∀w, ∀d: Σ_s x[w,d,s] ≤ {draft.constraints.max_shifts_per_day}",
        "",
        "  C2 — No consecutive shifts (afternoon→next morning):",
        "    ∀w, ∀d: x[w,d,afternoon] + x[w,d+1,morning] ≤ 1",
        "",
        f"  C3 — Rest after night ({draft.constraints.rest_days_after_night} days):",
        f"    ∀w, ∀d, ∀k∈{{1,...,{draft.constraints.rest_days_after_night}}}:",
        "        x[w,d,night] + Σ_s x[w,d+k,s] ≤ 1",
        "",
        f"  C4 — Max weekly hours ({draft.constraints.max_hours_per_week}h):",
        f"    ∀w, ∀week W: Σ_{{d∈W,s}} duration[s]*x[w,d,s] ≤ {draft.constraints.max_hours_per_week}",
        f"    Durations: {durations}",
        "",
        f"  C5 — Monthly shift count (= {draft.constraints.shifts_per_month} units):",
        f"    ∀w: Σ_{{d,s}} weight[s]*x[w,d,s] = {draft.constraints.shifts_per_month}",
        f"    Weights: {weights}",
        "",
        "  C6 — Shift coverage:",
    ]

    for cov in draft.coverage:
        if cov.min_specialized == 0:
            lines.append(
                f"    ∀d: Σ_w x[w,d,{cov.shift_type}] ≥ {cov.min_standard}  (all workers)"
            )
        else:
            lines.append(
                f"    ∀d: Σ_{{w∈standard}} x[w,d,{cov.shift_type}] ≥ {cov.min_standard}"
            )
            lines.append(
                f"         Σ_{{w∈specialized}} x[w,d,{cov.shift_type}] ≥ {cov.min_specialized}"
            )

    lines += [
        "",
        "[OBJECTIVE]",
        "  Maximize: Σ_{w,d,s} pref_score(w,d,s) * x[w,d,s]",
        "  where pref_score ∈ [-1.0, +1.0] is derived from worker preferences.",
        "  In refinement mode, the least-satisfied worker's terms",
        "  are multiplied by a boost_factor to improve Maximin fairness.",
        "",
        "[WORKFORCE]",
    ]

    for role, count in draft.workforce.roles.items():
        lines.append(f"  {role}: {count} workers")

    lines += [
        "",
        "  Workers list:",
    ]
    for w in workers:
        lines.append(
            f"    {w.worker_id} ({w.role}): "
            f"prefer={w.preferred_shifts}, avoid={w.avoid_shifts}, "
            f"night_tol={w.night_tolerance:.1f}"
        )

    lines += ["", "=" * 70, ""]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def extract_schedule_from_solution(
    workers: List[Worker],
    x: Vars,
    days: List[date],
    shift_types: List[str],
    solver: cp_model.CpSolver,
) -> Dict:
    """Estrae le assegnazioni positive dalla soluzione del solver."""
    assignments = {}
    for worker in workers:
        for day in days:
            for s in shift_types:
                key = (worker.worker_id, day, s)
                if solver.value(x[key]):
                    assignments[key] = True
    return assignments
