"""
agents/verification_agent.py

Stage 3: verifica simbolica dello schedule prodotto dal Drafting Agent.

Tutti i parametri vengono letti dal ModelDraft, non da costanti globali.
La verifica è interamente deterministica — nessun LLM coinvolto.
"""

import logging
from datetime import date, timedelta
from typing import Dict, List, Tuple

from models.worker import Worker
from models.schedule import Schedule
from input.model_draft_parser import ModelDraft
from solver.fairness import compute_satisfaction_scores

logger = logging.getLogger(__name__)
ViolationReport = List[str]


def _all_days(draft: ModelDraft) -> List[date]:
    days = []
    current = draft.start_date
    while current <= draft.end_date:
        days.append(current)
        current += timedelta(days=1)
    return days


def _check_one_shift_per_day(schedule: Schedule, draft: ModelDraft) -> ViolationReport:
    violations = []
    for worker in schedule.workers:
        for day in _all_days(draft):
            assigned = [s for s in draft.shift_names
                        if schedule.is_assigned(worker.worker_id, day, s)]
            if len(assigned) > draft.constraints.max_shifts_per_day:
                violations.append(
                    f"{worker.worker_id}: {len(assigned)} turni il {day.isoformat()}"
                )
    return violations


def _check_rest_after_night(schedule: Schedule, draft: ModelDraft) -> ViolationReport:
    violations = []
    days = _all_days(draft)
    days_set = set(days)
    k_max = draft.constraints.rest_days_after_night

    for worker in schedule.workers:
        for day in days:
            if not schedule.is_assigned(worker.worker_id, day, "night"):
                continue
            for k in range(1, k_max + 1):
                rest_day = day + timedelta(days=k)
                if rest_day not in days_set:
                    continue
                for s in draft.shift_names:
                    if schedule.is_assigned(worker.worker_id, rest_day, s):
                        violations.append(
                            f"{worker.worker_id}: turno {s} il {rest_day.isoformat()} "
                            f"dopo notte del {day.isoformat()}"
                        )
    return violations


def _check_weekly_hours(schedule: Schedule, draft: ModelDraft) -> ViolationReport:
    violations = []
    durations = draft.shift_durations
    days = _all_days(draft)
    max_h = draft.constraints.max_hours_per_week
    num_weeks = (len(days) + 6) // 7

    for worker in schedule.workers:
        for wi in range(num_weeks):
            week_days = days[wi * 7: (wi + 1) * 7]
            total_h = sum(
                durations[s]
                for d in week_days
                for s in draft.shift_names
                if schedule.is_assigned(worker.worker_id, d, s)
            )
            if total_h > max_h:
                violations.append(
                    f"{worker.worker_id}: {total_h}h settimana {wi+1} (max {max_h}h)"
                )
    return violations


def _check_monthly_count(schedule: Schedule, draft: ModelDraft) -> ViolationReport:
    violations = []
    weights = draft.shift_weights
    days = _all_days(draft)
    target = draft.constraints.shifts_per_month

    for worker in schedule.workers:
        total = sum(
            weights[s]
            for d in days
            for s in draft.shift_names
            if schedule.is_assigned(worker.worker_id, d, s)
        )
        if total != target:
            violations.append(
                f"{worker.worker_id}: {total} unità-turno (attese {target})"
            )
    return violations


def _check_coverage(schedule: Schedule, draft: ModelDraft) -> ViolationReport:
    violations = []
    days = _all_days(draft)

    for day in days:
        for cov in draft.coverage:
            s = cov.shift_type
            assigned = [w for w in schedule.workers
                        if schedule.is_assigned(w.worker_id, day, s)]

            if cov.min_specialized == 0:
                # Use Case A
                if len(assigned) < cov.min_standard:
                    violations.append(
                        f"Copertura: {day.isoformat()} {s} — "
                        f"{len(assigned)} lavoratori (min {cov.min_standard})"
                    )
            else:
                # Use Case B
                n_std = sum(1 for w in assigned if w.role == "standard")
                n_spe = sum(1 for w in assigned if w.role == "specialized")
                if n_spe < cov.min_specialized:
                    violations.append(
                        f"Copertura: {day.isoformat()} {s} — "
                        f"solo {n_spe} specializzati (min {cov.min_specialized})"
                    )
                if n_std < cov.min_standard:
                    violations.append(
                        f"Copertura: {day.isoformat()} {s} — "
                        f"solo {n_std} standard (min {cov.min_standard})"
                    )
    return violations


def verify(
    schedule: Schedule, draft: ModelDraft
) -> Tuple[bool, ViolationReport]:
    """Verifica tutti i vincoli hard. Restituisce (is_valid, violations)."""
    all_violations: ViolationReport = []

    checks = [
        ("Turni per giorno",    _check_one_shift_per_day(schedule, draft)),
        ("Riposo post-notte",   _check_rest_after_night(schedule, draft)),
        ("Ore settimanali",     _check_weekly_hours(schedule, draft)),
        ("Turni mensili",       _check_monthly_count(schedule, draft)),
        ("Copertura turni",     _check_coverage(schedule, draft)),
    ]

    for name, violations in checks:
        if violations:
            logger.warning(f"[{name}] {len(violations)} violazioni")
            all_violations.extend(violations)
        else:
            logger.info(f"[{name}] OK")

    return len(all_violations) == 0, all_violations


def evaluate_fairness(schedule: Schedule) -> Dict[str, float]:
    """Calcola e salva i punteggi di soddisfazione nello schedule."""
    scores = compute_satisfaction_scores(schedule.workers, schedule)
    least = schedule.least_satisfied_worker()
    logger.info(
        f"Fairness: min={schedule.min_satisfaction():.3f} (worker: {least})"
    )
    return scores
