"""
agents/verification_agent.py

Stage 3: verifica simbolica dello schedule prodotto dal Drafting Agent.

Tutti i parametri vengono letti dal ModelDraft, non da costanti globali.
La verifica è interamente deterministica — nessun LLM coinvolto.

L'output di `verify()` è un `VerificationReport`: un dataclass che separa
le violazioni per categoria (invece del precedente `(bool, list[str])`),
così che:

    - il Drafting Agent LLM possa costruire repair prompt mirati,
      sapendo ESATTAMENTE quale categoria di vincolo è coinvolta;
    - il Feasibility Analyzer possa leggere suggerimenti già pronti,
      generati qui in modo deterministico, senza dover reinterpretare
      stringhe di log;
    - la UI (tab Risultati/Confronto) possa mostrare un riepilogo
      strutturato invece di una lista piatta di messaggi.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List

from models.worker import Worker
from models.schedule import Schedule
from input.model_draft_parser import ModelDraft
from solver.fairness import compute_satisfaction_scores

logger = logging.getLogger(__name__)

ViolationReport = List[str]


# ──────────────────────────────────────────────────────────────────────
# VerificationReport
# ──────────────────────────────────────────────────────────────────────

@dataclass
class VerificationReport:
    """
    Esito strutturato della verifica simbolica di una schedule.

    `feasible` è True se e solo se tutte le liste di violazioni sono
    vuote. `total_violations` è la somma di tutte le violazioni hard
    (non include i suggerimenti, che sono testo derivato).
    """

    feasible: bool
    one_shift: List[str] = field(default_factory=list)
    rest_after_night: List[str] = field(default_factory=list)
    weekly_hours: List[str] = field(default_factory=list)
    monthly_workload: List[str] = field(default_factory=list)
    coverage: List[str] = field(default_factory=list)
    consecutive_shifts: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    total_violations: int = 0

    @property
    def all_violations(self) -> List[str]:
        """Tutte le violazioni hard concatenate, indipendentemente dalla categoria."""
        return (
            self.one_shift
            + self.rest_after_night
            + self.weekly_hours
            + self.monthly_workload
            + self.coverage
            + self.consecutive_shifts
        )

    def category_counts(self) -> Dict[str, int]:
        """Numero di violazioni per categoria, utile per logging/UI."""
        return {
            "one_shift": len(self.one_shift),
            "rest_after_night": len(self.rest_after_night),
            "weekly_hours": len(self.weekly_hours),
            "monthly_workload": len(self.monthly_workload),
            "coverage": len(self.coverage),
            "consecutive_shifts": len(self.consecutive_shifts),
        }


def _build_suggestions(report: VerificationReport) -> List[str]:
    """
    Genera suggerimenti human-readable in base alle categorie di
    violazione popolate nel report. Sono pensati per essere sia
    mostrati in UI sia iniettati nel repair prompt per l'LLM.
    """

    suggestions: List[str] = []

    if report.coverage:
        suggestions.append(
            f"Copertura ({len(report.coverage)} violazioni): "
            "assegnare un altro lavoratore disponibile ai turni scoperti."
        )

    if report.weekly_hours:
        suggestions.append(
            f"Ore settimanali ({len(report.weekly_hours)} violazioni): "
            "spostare turni dai lavoratori sovraccarichi verso lavoratori "
            "con margine settimanale residuo."
        )

    if report.rest_after_night:
        suggestions.append(
            f"Riposo dopo la notte ({len(report.rest_after_night)} violazioni): "
            "rimuovere le assegnazioni immediatamente successive ai turni notturni."
        )

    if report.monthly_workload:
        suggestions.append(
            f"Carico mensile ({len(report.monthly_workload)} violazioni): "
            "riequilibrare le unità-turno tra i lavoratori per avvicinarsi "
            "al target mensile previsto."
        )

    if report.one_shift:
        suggestions.append(
            f"Turni per giorno ({len(report.one_shift)} violazioni): "
            "rimuovere i turni in eccesso assegnati allo stesso lavoratore "
            "nello stesso giorno."
        )

    if report.consecutive_shifts:
        suggestions.append(
            f"Turni consecutivi ({len(report.consecutive_shifts)} violazioni): "
            "rimuovere il turno di mattina assegnato il giorno successivo a "
            "un pomeriggio dello stesso lavoratore."
        )

    return suggestions


# ──────────────────────────────────────────────────────────────────────
# Helper vincoli
# ──────────────────────────────────────────────────────────────────────

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


def _check_no_consecutive_shifts(schedule: Schedule, draft: ModelDraft) -> ViolationReport:
    """
    Replica simbolica del vincolo C2 del solver OR-Tools
    (`solver/constraints.add_no_consecutive_shifts`):

        ∀w, ∀d: x[w,d,afternoon] + x[w,d+1,morning] ≤ 1

    Un lavoratore non può fare il turno di pomeriggio un giorno e il
    turno di mattina il giorno immediatamente successivo. È un vincolo
    distinto da "riposo dopo la notte" (che riguarda solo i turni di
    notte) e da "un turno al giorno" (che riguarda lo stesso giorno).
    """
    violations = []
    days = _all_days(draft)
    days_set = set(days)

    for worker in schedule.workers:
        for day in days:
            next_day = day + timedelta(days=1)
            if next_day not in days_set:
                continue
            if (
                schedule.is_assigned(worker.worker_id, day, "afternoon")
                and schedule.is_assigned(worker.worker_id, next_day, "morning")
            ):
                violations.append(
                    f"{worker.worker_id}: pomeriggio il {day.isoformat()} seguito "
                    f"da mattina il {next_day.isoformat()} (turni consecutivi non ammessi)"
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
    """
    Verifica i requisiti di copertura minima per turno.

    Use Case A: controllo sul totale dei presenti (tutti omogenei).

    Use Case B: verifica speculare al vincolo del solver corretto.
        La traccia dice che uno specializzato PUÒ coprire il ruolo standard,
        quindi le due condizioni da verificare sono:
            1. n_spe ≥ min_specialized  (gli specializzati non sono sostituibili)
            2. n_std + n_spe ≥ min_standard + min_specialized  (copertura totale)

        CORREZIONE rispetto alla versione precedente:
        Il vecchio check `n_std < cov.min_standard` era troppo restrittivo e
        avrebbe segnalato come violazione configurazioni valide come
        (1 std + 2 spe) con min_standard=2, min_specialized=1, che invece
        la traccia ammette esplicitamente come esempio valido.
    """
    violations = []
    days = _all_days(draft)

    for day in days:
        for cov in draft.coverage:
            s = cov.shift_type
            assigned = [w for w in schedule.workers
                        if schedule.is_assigned(w.worker_id, day, s)]

            if cov.min_specialized == 0:
                # ── Use Case A: copertura omogenea ────────────────────────────
                if len(assigned) < cov.min_standard:
                    violations.append(
                        f"Copertura: {day.isoformat()} {s} — "
                        f"{len(assigned)} lavoratori (min {cov.min_standard})"
                    )
            else:
                # ── Use Case B: copertura eterogenea ──────────────────────────
                n_std = sum(1 for w in assigned if w.role == "standard")
                n_spe = sum(1 for w in assigned if w.role == "specialized")

                # Verifica 1: presenza minima di specializzati (non sostituibili).
                if n_spe < cov.min_specialized:
                    violations.append(
                        f"Copertura: {day.isoformat()} {s} — "
                        f"solo {n_spe} specializzati (min {cov.min_specialized})"
                    )

                # Verifica 2: copertura totale del turno.
                # Gli specializzati in eccesso rispetto a min_specialized
                # coprono i posti standard rimanenti.
                total_required = cov.min_standard + cov.min_specialized
                if n_std + n_spe < total_required:
                    violations.append(
                        f"Copertura: {day.isoformat()} {s} — "
                        f"totale presenti {n_std + n_spe} (min {total_required}: "
                        f"{n_std} std + {n_spe} spe, specializzati possono coprire "
                        f"slot standard)"
                    )

    return violations


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def verify(
    schedule: Schedule, draft: ModelDraft
) -> VerificationReport:
    """
    Verifica tutti i vincoli hard e restituisce un `VerificationReport`
    strutturato per categoria, con suggerimenti human-readable allegati.
    """

    one_shift = _check_one_shift_per_day(schedule, draft)
    rest_after_night = _check_rest_after_night(schedule, draft)
    weekly_hours = _check_weekly_hours(schedule, draft)
    monthly_workload = _check_monthly_count(schedule, draft)
    coverage = _check_coverage(schedule, draft)
    consecutive_shifts = _check_no_consecutive_shifts(schedule, draft)

    named_checks = [
        ("Turni per giorno", one_shift),
        ("Riposo post-notte", rest_after_night),
        ("Ore settimanali", weekly_hours),
        ("Turni mensili", monthly_workload),
        ("Copertura turni", coverage),
        ("Turni consecutivi", consecutive_shifts),
    ]

    for name, violations in named_checks:
        if violations:
            logger.warning(f"[{name}] {len(violations)} violazioni")
        else:
            logger.info(f"[{name}] OK")

    total_violations = (
        len(one_shift)
        + len(rest_after_night)
        + len(weekly_hours)
        + len(monthly_workload)
        + len(coverage)
        + len(consecutive_shifts)
    )

    report = VerificationReport(
        feasible=total_violations == 0,
        one_shift=one_shift,
        rest_after_night=rest_after_night,
        weekly_hours=weekly_hours,
        monthly_workload=monthly_workload,
        coverage=coverage,
        consecutive_shifts=consecutive_shifts,
        suggestions=[],
        total_violations=total_violations,
    )

    report.suggestions = _build_suggestions(report)

    return report


def evaluate_fairness(schedule: Schedule) -> Dict[str, float]:
    """Calcola e salva i punteggi di soddisfazione nello schedule."""
    scores = compute_satisfaction_scores(schedule.workers, schedule)
    least = schedule.least_satisfied_worker()
    logger.info(
        f"Fairness: min={schedule.min_satisfaction():.3f} (worker: {least})"
    )
    return scores