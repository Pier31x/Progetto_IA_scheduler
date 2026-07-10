"""
solver/constraints.py

Definisce e applica tutti i vincoli hard al modello CP-SAT di OR-Tools.

Scelta progettuale: ogni vincolo è una funzione indipendente che riceve
il modello, le variabili e il ModelDraft (che contiene tutti i parametri
istituzionali). Questo rende ogni vincolo leggibile, testabile e
modificabile in isolamento.

Notazione:
    x[w_id, day, shift_type] = BoolVar
        1 se il lavoratore w_id è assegnato al turno shift_type del giorno day
        0 altrimenti
"""

from datetime import date, timedelta
from typing import Dict, List, Tuple

from ortools.sat.python import cp_model

from models.worker import Worker
from input.model_draft_parser import ModelDraft

Vars = Dict[Tuple[str, date, str], cp_model.IntVar]


def _all_days(draft: ModelDraft) -> List[date]:
    days = []
    current = draft.start_date
    while current <= draft.end_date:
        days.append(current)
        current += timedelta(days=1)
    return days


def add_one_shift_per_day(
    model: cp_model.CpModel, x: Vars, workers: List[Worker], draft: ModelDraft
) -> None:
    """
    Vincolo: ogni lavoratore copre al massimo un turno al giorno.

    Formulazione: ∀w, ∀d: Σ_s x[w,d,s] ≤ max_shifts_per_day (= 1)
    """
    shift_types = draft.shift_names
    limit = draft.constraints.max_shifts_per_day
    for worker in workers:
        for day in _all_days(draft):
            model.add(
                sum(x[worker.worker_id, day, s] for s in shift_types) <= limit
            )


def add_no_consecutive_shifts(
    model: cp_model.CpModel, x: Vars, workers: List[Worker], draft: ModelDraft
) -> None:
    """
    Vincolo: nessun turno consecutivo tra giorni adiacenti.
    Pomeriggio del giorno d + Mattino del giorno d+1 ≤ 1.

    Formulazione: ∀w, ∀d: x[w,d,afternoon] + x[w,d+1,morning] ≤ 1
    """
    days = _all_days(draft)
    for worker in workers:
        for i in range(len(days) - 1):
            d, d_next = days[i], days[i + 1]
            model.add(
                x[worker.worker_id, d, "afternoon"] +
                x[worker.worker_id, d_next, "morning"] <= 1
            )


def add_rest_after_night(
    model: cp_model.CpModel, x: Vars, workers: List[Worker], draft: ModelDraft
) -> None:
    """
    Vincolo: dopo ogni turno notturno, il lavoratore deve avere
    rest_days_after_night (= 2) giorni completamente liberi.

    Formulazione: ∀w, ∀d, ∀k ∈ {1,...,K}: x[w,d,night] + Σ_s x[w,d+k,s] ≤ 1
    """
    days = _all_days(draft)
    days_set = set(days)
    shift_types = draft.shift_names
    k_max = draft.constraints.rest_days_after_night

    for worker in workers:
        for day in days:
            for k in range(1, k_max + 1):
                rest_day = day + timedelta(days=k)
                if rest_day in days_set:
                    for s in shift_types:
                        model.add(
                            x[worker.worker_id, day, "night"] +
                            x[worker.worker_id, rest_day, s] <= 1
                        )


def add_max_weekly_hours(
    model: cp_model.CpModel, x: Vars, workers: List[Worker], draft: ModelDraft
) -> None:
    """
    Vincolo: max max_hours_per_week (= 36) ore settimanali per lavoratore.

    Formulazione: ∀w, ∀W (settimana): Σ_{d∈W, s} duration[s]*x[w,d,s] ≤ 36
    """
    durations = draft.shift_durations
    shift_types = draft.shift_names
    days = _all_days(draft)
    max_h = draft.constraints.max_hours_per_week
    num_weeks = (len(days) + 6) // 7

    for worker in workers:
        for week_idx in range(num_weeks):
            week_days = days[week_idx * 7: (week_idx + 1) * 7]
            weekly_hours = sum(
                durations[s] * x[worker.worker_id, d, s]
                for d in week_days
                for s in shift_types
                if (worker.worker_id, d, s) in x
            )
            model.add(weekly_hours <= max_h)


def add_monthly_shift_count(
    model: cp_model.CpModel, x: Vars, workers: List[Worker], draft: ModelDraft
) -> None:
    """
    Vincolo: ogni lavoratore copre esattamente shifts_per_month (= 25)
    unità-turno nel mese (la notte pesa 2).

    Formulazione: ∀w: Σ_{d,s} weight[s] * x[w,d,s] = 25
    """
    weights = draft.shift_weights
    shift_types = draft.shift_names
    days = _all_days(draft)
    target = draft.constraints.shifts_per_month

    for worker in workers:
        total = sum(
            weights[s] * x[worker.worker_id, d, s]
            for d in days
            for s in shift_types
        )
        model.add(total == target)


def add_coverage(
    model: cp_model.CpModel, x: Vars, workers: List[Worker], draft: ModelDraft
) -> None:
    """
    Vincolo di copertura minima per turno, ricavato dal ModelDraft.

    Use Case A: tutti i lavoratori sono omogenei.
        Formulazione: ∀d, ∀s: Σ_w x[w,d,s] ≥ min_standard

    Use Case B: lavoratori standard e specializzati, con la regola che uno
        specializzato PUÒ coprire il ruolo standard se necessario.
        Dalla traccia: "a shift may be covered by one standard and two
        specialized workers", quindi un turno valido può avere n_std < min_standard
        purché il totale n_std + n_spe soddisfi la copertura complessiva.

        Formulazione corretta:
            ∀d, ∀s: n_spe[d,s] ≥ min_specialized
            ∀d, ∀s: n_std[d,s] + n_spe[d,s] ≥ min_standard + min_specialized

        La prima garantisce la presenza minima di specializzati nel loro ruolo
        specifico. La seconda garantisce che il totale dei presenti copra
        l'intero fabbisogno, permettendo agli specializzati di riempire
        slot standard quando i lavoratori standard non bastano.

        Esempio (min_standard=2, min_specialized=1):
            - 2 std + 1 spe → n_spe=1 ≥ 1 ✓ | n_std+n_spe=3 ≥ 3 ✓ → valido
            - 1 std + 2 spe → n_spe=2 ≥ 1 ✓ | n_std+n_spe=3 ≥ 3 ✓ → valido (traccia)
            - 0 std + 3 spe → n_spe=3 ≥ 1 ✓ | n_std+n_spe=3 ≥ 3 ✓ → valido
            - 2 std + 0 spe → n_spe=0 < 1 ✗ → non valido (manca lo specializzato)
            - 1 std + 1 spe → n_spe=1 ≥ 1 ✓ | n_std+n_spe=2 < 3 ✗ → non valido

    Scelta progettuale: una sola funzione gestisce entrambi i use case
    leggendo la specifica di copertura dal ModelDraft, evitando
    duplicazione di codice.
    """
    days = _all_days(draft)

    for day in days:
        for cov in draft.coverage:
            s = cov.shift_type

            if cov.min_specialized == 0:
                # ── Use Case A: copertura omogenea ────────────────────────────
                # Tutti i lavoratori sono equivalenti: basta il totale.
                model.add(
                    sum(x[w.worker_id, day, s] for w in workers) >= cov.min_standard
                )
            else:
                # ── Use Case B: copertura eterogenea ──────────────────────────
                std = [w for w in workers if w.role == "standard"]
                spe = [w for w in workers if w.role == "specialized"]

                n_std = sum(x[w.worker_id, day, s] for w in std)
                n_spe = sum(x[w.worker_id, day, s] for w in spe)

                # Vincolo 1: almeno min_specialized specializzati presenti.
                # Gli specializzati hanno competenze che gli standard non possono
                # sostituire, quindi questa soglia non è mai abbassabile.
                model.add(n_spe >= cov.min_specialized)

                # Vincolo 2: il totale dei presenti (std + spe) deve coprire
                # l'intero fabbisogno del turno (min_standard + min_specialized).
                # Questo permette agli specializzati di coprire slot standard
                # quando necessario, come esplicitato dalla traccia.
                #
                # CORREZIONE rispetto alla versione precedente:
                # Il vecchio vincolo `n_std >= min_standard` era troppo restrittivo:
                # escludeva configurazioni valide come (1 std + 2 spe) con
                # min_standard=2, min_specialized=1, che la traccia ammette
                # esplicitamente come esempio valido.
                total_required = cov.min_standard + cov.min_specialized
                model.add(n_std + n_spe >= total_required)