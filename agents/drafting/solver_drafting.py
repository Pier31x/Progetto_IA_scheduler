"""
agents/drafting/solver_drafting.py

Wrapper del Drafting Agent basato su OR-Tools CP-SAT.

Obiettivo di fairness
--------------------------
Massimizziamo il punteggio di soddisfazione normalizzato del worker
messo PEGGIO, usando la stessa identica normalizzazione di
`solver/fairness.py` (condivisa via `best_worst_pref_score`), così il
minimo che il solver garantisce ottimo è esattamente il minimo che
viene poi riportato.

La soddisfazione normalizzata di un worker è (raw - n·worst) / (n·(best - worst)), dove `n` (numero
di turni assegnati) VARIA in base al mix notte/giorno scelto (peso 2 vs
1) — non è quindi una funzione lineare fissa, ma un rapporto tra due
espressioni lineari. CP-SAT non supporta divisioni dirette; si aggira
con una variabile condivisa `t` (la soddisfazione minima, scalata a
intero) e, per ogni worker, un vincolo `t · denom_w ≤ SCALE · numer_w`
(algebricamente equivalente a `t ≤ soddisfazione_w · SCALE`), realizzato
con `add_multiplication_equality` per gestire il prodotto tra le due
variabili `t` e `denom_w`. Massimizzando `t` si massimizza esattamente
il minimo — non un'approssimazione.

Per spezzare i pareggi tra le (potenzialmente molte) soluzioni
ugualmente maximin-ottime, l'obiettivo è lessicografico su tre livelli,
ottenuto con pesi (`LEXICO_M1`, `LEXICO_M2`) abbastanza grandi da
garantire che un livello domini sempre completamente quello successivo:

    1. t                          (il minimo — priorità assoluta)
    2. Σ_w numer_w                (proxy della somma totale, per
                                    scegliere tra soluzioni maximin-tied
                                    quella complessivamente migliore)
    3. numer_target                (se `least_satisfied_id` è dato, un
                                    lieve favore verso quel worker
                                    specifico tra i pareggi residui)
"""

import logging
from typing import List, Optional
from datetime import date

from ortools.sat.python import cp_model

from models.worker import Worker
from models.schedule import Schedule
from models.assignment import Assignment
from models.shift import Shift
from input.model_draft_parser import ModelDraft
from solver.model_builder import build_model, extract_schedule_from_solution
from solver.fairness import pref_score, best_worst_pref_score

logger = logging.getLogger(__name__)

SOLVER_TIMEOUT_SECONDS = 1800

# Precisione di scaling per l'aritmetica intera di CP-SAT.
PREF_SCALE = 1000   # preserva 3 decimali di pref_score (in [-1, 1])
SAT_SCALE = 1000    # preserva 3 decimali della soddisfazione normalizzata

# Pesi per l'obiettivo lessicografico: ognuno deve dominare
# completamente la somma di tutto ciò che sta sotto di lui in priorità.
LEXICO_M2 = 200_000       # domina il termine di boost per least_satisfied_id
LEXICO_M1 = 10 ** 12      # domina l'intero termine di pareggio (somma totale)


def _has_real_preferences(workers: List[Worker]) -> bool:
    return any(
        w.preferred_shifts or w.avoid_shifts or w.night_tolerance != 0.5
        for w in workers
    )


def _add_maximin_fairness_objective(
    model: cp_model.CpModel,
    x: dict,
    workers: List[Worker],
    days: List[date],
    draft: ModelDraft,
    least_satisfied_id: Optional[str] = None,
    boost_factor: int = 3,
) -> None:
    """
    Costruisce un vero obiettivo MAXIMIN (vedi docstring di modulo per
    la derivazione matematica completa). Se nessun worker ha preferenze
    reali e nessun target è specificato, non aggiunge nulla (nessuna
    fairness da ottimizzare — comportamento invariato per lo Scenario A
    con worker neutri).
    """

    if not _has_real_preferences(workers) and least_satisfied_id is None:
        return

    shift_types = draft.shift_names
    n_days = len(days)

    t = model.new_int_var(0, SAT_SCALE, "fairness_min_t")

    secondary_terms = []   # Σ numer_w — proxy della somma totale (tie-break #2)
    tertiary_terms = []    # numer_w del target — tie-break #3

    for worker in workers:

        best_w, worst_w = best_worst_pref_score(worker)
        best_scaled = round(best_w * PREF_SCALE)
        worst_scaled = round(worst_w * PREF_SCALE)

        if best_scaled == worst_scaled:
            # Assegna il massimo della scala (1.0)
            model.add(t <= SAT_SCALE)
            secondary_terms.append(SAT_SCALE)
            continue

        range_scaled = best_scaled - worst_scaled  # > 0

        # n_w = numero di turni assegnati (varia col mix notte/giorno)
        n_w = model.new_int_var(0, n_days, f"n_{worker.worker_id}")
        model.add(
            n_w == sum(
                x[(worker.worker_id, day, s)]
                for day in days
                for s in shift_types
            )
        )

        # raw_w = Σ pref_score_scaled * x[w,d,s]  (lineare)
        raw_terms = [
            round(pref_score(worker, day, s) * PREF_SCALE) * x[(worker.worker_id, day, s)]
            for day in days
            for s in shift_types
            if round(pref_score(worker, day, s) * PREF_SCALE) != 0
        ]
        raw_w = sum(raw_terms) if raw_terms else 0

        # numer_w = raw_w - worst_scaled * n_w        (lineare)
        numer_w = raw_w - worst_scaled * n_w

        # denom_w = range_scaled * n_w                (lineare, va
        # materializzato come variabile per il prodotto con t)
        max_denom = range_scaled * n_days
        denom_w = model.new_int_var(0, max_denom, f"denom_{worker.worker_id}")
        model.add(denom_w == range_scaled * n_w)

        # t <= SAT_SCALE * numer_w / denom_w
        #   <=> t * denom_w <= SAT_SCALE * numer_w
        prod_w = model.new_int_var(0, SAT_SCALE * max_denom, f"prod_{worker.worker_id}")
        model.add_multiplication_equality(prod_w, [t, denom_w])
        model.add(prod_w <= SAT_SCALE * numer_w)

        secondary_terms.append(numer_w)

        if worker.worker_id == least_satisfied_id:
            tertiary_terms.append(numer_w * boost_factor)

    model.maximize(
        LEXICO_M1 * t
        + LEXICO_M2 * sum(secondary_terms)
        + sum(tertiary_terms)
    )


def _violation_count(violation_feedback) -> Optional[int]:
    """
    Restituisce un conteggio delle violazioni da loggare, sia che
    `violation_feedback` sia il vecchio `list[str]` sia il nuovo
    `VerificationReport` (che espone `total_violations` ma non ha
    `__len__`). Ritorna None se il conteggio non è disponibile.
    """
    if violation_feedback is None:
        return None

    total = getattr(violation_feedback, "total_violations", None)
    if total is not None:
        return total

    try:
        return len(violation_feedback)
    except TypeError:
        return None


class SolverDraftingAgent:
    """
    Adapter class per uniformare interfaccia con LLM agent.
    """

    def solve(
        self,
        workers: List[Worker],
        draft: ModelDraft,
        least_satisfied_id: Optional[str] = None,
        boost_factor: int = 3,
        violation_feedback: Optional[object] = None,
        model_export_path: Optional[str] = None,
        pinned_assignments: Optional[List[Assignment]] = None,
        timeout_seconds: int = 60,
    ) -> Optional[Schedule]:
        """
        `pinned_assignments`: assegnazioni da forzare come vincoli di
        uguaglianza nel modello CP-SAT (tipicamente proposte da un
        Drafting Agent LLM a monte). Il solver applica comunque TUTTI
        i vincoli hard anche su queste variabili, e ottimizza comunque
        il vero obiettivo maximin su TUTTA la schedule.
        """

        if violation_feedback:
            n_violations = _violation_count(violation_feedback)
            if n_violations is not None:
                logger.warning(f"[Solver] Retry con {n_violations} violazioni")
            else:
                logger.warning("[Solver] Retry con violazioni (conteggio non disponibile)")

        model, x, days = build_model(workers, draft, pinned_assignments=pinned_assignments)

        if model_export_path:
            from solver.model_builder import export_model_summary
            export_model_summary(workers, draft, x, days, model_export_path)
            logger.info(f"Modello esportato in: {model_export_path}")

        _add_maximin_fairness_objective(
            model,
            x,
            workers,
            days,
            draft,
            least_satisfied_id,
            boost_factor,
        )

        solver = cp_model.CpSolver()
        #solver.parameters.max_time_in_seconds = SOLVER_TIMEOUT_SECONDS
        # TEMPORANEEEE
        solver.parameters.log_search_progress = True
        solver.parameters.cp_model_presolve = True
        solver.parameters.linearization_level = 2
        # L'obiettivo maximin esatto usa add_multiplication_equality (un
        # vincolo per worker): matematicamente corretto, ma con un
        # rilassamento LP più debole di una somma pesata — il solver
        # trova spesso una buona soluzione in fretta ma fatica a
        # DIMOSTRARNE l'ottimalità entro pochi secondi. Verificato
        # empiricamente a scala piena (13 worker, 30 giorni): con 30s si
        # ottiene FEASIBLE (non OPTIMAL certificato) ma con valori
        # stabili anche raddoppiando il timeout — quindi la soluzione è
        # verosimilmente già ottima o vicinissima, solo non certificata.
        solver.parameters.max_time_in_seconds = timeout_seconds
        # TEMPORANEEEE
        status = solver.solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):

            status_label = (
                "OPTIMAL (ottimalità dimostrata)"
                if status == cp_model.OPTIMAL
                else "FEASIBLE (timeout raggiunto, non è detto sia l'ottimo globale)"
            )
            try:
                objective_str = str(solver.objective_value)
            except Exception:
                objective_str = "N/A (nessun obiettivo impostato)"
            logger.info(
                f"[Solver] Stato: {status_label} — "
                f"tempo: {solver.wall_time:.1f}s, "
                f"obiettivo: {objective_str}"
            )

            assignments = extract_schedule_from_solution(
                workers, x, days, draft.shift_names, solver
            )

            all_shifts = [
                Shift(day=d, shift_type=s)
                for d in days
                for s in draft.shift_names
            ]

            return Schedule(assignments=assignments, workers=workers, shifts=all_shifts)

        elif status == cp_model.INFEASIBLE:
            logger.error(
                "[Solver] Modello INFEASIBLE"
                + (
                    f" ({len(pinned_assignments)} assegnazioni bloccate: "
                    "potrebbero essere incompatibili tra loro o con i vincoli hard)"
                    if pinned_assignments
                    else ""
                )
            )

        else:
            logger.warning("Timeout solver")

        return None