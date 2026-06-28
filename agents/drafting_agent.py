"""
agents/drafting_agent.py

Stage 2: costruisce e risolve il modello CP-SAT.

Scelta progettuale: il Drafting Agent riceve ora il ModelDraft come
parametro invece di usare costanti globali. In questo modo il
comportamento del solver è completamente determinato dal file di input
istituzionale — cambiare scenario significa cambiare file, non codice.

Gestione del fallback: se la verifica (Stage 3) rileva violazioni,
il main.py può richiamare questo agente con un messaggio di feedback.
In questa versione, il feedback viene loggato — in un sistema completo
con LLM generatore di codice, verrebbe incluso nel prompt di retry.
"""

import logging
from datetime import date
from typing import List, Optional

from ortools.sat.python import cp_model

from models.worker import Worker
from models.schedule import Schedule
from models.shift import Shift
from input.model_draft_parser import ModelDraft
from solver.model_builder import build_model, extract_schedule_from_solution
from solver.fairness import pref_score

logger = logging.getLogger(__name__)

SOLVER_TIMEOUT_SECONDS = 180  # aumentato per istanze più grandi (13+ lavoratori)


def _has_real_preferences(workers: List[Worker]) -> bool:
    """
    Restituisce True se almeno un lavoratore ha preferenze non neutre.

    Scelta progettuale: con Worker tutti neutri (preferred_shifts=[],
    avoid_shifts=[], night_tolerance=0.5), pref_score produce -0.5
    per ogni notte — non per una preferenza dichiarata ma come
    artefatto della formula con tolleranza di default. Aggiungere
    questo termine all'obiettivo non rappresenta alcuna preferenza
    reale e distorce il confronto.

    Quando tutti i Worker sono neutri e non siamo in refinement,
    omettiamo l'obiettivo completamente: il solver trova una soluzione
    ammissibile basata sui soli vincoli hard. Questo è il Run B corretto
    nel confronto LLM vs neutro — scheduling puro senza soft constraint.
    """
    return any(
        w.preferred_shifts or w.avoid_shifts or w.night_tolerance != 0.5
        for w in workers
    )


def _add_preference_objective(
        model: cp_model.CpModel,
        x: dict,
        workers: List[Worker],
        days: List[date],
        draft: ModelDraft,
        least_satisfied_id: Optional[str] = None,
        boost_factor: int = 3,
) -> None:
    """
    Aggiunge l'obiettivo di massimizzazione della soddisfazione.

    Modalità Stage 2 (least_satisfied_id is None):
        Ottimizzazione globale — l'obiettivo è la somma pesata dei
        pref_score di tutti i lavoratori.

        Formulazione: maximize Σ_{w,d,s} pref_score(w,d,s) * x[w,d,s]

    Modalità Stage 4 — Refinement Maximin (least_satisfied_id is not None):
        L'obiettivo include tutti i lavoratori, ma i termini del lavoratore
        target vengono moltiplicati per boost_factor. Questo spinge il solver
        a migliorare il lavoratore meno soddisfatto senza ignorare gli altri,
        in linea con la traccia: "by considering the preferences and associated
        priorities declared by all workers".

        Formulazione:
            maximize Σ_{w≠target, d, s} pref_score(w,d,s) * x[w,d,s]
                   + Σ_{d, s} boost_factor * pref_score(target,d,s) * x[target,d,s]

        Perché non massimizzare solo il target (come nella versione precedente):
        Massimizzare unicamente il target porta il solver a trovare schedule
        che massimizzano la soddisfazione di un solo lavoratore spesso a
        scapito degli altri, perché tutti gli altri termini hanno peso zero
        e il solver è libero di ignorarli. Con il boost, il target ha
        priorità ma gli altri lavoratori rimangono nell'obiettivo con
        peso 1, così il solver cerca un equilibrio.

        CORREZIONE rispetto alla versione precedente:
        La vecchia implementazione riceveva boost_factor ma non lo usava mai
        (parametro dead). Inoltre massimizzava solo il target, escludendo
        dal calcolo le preferenze di tutti gli altri lavoratori.
    """
    if not _has_real_preferences(workers) and least_satisfied_id is None:
        return  # Baseline pura senza soft constraint

    shift_types = draft.shift_names

    # ── SCENARIO A: REFINEMENT MAXIMIN (STAGE 4) ──────────────────────────────
    if least_satisfied_id is not None:
        # Pre-fetch del lavoratore target per evitare una ricerca O(n)
        # ad ogni iterazione del doppio loop (days × shifts).
        # La versione precedente eseguiva next(w for w in workers ...) dentro
        # il loop, risultando in O(n * |days| * |shifts|) ricerche totali.
        target_worker = next(
            (w for w in workers if w.worker_id == least_satisfied_id), None
        )
        if target_worker is None:
            logger.warning(
                f"[Refinement] Worker target '{least_satisfied_id}' non trovato. "
                "Fallback su obiettivo globale senza boost."
            )
            least_satisfied_id = None  # ricade nello Scenario B

        else:
            objective_terms = []
            for worker in workers:
                # Il peso di ogni lavoratore nell'obiettivo:
                #   - boost_factor per il target (lavoratore meno soddisfatto)
                #   - 1 per tutti gli altri (rimangono nell'obiettivo)
                weight = boost_factor if worker.worker_id == least_satisfied_id else 1

                for day in days:
                    for s in shift_types:
                        # pref_score ∈ [-1.0, +1.0]; moltiplichiamo per 100
                        # per lavorare con interi (CP-SAT richiede coefficienti interi).
                        score_int = int(pref_score(worker, day, s) * 100)
                        if score_int != 0:
                            objective_terms.append(
                                weight * score_int * x[(worker.worker_id, day, s)]
                            )

            if objective_terms:
                model.maximize(sum(objective_terms))
            return

    # ── SCENARIO B: OTTIMIZZAZIONE GLOBALE INIZIALE (STAGE 2) ─────────────────
    # Se least_satisfied_id era None dall'inizio, o è stato azzerato
    # per target non trovato, eseguiamo l'ottimizzazione globale standard.
    objective_terms = []
    for worker in workers:
        for day in days:
            for s in shift_types:
                score_int = int(pref_score(worker, day, s) * 100)
                if score_int != 0:
                    objective_terms.append(
                        score_int * x[(worker.worker_id, day, s)]
                    )

    if objective_terms:
        model.maximize(sum(objective_terms))


def solve(
    workers: List[Worker],
    draft: ModelDraft,
    least_satisfied_id: Optional[str] = None,
    boost_factor: int = 3,
    violation_feedback: Optional[List[str]] = None,
    model_export_path: Optional[str] = None,
) -> Optional[Schedule]:
    """
    Costruisce e risolve il modello CP-SAT.

    Args:
        workers: lavoratori con preferenze formalizzate
        draft: model draft istituzionale
        least_satisfied_id: in refinement, ID del lavoratore target
        boost_factor: peso amplificato per il lavoratore target.
            Con boost_factor=5 (usato dal Refinement Agent), i termini
            del target pesano 5x rispetto agli altri nell'obiettivo.
        violation_feedback: lista di violazioni dal Verification Agent
            (loggate per debugging; in un sistema LLM full verrebbero
            incluse nel prompt di retry al modello)
        model_export_path: se fornito, esporta il modello parziale in
            questo percorso (deliverable richiesto dalla traccia)

    Returns:
        Schedule se trovata una soluzione, None altrimenti.
    """
    if violation_feedback:
        logger.warning(
            f"Retry con {len(violation_feedback)} violazioni dal ciclo precedente:"
        )
        for v in violation_feedback[:5]:
            logger.warning(f"  • {v}")

    model, x, days = build_model(workers, draft)

    # Export del modello parziale (deliverable richiesto dalla traccia)
    if model_export_path:
        from solver.model_builder import export_model_summary
        export_model_summary(workers, draft, x, days, model_export_path)
        logger.info(f"Modello CP-SAT parziale esportato in: {model_export_path}")

    _add_preference_objective(model, x, workers, days, draft, least_satisfied_id, boost_factor)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = SOLVER_TIMEOUT_SECONDS
    solver.parameters.log_search_progress = False

    status = solver.solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        label = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
        logger.info(f"Soluzione trovata ({label}), obiettivo={solver.objective_value:.1f}")

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
        logger.error("Modello INFEASIBLE: nessuna soluzione esiste con i vincoli dati.")
    else:
        logger.warning("Timeout del solver.")

    return None