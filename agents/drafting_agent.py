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
    Aggiunge l'obiettivo di massimizzazione della soddisfazione pesata.

    In fase di refinement, il lavoratore target riceve peso amplificato
    per spingere il solver a privilegiarlo nell'allocazione.
    I float sono scalati x100 perché OR-Tools lavora con interi.
    """
    shift_types = draft.shift_names
    objective_terms = []

    for worker in workers:
        weight = boost_factor if worker.worker_id == least_satisfied_id else 1
        for day in days:
            for s in shift_types:
                score_int = int(pref_score(worker, day, s) * 100)
                if score_int != 0:
                    objective_terms.append(
                        score_int * weight * x[(worker.worker_id, day, s)]
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
        boost_factor: peso amplificato per il lavoratore target
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
