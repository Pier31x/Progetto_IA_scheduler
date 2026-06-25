"""
agents/refinement_agent.py

Stage 4: loop di refinement Maximin.

Il loop ora passa il ModelDraft al Drafting Agent invece del
parametro use_case stringa, rendendo il sistema completamente
parametrizzato dal file di input istituzionale.
"""

import logging
from typing import Optional

from models.schedule import Schedule
from input.model_draft_parser import ModelDraft
from agents.drafting_agent import solve
from agents.verification_agent import verify, evaluate_fairness
from solver.fairness import is_fairness_improvement

logger = logging.getLogger(__name__)
MAX_REFINEMENT_ITERATIONS = 30


def refine(
    initial_schedule: Schedule,
    draft: ModelDraft,
    max_iterations: int = MAX_REFINEMENT_ITERATIONS,
) -> Schedule:
    """
    Loop di refinement Maximin a partire da uno schedule valido.

    Ad ogni iterazione:
        1. Identifica il lavoratore meno soddisfatto (target)
        2. Chiede al Drafting Agent uno schedule con peso amplificato sul target
        3. Verifica hard constraints
        4. Accetta solo se il minimo globale di soddisfazione aumenta
           (criterio che previene oscillazioni)
    """
    current = initial_schedule
    workers = initial_schedule.workers

    if not current.satisfaction_scores:
        evaluate_fairness(current)

    logger.info(f"Refinement avviato. Min iniziale: {current.min_satisfaction():.3f}")

    for iteration in range(1, max_iterations + 1):
        old_scores = dict(current.satisfaction_scores)
        target_id = current.least_satisfied_worker()
        old_global_min = min(old_scores.values())

        logger.info(
            f"[Iter {iteration}] Target: {target_id} "
            f"(score={old_scores[target_id]:.3f}, global_min={old_global_min:.3f})"
        )

        new_schedule = solve(
            workers=workers,
            draft=draft,
            least_satisfied_id=target_id,
            boost_factor=5,
        )

        if new_schedule is None:
            logger.warning(f"[Iter {iteration}] Nessuna soluzione trovata. Stop.")
            break

        is_valid, violations = verify(new_schedule, draft)
        if not is_valid:
            logger.warning(
                f"[Iter {iteration}] Schedule non valido ({len(violations)} violazioni). Stop."
            )
            break

        evaluate_fairness(new_schedule)
        new_scores = new_schedule.satisfaction_scores
        new_global_min = min(new_scores.values())

        if is_fairness_improvement(old_scores, new_scores, target_id):
            if new_global_min <= old_global_min:
                logger.info(
                    f"[Iter {iteration}] Oscillazione rilevata: "
                    f"min globale non migliora ({old_global_min:.3f} → {new_global_min:.3f}). "
                    f"Convergenza."
                )
                break
            logger.info(
                f"[Iter {iteration}] Accettato: {target_id} "
                f"{old_scores[target_id]:.3f}→{new_scores[target_id]:.3f} | "
                f"min globale {old_global_min:.3f}→{new_global_min:.3f}"
            )
            current = new_schedule
        else:
            logger.info(f"[Iter {iteration}] No Maximin improvement. Convergenza.")
            break

    logger.info(f"Refinement completato. Min finale: {current.min_satisfaction():.3f}")
    return current
