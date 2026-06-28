"""
agents/evaluation_agent.py

Agente di valutazione comparativa: confronta lo schedule prodotto
con preferenze estratte via LLM rispetto a uno prodotto con
preferenze neutre (baseline senza LLM).

Flusso:
    1. Carica preferences.json (già estratto da preference_agent)
    2. Run A: schedule con preferenze reali (da LLM)
    3. Run B: schedule con preferenze neutre (stesso set di lavoratori)
    4. Confronta con metriche oggettive indipendenti dalle preferenze

Perché il confronto è valido:
    - Stessi lavoratori, stesso draft istituzionale
    - Unica variabile: presenza/assenza delle preferenze nell'obiettivo
    - Le metriche oggettive (std notti, Gini) sono confrontabili
      perché non dipendono dalle preferenze dichiarate
"""

import logging
import os
from dataclasses import dataclass

from models.worker import Worker
from models.schedule import Schedule
from solver.fairness import compute_equity_metrics

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Risultato del confronto tra i due run."""
    workers_with_llm:  list[Worker]
    workers_neutral:   list[Worker]
    schedule_with_llm: Schedule
    schedule_neutral:  Schedule
    metrics_with_llm:  dict
    metrics_neutral:   dict


def make_neutral_workers(workers: list[Worker]) -> list[Worker]:
    """
    Crea una copia dei Worker con preferenze neutre.
    Mantiene worker_id e role invariati — solo le preferenze
    soggettive vengono azzerate.
    """
    from agents.preference_agent import make_neutral_worker
    return [Worker.from_dict(make_neutral_worker(w.worker_id, w.role)) for w in workers]


def _run_single(workers: list[Worker], draft, label: str, model_export_path=None):
    """
    Esegue un singolo run completo: solve → verify → refine.
    Restituisce lo schedule finale o solleva eccezione se fallisce.
    """
    from agents.drafting_agent import solve
    from agents.verification_agent import verify, evaluate_fairness
    from agents.refinement_agent import refine

    logger.info(f"[{label}] Avvio run...")
    schedule = solve(workers=workers, draft=draft, model_export_path=model_export_path)
    if schedule is None:
        raise RuntimeError(f"[{label}] Solver non ha trovato soluzioni.")

    is_valid, violations = verify(schedule, draft)
    if not is_valid:
        raise RuntimeError(f"[{label}] Schedule non valido: {violations[:3]}")

    evaluate_fairness(schedule)

    if label == "Con preferenze LLM":
        schedule = refine(schedule, draft)
    else:
        logger.info(f"[{label}] Salto lo Stage 4 (Refinement) per preservare la baseline pura.")

    logger.info(f"[{label}] Completato. Min soddisfazione: {schedule.min_satisfaction():.3f}")
    return schedule


def run_comparison(
    preferences_path: str,
    draft_file: str,
    output_dir: str = "output",
) -> ComparisonResult:
    """
    Esegue il confronto tra Run A (preferenze LLM) e Run B (neutro).

    Precondizione: preferences_path deve esistere già — l'estrazione
    via LLM è responsabilità di preference_agent.extract_and_save(),
    chiamata dal tab Confronto prima di questo metodo.

    Args:
        preferences_path: percorso a preferences.json già estratto
        draft_file:       percorso al model draft istituzionale
        output_dir:       cartella di output per i CSV

    Returns:
        ComparisonResult con schedule e metriche di entrambi i run
    """
    from agents.preference_agent import load_preferences
    from input.model_draft_parser import parse_model_draft

    if not os.path.exists(preferences_path):
        raise FileNotFoundError(
            f"File preferences.json non trovato: {preferences_path}. "
            "Eseguire prima Stage 1 dal tab Confronto."
        )

    workers_llm     = load_preferences(preferences_path)
    workers_neutral = make_neutral_workers(workers_llm)
    draft           = parse_model_draft(draft_file)

    os.makedirs(output_dir, exist_ok=True)

    # Run A — con preferenze LLM
    schedule_llm = _run_single(
        workers=workers_llm,
        draft=draft,
        label="Con preferenze LLM",
        model_export_path=os.path.join(output_dir, "cp_model_partial.txt"),
    )

    # Run B — preferenze neutre (baseline)
    schedule_neutral = _run_single(
        workers=workers_neutral,
        draft=draft,
        label="Baseline senza preferenze",
    )

    metrics_llm = compute_equity_metrics(schedule_llm, workers=workers_llm)
    metrics_neutral = compute_equity_metrics(schedule_neutral, workers=workers_llm)

    logger.info(
        "Confronto completato.\n"
        f"  LLM:    std={metrics_llm['std_nights']:.3f}, Gini={metrics_llm['gini_nights']:.3f}\n"
        f"  Neutro: std={metrics_neutral['std_nights']:.3f}, Gini={metrics_neutral['gini_nights']:.3f}"
    )

    return ComparisonResult(
        workers_with_llm=workers_llm,
        workers_neutral=workers_neutral,
        schedule_with_llm=schedule_llm,
        schedule_neutral=schedule_neutral,
        metrics_with_llm=metrics_llm,
        metrics_neutral=metrics_neutral,
    )