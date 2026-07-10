"""
agents/evaluation_agent.py

Agente di valutazione comparativa a TRE scenari:

    Scenario A — solo vincoli hard (nessuna preferenza nell'obiettivo)
    Scenario B — vincoli hard + preferenze + Drafting Agent OR-Tools
    Scenario C — vincoli hard + preferenze + Drafting Agent LLM

Perché il confronto è valido
-----------------------------
- L'estrazione delle preferenze avviene UNA SOLA VOLTA
  (`load_preferences(preferences_path)`), non una volta per scenario.
- Gli STESSI oggetti Worker (con le preferenze realmente estratte)
  vengono usati per misurare la fairness in TUTTI e tre gli scenari,
  indipendentemente da quali oggetti abbiano effettivamente guidato
  il solver in quello scenario. Questo è ciò che rende le metriche
  comparabili: cambia solo la strategia di drafting, non il metro con
  cui la valutiamo.
- Nello Scenario A il solver riceve dei worker "neutri" (preferenze
  azzerate) come input — perché quello scenario per definizione NON
  deve ottimizzare le preferenze — ma la fairness/preference
  satisfaction viene comunque calcolata contro le preferenze reali,
  per mostrare quanto lo scenario "cieco" le soddisfi per puro caso.
"""

from __future__ import annotations

import logging
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from models.worker import Worker
from models.schedule import Schedule
from solver.fairness import compute_equity_metrics

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Risultati
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    """Risultato di un singolo scenario del confronto."""
    label: str
    schedule: Schedule
    metrics: Dict[str, object] = field(default_factory=dict)


@dataclass
class ThreeWayComparisonResult:
    """Risultato completo del confronto A / B / C."""
    workers: List[Worker]              # le preferenze reali, estratte UNA volta
    scenario_a: ScenarioResult          # solo vincoli hard
    scenario_b: ScenarioResult          # vincoli + preferenze + OR-Tools
    scenario_c: ScenarioResult          # vincoli + preferenze + LLM


# ──────────────────────────────────────────────────────────────────────
# Worker neutri (per lo Scenario A)
# ──────────────────────────────────────────────────────────────────────

def make_neutral_workers(workers: List[Worker]) -> List[Worker]:
    """
    Crea una copia dei Worker con preferenze neutre.
    Mantiene worker_id e role invariati — solo le preferenze
    soggettive vengono azzerate.
    """
    from agents.preference_agent import make_neutral_worker
    return [Worker.from_dict(make_neutral_worker(w.worker_id, w.role)) for w in workers]


# ──────────────────────────────────────────────────────────────────────
# Metriche di fairness
# ──────────────────────────────────────────────────────────────────────


def _preference_satisfaction_percentage(workers: List[Worker], schedule: Schedule) -> float:
    """
    Percentuale di lavoratori che hanno ottenuto un orario mensile di alta qualità
    (con uno score di soddisfazione normalizzato complessivo >= (SOGLIA_SODDISFAZIONE*100)%).
    """
    SOGLIA_SODDISFAZIONE = 0.7
    if not hasattr(schedule, 'satisfaction_scores') or not schedule.satisfaction_scores:
        return 0.0

    satisfied = sum(1 for score in schedule.satisfaction_scores.values() if score >= SOGLIA_SODDISFAZIONE)
    return round(100.0 * satisfied / len(workers), 1)


def _fairness_metrics(schedule: Schedule, measure_workers: List[Worker]) -> Dict[str, object]:
    """
    Calcola l'intero pacchetto di metriche di fairness essenziali per uno scenario,
    misurate SEMPRE contro `measure_workers` — gli oggetti Worker con le
    preferenze realmente estratte.
    """
    import statistics
    from agents.verification_agent import evaluate_fairness

    # Rendiamo esplicito che la fairness va misurata sui worker "veri":
    # gli assignment referenziano worker_id (stringhe), quindi sostituire
    # la lista di oggetti Worker associata alla schedule è sicuro.
    schedule.workers = measure_workers

    scores = evaluate_fairness(schedule)
    values = list(scores.values())

    # Recupera le metriche semplificate e ottimizzate (Notti, Festivi, Straordinari)
    equity = compute_equity_metrics(schedule, workers=measure_workers)

    return {
        # 1. Soddisfazione (Soggettiva)
        "min_satisfaction": equity.get("min_satisfaction", 0.0),
        "avg_satisfaction": equity.get("avg_satisfaction", 0.0),
        "preference_satisfaction_pct": _preference_satisfaction_percentage(
            measure_workers, schedule
        ),
        "least_satisfied_worker": schedule.least_satisfied_worker(),

        # 2. Distribuzione Notti (Oggettivo)
        "min_nights": equity.get("min_nights", 0),
        "max_nights": equity.get("max_nights", 0),
        "discrepancy_nights": equity.get("discrepancy_nights", 0),
        "night_distribution": equity.get("night_counts", {}),  # Per i grafici/tabelle puntuali per ID

        # 3. Distribuzione Festivi (Oggettivo)
        "min_holidays": equity.get("min_holidays", 0),
        "max_holidays": equity.get("max_holidays", 0),
        "holiday_distribution": equity.get("holiday_counts", {}),

        # 4. Gestione Straordinari ed Emergenze (Efficacia)
        "total_overtime_shifts": equity.get("total_overtime_shifts", 0),
        "emergency_alignment": equity.get("emergency_alignment", 1.0),
        "overtime_distribution": equity.get("overtime_counts", {}),
    }



# ──────────────────────────────────────────────────────────────────────
# Esecuzione di un singolo scenario
# ──────────────────────────────────────────────────────────────────────

def _run_scenario(
        label: str,
        solve_workers: List[Worker],
        measure_workers: List[Worker],
        draft,
        drafting_agent,
        apply_refinement: bool,
        model_export_path: Optional[str] = None,
) -> ScenarioResult:
    """
    Esegue un singolo scenario con ciclo di reiterazione basato su feedback:
    solve → verify → retry (fino a 3 volte se illegale) → (refine) → fairness.
    """
    from agents.verification_agent import verify, evaluate_fairness
    from agents.refinement_agent import refine

    logger.info(f"[Evaluation Agent] {label}: avvio...")

    schedule = None
    violation_feedback = None
    MAX_RETRIES = 3

    # ── CICLO DI REITERAZIONE / REPAIR ────────────────────────────────────────
    for attempt in range(1, MAX_RETRIES + 1):
        logger.info(f"[{label}] Generazione schedule (tentativo {attempt}/{MAX_RETRIES})...")

        # Chiamiamo l'agente passando anche il feedback delle violazioni del giro prima
        # Nota: assicurati che il metodo .solve dell'agente accetti 'violation_feedback'
        schedule = drafting_agent.solve(
            workers=solve_workers,
            draft=draft,
            model_export_path=model_export_path if attempt == 1 else None,
            violation_feedback=violation_feedback
        )

        if schedule is None:
            raise RuntimeError(f"[{label}] Il Drafting Agent non ha prodotto una schedule al tentativo {attempt}.")

        # Verifica legale (Vincoli Hard)
        report = verify(schedule, draft)

        if report.feasible:
            logger.info(f"✅ [{label}] Soluzione ammissibile trovata al tentativo {attempt}.")
            break
        else:
            logger.warning(
                f"⚠️ [{label}] Tentativo {attempt} fallito: {report.total_violations} violazioni legali. "
                f"Invio feedback all'agente per la reiterazione."
            )
            # Salviamo le violazioni per darle in pasto all'agente al prossimo ciclo
            violation_feedback = report.all_violations

    # Se dopo 3 giri siamo ancora fuori legge, interrompiamo tutto
    if schedule is None or not verify(schedule, draft).feasible:
        raise RuntimeError(f"❌ [{label}] Impossibile ottenere uno schedule legale dopo {MAX_RETRIES} tentativi.")

    # ── FASE DI REFINEMENT (EQUITA') ──────────────────────────────────────────
    if apply_refinement:
        schedule = refine(schedule, drafting_agent, draft)
    else:
        logger.info(f"[{label}] Refinement (Stage 4) saltato: scenario a solo vincoli hard.")
        evaluate_fairness(schedule)

    # Calcolo metriche per i widget di app.py
    metrics = _fairness_metrics(schedule, measure_workers)

    # Log finale pulito e corretto (senza il bug di std che copiava il min)
    logger.info(
        f"[{label}] Completato — Sat Media: {metrics['avg_satisfaction']:.3f} | "
        f"Sat Minima: {metrics['min_satisfaction']:.3f} | "
        f"Allineamento Straordinari: {metrics.get('emergency_alignment', 1.0) * 100:.1f}%"
    )

    return ScenarioResult(label=label, schedule=schedule, metrics=metrics)


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def run_three_way_comparison(
    preferences_path: str,
    draft_file: str,
    output_dir: str = "output",
) -> ThreeWayComparisonResult:
    """
    Esegue il confronto A / B / C.

    Precondizione: `preferences_path` deve esistere già — l'estrazione
    via LLM è responsabilità di `preference_agent.extract_and_save()`,
    chiamata dal tab Stage 1 prima di questo metodo.

    Args:
        preferences_path: percorso a preferences.json già estratto
        draft_file:       percorso al model draft istituzionale
        output_dir:       cartella di output per gli export

    Returns:
        ThreeWayComparisonResult con schedule e metriche dei tre scenari
    """
    from agents.preference_agent import load_preferences
    from input.model_draft_parser import parse_model_draft
    from agents.drafting.solver_drafting import SolverDraftingAgent
    from agents.drafting.llm_drafting import LLMDraftingAgent

    if not os.path.exists(preferences_path):
        raise FileNotFoundError(
            f"File preferences.json non trovato: {preferences_path}. "
            "Eseguire prima Stage 1 dal tab Confronto."
        )

    # ── Estrazione preferenze — UNA SOLA VOLTA ──────────────────────
    workers = load_preferences(preferences_path)
    neutral_workers = make_neutral_workers(workers)
    draft = parse_model_draft(draft_file)

    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        f"[Evaluation Agent] {len(workers)} preferenze riutilizzate per tutti e tre gli scenari."
    )

    # ── Scenario A — solo vincoli hard ──────────────────────────────
    scenario_a = _run_scenario(
        label="Scenario A — solo vincoli hard",
        solve_workers=neutral_workers,
        measure_workers=workers,
        draft=draft,
        drafting_agent=SolverDraftingAgent(),
        apply_refinement=False,
    )

    # ── Scenario B — vincoli + preferenze + OR-Tools ────────────────
    scenario_b = _run_scenario(
        label="Scenario B — vincoli + preferenze + OR-Tools",
        solve_workers=workers,
        measure_workers=workers,
        draft=draft,
        drafting_agent=SolverDraftingAgent(),
        apply_refinement=True,
    )

    # ── Scenario C — vincoli + preferenze + LLM ─────────────────────
    scenario_c = _run_scenario(
        label="Scenario C — vincoli + preferenze + LLM",
        solve_workers=workers,
        measure_workers=workers,
        draft=draft,
        drafting_agent=LLMDraftingAgent(),
        apply_refinement=False,
        model_export_path=os.path.join(output_dir, "cp_model_partial.txt"),
    )

    logger.info(
        "[Evaluation Agent] Confronto a tre scenari completato.\n"
        f"  A: min={scenario_a.metrics['min_satisfaction']:.3f} "
        f"  B: min={scenario_b.metrics['min_satisfaction']:.3f} "
        f"  C: min={scenario_c.metrics['min_satisfaction']:.3f} "
    )

    return ThreeWayComparisonResult(
        workers=workers,
        scenario_a=scenario_a,
        scenario_b=scenario_b,
        scenario_c=scenario_c,
    )