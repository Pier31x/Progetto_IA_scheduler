"""
main.py

Entry point di SmartScheduler.

Flusso completo:
    1. Legge il model draft istituzionale (file .txt)
    2. Legge le preferenze dei lavoratori (file .json)
    3. Stage 1 — Preference Agent: NL → Worker objects
    4. Stage 2 — Drafting Agent: OR-Tools solve (+ export modello parziale)
    5. Stage 3 — Verification Agent: verifica hard + fairness
       → se fallisce: retry al Drafting Agent con feedback (max MAX_DRAFT_RETRIES)
    6. Stage 4 — Refinement Agent: loop Maximin
    7. Output: stampa a schermo + CSV

Uso:
    python main.py --use-case A --fallback
    python main.py --use-case B
    python main.py --draft input/model_draft_use_case_a.txt --workers input/workers_use_case_a.json
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from input.model_draft_parser import parse_model_draft
from agents.preference_agent import load_and_extract
from agents.drafting_agent import solve
from agents.verification_agent import verify, evaluate_fairness
from agents.refinement_agent import refine
from output.schedule_output import print_schedule, export_to_csv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

MAX_DRAFT_RETRIES = 3  # tentativi massimi se la verifica fallisce


def run(
    draft_file: str,
    workers_file: str,
    force_fallback: bool = False,
) -> None:

    os.makedirs("output", exist_ok=True)

    # ── Lettura input istituzionale ────────────────────────────────────────────
    logger.info(f"Lettura model draft: {draft_file}")
    draft = parse_model_draft(draft_file)
    logger.info(
        f"Use Case {draft.use_case} | "
        f"Lavoratori: {draft.workforce.total_workers} | "
        f"Orizzonte: {draft.start_date} → {draft.end_date}"
    )

    # ── Stage 1: Preference Agent ──────────────────────────────────────────────
    logger.info("=== Stage 1: Raccolta e formalizzazione preferenze ===")
    workers, _ = load_and_extract(workers_file, force_fallback=force_fallback)
    logger.info(f"Preferenze estratte per {len(workers)} lavoratori.")
    for w in workers:
        logger.info(
            f"  {w.worker_id} ({w.role}): prefer={w.preferred_shifts}, "
            f"avoid={w.avoid_shifts}, night_tol={w.night_tolerance:.1f}, "
            f"days_off={w.preferred_days_off}"
        )

    # ── Stage 2 + 3: Drafting → Verification loop ─────────────────────────────
    schedule = None
    violation_feedback = None

    for attempt in range(1, MAX_DRAFT_RETRIES + 1):
        logger.info(f"=== Stage 2: Drafting (tentativo {attempt}/{MAX_DRAFT_RETRIES}) ===")

        # Esporta il modello parziale al primo tentativo
        model_export = "output/cp_model_partial.txt" if attempt == 1 else None

        schedule = solve(
            workers=workers,
            draft=draft,
            violation_feedback=violation_feedback,
            model_export_path=model_export,
        )

        if schedule is None:
            logger.error("Drafting Agent non ha trovato soluzioni. Uscita.")
            sys.exit(1)

        logger.info(f"=== Stage 3: Verifica (tentativo {attempt}) ===")
        is_valid, violations = verify(schedule, draft)

        if is_valid:
            logger.info("Schedule valido — tutti i vincoli hard soddisfatti.")
            break
        else:
            logger.warning(
                f"Schedule non valido: {len(violations)} violazioni. "
                f"{'Retry.' if attempt < MAX_DRAFT_RETRIES else 'Tentativi esauriti.'}"
            )
            for v in violations[:5]:
                logger.warning(f"  • {v}")
            violation_feedback = violations
            schedule = None

    if schedule is None:
        logger.error(
            f"Impossibile generare uno schedule valido dopo {MAX_DRAFT_RETRIES} tentativi."
        )
        sys.exit(1)

    evaluate_fairness(schedule)
    logger.info(
        f"Soddisfazione iniziale: min={schedule.min_satisfaction():.3f}, "
        f"worker penalizzato: {schedule.least_satisfied_worker()}"
    )

    # ── Stage 4: Refinement Agent ──────────────────────────────────────────────
    logger.info("=== Stage 4: Refinement Maximin ===")
    final_schedule = refine(schedule, draft)

    # ── Output ─────────────────────────────────────────────────────────────────
    print_schedule(final_schedule)
    csv_path = f"output/schedule_use_case_{draft.use_case}.csv"
    export_to_csv(final_schedule, csv_path)
    logger.info(f"Schedule salvato in: {csv_path}")
    logger.info("Modello CP-SAT parziale salvato in: output/cp_model_partial.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartScheduler")
    parser.add_argument("--use-case", choices=["A", "B"], default="A")
    parser.add_argument("--draft", default=None, help="Percorso al model draft .txt")
    parser.add_argument("--workers", default=None, help="Percorso al file workers .json")
    parser.add_argument("--fallback", action="store_true",
                        help="Usa parser rule-based invece di LLaMA")
    args = parser.parse_args()

    uc = args.use_case.upper()
    draft_file   = args.draft   or f"input/model_draft_use_case_{uc.lower()}.txt"
    workers_file = args.workers or f"input/workers_use_case_{uc.lower()}.json"

    run(draft_file=draft_file, workers_file=workers_file, force_fallback=args.fallback)
