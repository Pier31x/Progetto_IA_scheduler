"""
main.py

Entry point CLI di SmartScheduler.

Flusso:
    1. Legge model draft (.txt) e workers (.json)
    2. Stage 1: estrae preferenze via LLM e le salva in preferences.json
       (se preferences.json esiste già, lo riusa senza chiamare l'LLM)
    3. Stage 2: drafting OR-Tools CP-SAT
    4. Stage 3: verifica hard constraints + fairness
    5. Stage 4: refinement Maximin
    6. Output: stampa + CSV

Uso:
    python main.py --use-case A
    python main.py --use-case B
    python main.py --use-case A --reextract   # forza ri-estrazione via LLM
    python main.py --use-case A --fallback    # preferenze neutre (baseline)
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from input.model_draft_parser import parse_model_draft
from agents.preference_agent import extract_and_save, load_preferences
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

MAX_DRAFT_RETRIES = 3
PREFERENCES_PATH  = "output/preferences.json"


def run(
    draft_file: str,
    workers_file: str,
    force_fallback: bool = False,
    force_reextract: bool = False,
) -> None:

    os.makedirs("output", exist_ok=True)

    # ── Lettura draft istituzionale ────────────────────────────────────────────
    logger.info(f"Lettura model draft: {draft_file}")
    draft = parse_model_draft(draft_file)
    logger.info(
        f"Use Case {draft.use_case} | "
        f"Lavoratori: {draft.workforce.total_workers} | "
        f"Orizzonte: {draft.start_date} → {draft.end_date}"
    )

    # ── Stage 1: preferenze ────────────────────────────────────────────────────
    # Se preferences.json esiste e non si forza la ri-estrazione, lo riusiamo.
    # Questo evita di chiamare LLaMA inutilmente nelle run successive.
    if os.path.exists(PREFERENCES_PATH) and not force_reextract and not force_fallback:
        logger.info(
            f"=== Stage 1: Carico preferenze da {PREFERENCES_PATH} "
            f"(usa --reextract per riestrarre via LLM) ==="
        )
        workers = load_preferences(PREFERENCES_PATH)
    else:
        logger.info("=== Stage 1: Estrazione preferenze via LLM ===")
        workers = extract_and_save(
            workers_file=workers_file,
            preferences_path=PREFERENCES_PATH,
            force_fallback=force_fallback,
        )

    for w in workers:
        logger.info(
            f"  {w.worker_id} ({w.role}): prefer={w.preferred_shifts}, "
            f"avoid={w.avoid_shifts}, night_tol={w.night_tolerance:.1f}"
        )

    # ── Stage 2 + 3: Drafting → Verification con retry ────────────────────────
    schedule = None
    violation_feedback = None

    for attempt in range(1, MAX_DRAFT_RETRIES + 1):
        logger.info(f"=== Stage 2: Drafting (tentativo {attempt}/{MAX_DRAFT_RETRIES}) ===")
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
            logger.info("Schedule valido.")
            break
        else:
            logger.warning(f"{len(violations)} violazioni rilevate.")
            for v in violations[:5]:
                logger.warning(f"  • {v}")
            violation_feedback = violations
            schedule = None

    if schedule is None:
        logger.error(f"Impossibile generare uno schedule valido dopo {MAX_DRAFT_RETRIES} tentativi.")
        sys.exit(1)

    evaluate_fairness(schedule)
    logger.info(
        f"Soddisfazione: min={schedule.min_satisfaction():.3f}, "
        f"worker penalizzato: {schedule.least_satisfied_worker()}"
    )

    # ── Stage 4: Refinement ────────────────────────────────────────────────────
    logger.info("=== Stage 4: Refinement Maximin ===")
    final_schedule = refine(schedule, draft)

    # ── Output ─────────────────────────────────────────────────────────────────
    print_schedule(final_schedule)
    csv_path = f"output/schedule_use_case_{draft.use_case}.csv"
    export_to_csv(final_schedule, csv_path)
    logger.info(f"Schedule salvato in: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartScheduler")
    parser.add_argument("--use-case", choices=["A", "B"], default="A")
    parser.add_argument("--draft",   default=None)
    parser.add_argument("--workers", default=None)
    parser.add_argument(
        "--fallback", action="store_true",
        help="Usa preferenze neutre invece di LLaMA (baseline senza LLM)"
    )
    parser.add_argument(
        "--reextract", action="store_true",
        help="Forza ri-estrazione via LLM anche se preferences.json esiste"
    )
    args = parser.parse_args()

    uc           = args.use_case.upper()
    draft_file   = args.draft   or f"input/model_draft_use_case_{uc.lower()}.txt"
    workers_file = args.workers or f"input/workers_use_case_{uc.lower()}.json"

    run(
        draft_file=draft_file,
        workers_file=workers_file,
        force_fallback=args.fallback,
        force_reextract=args.reextract,
    )