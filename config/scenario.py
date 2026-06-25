"""
config/scenario.py

Centralizza i parametri dei due use case, ora derivati direttamente
dal model draft istituzionale invece di essere hardcoded.

Scelta progettuale: la singola fonte di verità per i parametri di
scheduling è il file model_draft_*.txt. Questo modulo li carica e li
espone come costanti globali per il resto del sistema.
In questo modo modificare il numero di lavoratori o i vincoli
richiede solo aggiornare il file di testo — non il codice.
"""

from input.model_draft_parser import parse_model_draft, ModelDraft

# ── Caricamento model draft ────────────────────────────────────────────────────
# I due model draft vengono caricati all'avvio e usati come riferimento.
# Il main.py seleziona quale usare in base all'argomento --use-case.

MODEL_DRAFT_A: ModelDraft = parse_model_draft("input/model_draft_use_case_a.txt")
MODEL_DRAFT_B: ModelDraft = parse_model_draft("input/model_draft_use_case_b.txt")

# ── Costanti derivate (Use Case A) ─────────────────────────────────────────────
SCHEDULE_START = MODEL_DRAFT_A.start_date
SCHEDULE_END   = MODEL_DRAFT_A.end_date

# Queste vengono sostituite a runtime dalla funzione get_scenario_config()
SHIFT_TYPES          = MODEL_DRAFT_A.shift_names        # ["morning", "afternoon", "night"]
MAX_HOURS_PER_WEEK   = MODEL_DRAFT_A.constraints.max_hours_per_week
SHIFTS_PER_MONTH     = MODEL_DRAFT_A.constraints.shifts_per_month
REST_DAYS_AFTER_NIGHT = MODEL_DRAFT_A.constraints.rest_days_after_night
MAX_SHIFTS_PER_DAY   = MODEL_DRAFT_A.constraints.max_shifts_per_day


def get_model_draft(use_case: str) -> ModelDraft:
    """Restituisce il ModelDraft corrispondente al use case selezionato."""
    if use_case.upper() == "A":
        return MODEL_DRAFT_A
    elif use_case.upper() == "B":
        return MODEL_DRAFT_B
    else:
        raise ValueError(f"Use case '{use_case}' non riconosciuto. Usare 'A' o 'B'.")
