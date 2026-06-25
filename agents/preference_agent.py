"""
agents/preference_agent.py

Stage 1: raccoglie le preferenze dei lavoratori in linguaggio naturale
e le formalizza in oggetti Worker tramite un LLM (LLaMA).

Scelta progettuale — modalità duale:
    Il Preference Agent supporta due modalità operative:

    1. LLM mode (produzione): chiama LLaMA via API Ollama.
       L'LLM interpreta il testo libero e produce JSON strutturato.

    2. Rule-based fallback (sviluppo/test): un parser deterministico
       basato su keyword estrae le preferenze senza bisogno di un LLM.
       È meno espressivo ma garantisce sviluppo e test offline.

    La scelta della modalità è automatica: se l'API di Ollama non è
    raggiungibile, il sistema degrada gracefully alla modalità fallback
    invece di bloccarsi. Questo approccio segue il principio di
    "graceful degradation": il sistema rimane funzionale anche in
    ambienti senza GPU o connessione al modello.

Flusso in LLM mode:
    testo libero → prompt con schema JSON → LLaMA → JSON → Worker

Flusso in fallback mode:
    testo libero → keyword matching → Worker (con score euristici)
"""

import json
import re
import logging
from typing import Optional

import requests

from models.worker import Worker

logger = logging.getLogger(__name__)

# ── Configurazione LLM ─────────────────────────────────────────────────────────
LLAMA_API_URL = "http://localhost:11434/api/generate"
LLAMA_MODEL = "llama3"
LLM_TIMEOUT_SECONDS = 60


# ── Prompt engineering ─────────────────────────────────────────────────────────

def _build_prompt(worker_id: str, role: str, statement: str) -> str:
    """
    Costruisce il prompt per l'LLM con schema JSON esplicito.

    Scelta progettuale: includere lo schema nel prompt (approccio
    "constrained generation") riduce gli errori di formato rispetto
    a un prompt generico. L'istruzione "Only output valid JSON" è
    necessaria perché i modelli medio-piccoli tendono ad aggiungere
    spiegazioni non richieste prima del JSON.
    """
    return f"""You are a scheduling assistant for a hospital.
Extract scheduling preferences from the worker's statement and return
a single valid JSON object matching this exact schema:

{{
  "worker_id": "{worker_id}",
  "role": "{role}",
  "preferred_shifts": [],
  "avoid_shifts": [],
  "preferred_days_off": [],
  "night_tolerance": 0.5,
  "holiday_tolerance": 0.5,
  "emergency_availability": 0
}}

Field rules:
- preferred_shifts / avoid_shifts: subsets of ["morning", "afternoon", "night"]
- preferred_days_off: subsets of ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
- night_tolerance: 0.0 (hates nights) to 1.0 (indifferent)
- holiday_tolerance: 0.0 (hates holidays) to 1.0 (indifferent)
- emergency_availability: integer, max extra shifts per month
- Output ONLY the JSON object. No markdown, no explanation.

Worker statement: "{statement}"

JSON:"""


# ── Chiamata LLM ───────────────────────────────────────────────────────────────

def _llm_available() -> bool:
    """Controlla se l'API di Ollama è raggiungibile."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _call_llama(prompt: str) -> str:
    """Chiama LLaMA via Ollama e restituisce il testo generato."""
    payload = {
        "model": LLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 300},
    }
    response = requests.post(LLAMA_API_URL, json=payload, timeout=LLM_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json().get("response", "")


def _extract_json_from_llm_output(raw: str) -> dict:
    """
    Estrae il primo oggetto JSON dal testo grezzo dell'LLM.

    Anche con istruzioni esplicite, i modelli a volte avvolgono il JSON
    in backtick markdown o aggiungono testo prima/dopo. La regex cattura
    il primo blocco {...} valido.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Nessun JSON trovato nell'output LLM: {raw[:200]}")
    return json.loads(match.group())


# ── Fallback rule-based ────────────────────────────────────────────────────────

def _rule_based_parse(worker_id: str, role: str, statement: str) -> dict:
    """
    Parser deterministico basato su keyword per estrarre preferenze.

    Scelta progettuale: il fallback non cerca di replicare la flessibilità
    dell'LLM — si limita a riconoscere pattern comuni e ad assegnare valori
    di default ragionevoli per il resto. È preferibile avere preferenze
    parzialmente estratte che bloccare il sistema.

    Keyword riconosciute:
        Turni: "morning", "afternoon", "night"
        Avoidance: "avoid", "dislike", "hate", "hard for me", "strongly prefer not"
        Tolleranza notte: modulata in base all'intensità del linguaggio
        Giorni: nomi dei giorni della settimana
    """
    text = statement.lower()

    preferred_shifts = []
    avoid_shifts = []
    preferred_days_off = []
    night_tolerance = 0.5
    holiday_tolerance = 0.5
    emergency_availability = 0

    # ── Turni preferiti ────────────────────────────────────────────────────────
    strong_prefer = bool(re.search(r"(prefer|like|love|enjoy).{0,30}morning", text))
    if strong_prefer:
        preferred_shifts.append("morning")

    if re.search(r"(prefer|like|love|enjoy).{0,30}afternoon", text):
        preferred_shifts.append("afternoon")

    if re.search(r"(prefer|like|love|enjoy).{0,30}night", text):
        preferred_shifts.append("night")
        night_tolerance = min(1.0, night_tolerance + 0.4)

    # ── Turni da evitare ───────────────────────────────────────────────────────
    avoid_pattern = r"(avoid|dislike|hate|hard for|strongly prefer not|barely tolerate)"

    if re.search(avoid_pattern + r".{0,30}night", text):
        avoid_shifts.append("night")
        # Tolleranza inversamente proporzionale all'intensità dell'avversione
        if re.search(r"(barely tolerate|strongly|really dislike|hate)", text):
            night_tolerance = 0.1
        else:
            night_tolerance = 0.3

    if re.search(avoid_pattern + r".{0,30}morning", text):
        avoid_shifts.append("morning")

    if re.search(avoid_pattern + r".{0,30}afternoon", text):
        avoid_shifts.append("afternoon")

    # ── Preferenza notti (esplicita positiva) ──────────────────────────────────
    if re.search(r"(prefer|fine with|ok with).{0,20}night", text):
        night_tolerance = 0.9
        if "night" not in preferred_shifts:
            preferred_shifts.append("night")

    # ── Giorni di riposo preferiti ─────────────────────────────────────────────
    days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for day in days_of_week:
        if re.search(rf"{day}.{{0,20}}(off|free|rest)", text) or \
           re.search(rf"(off|free|rest).{{0,20}}{day}", text):
            preferred_days_off.append(day)

    # Weekend generico
    if re.search(r"weekend.{0,20}(off|free|day)", text) or \
       re.search(r"(off|free).{0,20}weekend", text):
        for d in ["saturday", "sunday"]:
            if d not in preferred_days_off:
                preferred_days_off.append(d)

    # ── Disponibilità emergenze ────────────────────────────────────────────────
    emerg_match = re.search(r"(\d+).{0,20}(emergency|extra|coverage)", text)
    if emerg_match:
        emergency_availability = int(emerg_match.group(1))

    # ── Tolleranza festivi ─────────────────────────────────────────────────────
    if re.search(r"(prefer not|avoid|dislike).{0,30}(holiday|christmas|festiv)", text):
        holiday_tolerance = 0.2
    elif re.search(r"(fine|flexible|available).{0,30}(holiday|weekend|festiv)", text):
        holiday_tolerance = 0.8

    return {
        "worker_id": worker_id,
        "role": role,
        "preferred_shifts": preferred_shifts,
        "avoid_shifts": avoid_shifts,
        "preferred_days_off": preferred_days_off,
        "night_tolerance": night_tolerance,
        "holiday_tolerance": holiday_tolerance,
        "emergency_availability": emergency_availability,
    }


# ── Interfaccia pubblica ───────────────────────────────────────────────────────

def extract_preferences(
    worker_id: str,
    role: str,
    statement: str,
    max_retries: int = 2,
    force_fallback: bool = False,
) -> Worker:
    """
    Estrae le preferenze di un lavoratore dal suo statement.

    Tenta prima la modalità LLM; se non disponibile o se fallisce,
    usa il parser rule-based come fallback.

    Args:
        worker_id: identificatore del lavoratore
        role: "standard" o "specialized"
        statement: testo libero del lavoratore
        max_retries: tentativi in modalità LLM prima di passare al fallback
        force_fallback: se True, salta direttamente al parser rule-based
                        (utile in fase di sviluppo/test)

    Returns:
        Oggetto Worker con preferenze formalizzate.
    """
    # Modalità LLM
    if not force_fallback and _llm_available():
        prompt = _build_prompt(worker_id, role, statement)
        for attempt in range(1, max_retries + 1):
            try:
                raw = _call_llama(prompt)
                data = _extract_json_from_llm_output(raw)
                data["worker_id"] = worker_id  # override: non ci fidiamo dell'LLM per l'ID
                data["role"] = role
                worker = Worker.from_dict(data)
                logger.info(f"[{worker_id}] Preferenze estratte via LLM (tentativo {attempt}).")
                return worker
            except Exception as e:
                logger.warning(f"[{worker_id}] LLM tentativo {attempt}/{max_retries}: {e}")

        logger.warning(f"[{worker_id}] LLM fallito dopo {max_retries} tentativi. Uso fallback.")
    else:
        if not force_fallback:
            logger.info(f"[{worker_id}] LLaMA non disponibile. Uso parser rule-based.")

    # Modalità fallback
    data = _rule_based_parse(worker_id, role, statement)
    worker = Worker.from_dict(data)
    logger.info(f"[{worker_id}] Preferenze estratte via rule-based fallback.")
    return worker


def extract_all_preferences(
    worker_inputs: list[dict],
    force_fallback: bool = False,
) -> list[Worker]:
    """
    Processa tutti i lavoratori e restituisce la lista di Worker.

    Args:
        worker_inputs: lista di dict con chiavi
            {"worker_id": str, "role": str, "statement": str}
        force_fallback: se True, usa il parser rule-based per tutti

    Returns:
        Lista di oggetti Worker.
    """
    workers = []
    for entry in worker_inputs:
        worker = extract_preferences(
            worker_id=entry["worker_id"],
            role=entry["role"],
            statement=entry["statement"],
            force_fallback=force_fallback,
        )
        workers.append(worker)
    return workers


def load_and_extract(filepath: str, force_fallback: bool = False) -> tuple[list[Worker], str]:
    """
    Carica il file JSON di input e ne estrae i Worker.

    Args:
        filepath: percorso al file JSON (es. input/workers_use_case_a.json)
        force_fallback: passa al fallback rule-based

    Returns:
        (lista di Worker, use_case "A" o "B")
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    use_case = data.get("use_case", "A").upper()
    workers = extract_all_preferences(data["workers"], force_fallback=force_fallback)
    logger.info(f"Caricati {len(workers)} lavoratori da {filepath} (use case {use_case}).")
    return workers, use_case
