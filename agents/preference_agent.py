"""
agents/preference_agent.py

Stage 1: raccoglie le preferenze dei lavoratori in linguaggio naturale
e le formalizza in oggetti Worker tramite un LLM.

Scelta progettuale — due modalità, non tre:

    1. LLM mode (default): chiama un modello linguistico via API.
       Il modello interpreta liberamente il testo del lavoratore e
       produce JSON strutturato. È l'unica modalità che soddisfa
       l'obiettivo del progetto (comprensione del linguaggio naturale).

    2. Hard-constraint fallback: se nessun LLM è disponibile, il
       sistema crea un Worker con preferenze completamente neutre.
       OR-Tools ottimizzerà solo sui vincoli istituzionali, ignorando
       le preferenze individuali — comportamento onesto e prevedibile.

    Motivazione: un parser a regex manuale è concettualmente scorretto
    per questo progetto. Scrivere regole per capire il linguaggio
    naturale è esattamente il problema che l'LLM risolve. Un fallback
    rule-based darebbe l'illusione di aver capito le preferenze quando
    in realtà le sta solo riconoscendo in forma rigida — peggio di non
    farne nulla, perché nasconde il limite invece di dichiararlo.

Modularità LLM:
    Il sistema supporta backend LLM multipli tramite la classe astratta
    LLMBackend. Aggiungere un nuovo modello (es. GPT, Mistral, Gemini)
    richiede solo implementare due metodi: is_available() e call().
    Il PreferenceAgent seleziona automaticamente il primo backend
    disponibile nella lista BACKENDS.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

import requests

from models.worker import Worker

logger = logging.getLogger(__name__)


# ── Interfaccia astratta LLM ──────────────────────────────────────────────────

class LLMBackend(ABC):
    """
    Interfaccia astratta per un backend LLM.

    Scelta progettuale: definire un'interfaccia comune permette di
    sostituire il modello senza toccare il resto del Preference Agent.
    Questo rispetta il principio Open/Closed: aperto all'estensione
    (aggiungere backend), chiuso alla modifica (il chiamante non cambia).
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Restituisce True se il backend è raggiungibile."""
        ...

    @abstractmethod
    def call(self, prompt: str) -> str:
        """Invia il prompt e restituisce il testo generato."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome leggibile del backend (per il logging)."""
        ...


# ── Backend: Ollama (LLaMA locale) ────────────────────────────────────────────

class OllamaBackend(LLMBackend):
    """
    Backend per modelli locali serviti via Ollama.
    Documentazione API: https://github.com/ollama/ollama/blob/main/docs/api.md
    """

    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.timeout = 60

    @property
    def name(self) -> str:
        return f"Ollama/{self.model}"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def call(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,   # bassa temperatura = output più deterministico
                "num_predict": 300,
            },
        }
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json().get("response", "")


# ── Lista dei backend disponibili (in ordine di priorità) ─────────────────────
# Per aggiungere un nuovo modello: implementa LLMBackend e aggiungilo qui.

BACKENDS: list[LLMBackend] = [
    OllamaBackend(model="llama3")
]


def _get_available_backend() -> Optional[LLMBackend]:
    """
    Scorre la lista dei backend e restituisce il primo disponibile.
    Restituisce None se nessuno è raggiungibile.
    """
    for backend in BACKENDS:
        if backend.is_available():
            logger.info(f"Backend LLM selezionato: {backend.name}")
            return backend
    return None


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(worker_id: str, role: str, statement: str) -> str:
    """
    Costruisce il prompt con schema JSON esplicito.

    Includere lo schema nel prompt (constrained generation) riduce
    significativamente gli errori di formato nei modelli medio-piccoli.
    La temperatura bassa (0.1) e l'istruzione "ONLY the JSON object"
    limitano l'aggiunta di testo non richiesto prima o dopo il JSON.
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
- Output ONLY the JSON object. No markdown, no explanation, no preamble.

Worker statement: "{statement}"

JSON:"""


def _extract_json(raw: str) -> dict:
    """
    Estrae il primo oggetto JSON valido dal testo grezzo dell'LLM.
    Gestisce il caso in cui il modello aggiunga testo o backtick markdown.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"Nessun JSON trovato nell'output LLM: {raw[:200]}")
    return json.loads(match.group())


# ── Fallback: preferenze neutre ───────────────────────────────────────────────

def _neutral_worker(worker_id: str, role: str) -> dict:
    """
    Restituisce un Worker con preferenze completamente neutre.

    Questo è il fallback corretto quando nessun LLM è disponibile:
    dichiarare esplicitamente che non si hanno informazioni sulle
    preferenze è più onesto che tentare di estrarle con regex,
    che darebbe l'illusione di aver capito il testo.

    Con preferenze neutre, OR-Tools ottimizza esclusivamente sui
    vincoli hard istituzionali, distribuendo i turni in modo equo
    senza alcun bias sulle preferenze individuali.
    """
    return {
        "worker_id": worker_id,
        "role": role,
        "preferred_shifts": [],
        "avoid_shifts": [],
        "preferred_days_off": [],
        "night_tolerance": 0.5,
        "holiday_tolerance": 0.5,
        "emergency_availability": 0,
    }


# ── Interfaccia pubblica ───────────────────────────────────────────────────────

def _apply_backend(
    worker_id: str,
    role: str,
    statement: str,
    backend: Optional[LLMBackend],
    max_retries: int = 2,
) -> Worker:
    """
    Tenta di estrarre le preferenze usando il backend fornito.

    Scelta progettuale: questa funzione NON fa discovery — riceve
    il backend già risolto da extract_all_preferences. Questo garantisce
    che la chiamata di rete per verificare la disponibilità dell'LLM
    avvenga UNA SOLA VOLTA per batch, non una volta per lavoratore.

    Args:
        backend: backend già verificato come disponibile, oppure None
                 se nessun LLM è raggiungibile.
    """
    if backend is None:
        logger.debug(f"[{worker_id}] Nessun LLM disponibile — preferenze neutre.")
        return Worker.from_dict(_neutral_worker(worker_id, role))

    prompt = _build_prompt(worker_id, role, statement)
    for attempt in range(1, max_retries + 1):
        try:
            raw = backend.call(prompt)
            data = _extract_json(raw)
            data["worker_id"] = worker_id
            data["role"] = role
            worker = Worker.from_dict(data)
            logger.info(f"[{worker_id}] Preferenze estratte via {backend.name}.")
            return worker
        except Exception as e:
            logger.warning(
                f"[{worker_id}] {backend.name} tentativo {attempt}/{max_retries}: {e}"
            )

    logger.warning(f"[{worker_id}] LLM fallito dopo {max_retries} tentativi. Preferenze neutre.")
    return Worker.from_dict(_neutral_worker(worker_id, role))


def extract_all_preferences(
    worker_inputs: list[dict],
    force_fallback: bool = False,
) -> list[Worker]:
    """
    Processa tutti i lavoratori estraendo le preferenze via LLM.

    La discovery del backend avviene UNA SOLA VOLTA qui, prima del loop.
    Tutti i lavoratori usano lo stesso backend già verificato.

    Args:
        worker_inputs: lista di dict con chiavi worker_id, role, statement
        force_fallback: se True, salta la discovery e usa preferenze neutre
    """
    # Discovery: una sola chiamata di rete per l'intero batch
    backend = None if force_fallback else _get_available_backend()

    if backend is None:
        msg = ("LLM non richiesto (force_fallback)" if force_fallback
               else "Nessun backend LLM disponibile")
        logger.warning(f"{msg} — tutti i Worker avranno preferenze neutre.")

    return [
        _apply_backend(
            worker_id=entry["worker_id"],
            role=entry["role"],
            statement=entry["statement"],
            backend=backend,
        )
        for entry in worker_inputs
    ]


def load_and_extract(
    filepath: str,
    force_fallback: bool = False,
) -> tuple[list[Worker], str]:
    """
    Carica il file JSON di input e ne estrae i Worker.

    Args:
        filepath: percorso al file JSON
        force_fallback: se True, preferenze neutre senza chiamare LLM

    Returns:
        (lista di Worker, use_case "A" o "B")
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    use_case = data.get("use_case", "A").upper()
    workers = extract_all_preferences(data["workers"], force_fallback=force_fallback)
    logger.info(f"Caricati {len(workers)} lavoratori da {filepath} (use case {use_case}).")
    return workers, use_case