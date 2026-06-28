"""
agents/preference_agent.py

Stage 1: raccoglie le preferenze dei lavoratori in linguaggio naturale
e le formalizza in oggetti Worker tramite un LLM.

Flusso:
    statement NL → LLM → JSON → Worker → preferences.json

Le preferenze estratte vengono salvate in output/preferences.json.
Questo file è il punto di disaccoppiamento centrale del sistema:
    - Tutti gli stadi successivi (drafting, refinement, confronto)
      leggono da preferences.json senza mai richiamare l'LLM
    - Il tab Confronto usa le stesse preferenze per i due run,
      garantendo che il confronto sia valido

Modalità operative:
    1. LLM mode: chiama LLaMA via Ollama, salva preferences.json
    2. Neutral mode (force_fallback=True): crea Workers con preferenze
       neutre senza chiamare l'LLM — usato come baseline nel confronto,
       non come sostituto dell'LLM

Modularità LLM:
    Aggiungere un nuovo backend (GPT, Mistral, ecc.) richiede solo
    implementare LLMBackend e aggiungerlo a BACKENDS.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

import requests

from models.worker import Worker

logger = logging.getLogger(__name__)


# ── Controllo disponibilità LLM ───────────────────────────────────────────────

def check_llm_availability(
    model: str = "llama3",
    base_url: str = "http://localhost:11434",
) -> tuple[bool, str]:
    """
    Verifica che Ollama sia in esecuzione e che il modello sia scaricato.
    Esegue due controlli: raggiungibilità del server + presenza del modello.
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
    except requests.ConnectionError:
        return False, f"Ollama non raggiungibile su {base_url}. Avviare con: ollama serve"
    except requests.Timeout:
        return False, f"Timeout connessione a Ollama ({base_url})."
    except requests.RequestException as e:
        return False, f"Errore connessione Ollama: {e}"

    if response.status_code != 200:
        return False, f"Ollama risponde con HTTP {response.status_code}."

    try:
        available_models = [m["name"] for m in response.json().get("models", [])]
    except (ValueError, KeyError):
        return False, "Risposta Ollama malformata."

    model_found = any(m.startswith(model) for m in available_models)
    if not model_found:
        models_str = ", ".join(available_models) if available_models else "(nessuno)"
        return False, (
            f"Modello '{model}' non trovato. "
            f"Disponibili: {models_str}. "
            f"Scaricarlo con: ollama pull {model}"
        )
    return True, f"Modello '{model}' disponibile su {base_url}."


# ── Interfaccia astratta LLM ──────────────────────────────────────────────────

class LLMBackend(ABC):
    """
    Interfaccia per backend LLM. Principio Open/Closed:
    aggiungere nuovi modelli non richiede modificare il chiamante.
    """
    @abstractmethod
    def is_available(self) -> bool: ...
    @abstractmethod
    def call(self, prompt: str) -> str: ...
    @property
    @abstractmethod
    def name(self) -> str: ...


class OllamaBackend(LLMBackend):
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.timeout = 60

    @property
    def name(self) -> str:
        return f"Ollama/{self.model}"

    def is_available(self) -> bool:
        available, msg = check_llm_availability(self.model, self.base_url)
        if not available:
            logger.debug(f"[{self.name}] Non disponibile: {msg}")
        return available

    def call(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512},
        }
        response = requests.post(
            f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json().get("response", "")


# Backend in ordine di priorità: il primo disponibile viene usato
BACKENDS: list[LLMBackend] = [
    OllamaBackend(model="llama3.2:1b"),  # veloce su CPU (~2-4s per worker)
    OllamaBackend(model="llama3.2"),
    OllamaBackend(model="llama3"),
    OllamaBackend(model="mistral"),
]


def _get_available_backend() -> Optional[LLMBackend]:
    """Restituisce il primo backend disponibile, None se nessuno lo è."""
    for backend in BACKENDS:
        if backend.is_available():
            logger.info(f"Backend LLM selezionato: {backend.name}")
            return backend
    return None


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(worker_id: str, role: str, statement: str) -> str:
    return f"""Sei un assistente di pianificazione turni per un ospedale.
Estrai le preferenze dalla dichiarazione del lavoratore (che può essere scritta in qualsiasi lingua) e restituisci
un unico oggetto JSON valido con questo schema:

{{
  "worker_id": "{worker_id}",
  "role": "{role}",
  "preferred_shifts": [],
  "avoid_shifts": [],
  "preferred_days_off": [],
  "night_tolerance": 0.5,
  "holiday_tolerance": 0.5,
  "emergency": 0
}}

Regole:
- preferred_shifts / avoid_shifts: sottoinsiemi di ["morning", "afternoon", "night"]
  (mattino=morning, pomeriggio=afternoon, notte=night)
- preferred_days_off: sottoinsiemi di ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
- night_tolerance: 0.0 (detesta i notturni) a 1.0 (indifferente o li preferisce)
- holiday_tolerance: 0.0 (detesta i festivi) a 1.0 (indifferente)
- emergency: intero 0-5, turni straordinari massimi al mese
- Restituisci SOLO l'oggetto JSON. Nessun markdown, nessuna spiegazione.

Dichiarazione: "{statement}"

JSON:"""


def _extract_json(raw: str) -> dict:
    """
    Estrae JSON dall'output LLM. Gestisce sia JSON completo che troncato
    (quando num_predict scatta a metà dell'ultimo campo).
    """
    # Caso 1: JSON completo
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Caso 2: JSON troncato — chiudi manualmente rimuovendo l'ultima riga incompleta
    start = raw.find("{")
    if start != -1:
        lines = raw[start:].split("\n")
        while lines and not lines[-1].strip().endswith(
            (",", "}", "]", "true", "false", "null") + tuple("0123456789")
        ):
            lines.pop()
        truncated = "\n".join(lines).rstrip().rstrip(",") + "\n}"
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Nessun JSON estraibile: {raw[:300]}")


# Mappature italiano → inglese per i valori nei campi JSON
_SHIFT_NORM = {
    "mattino": "morning", "mattina": "morning", "mat": "morning",
    "pomeriggio": "afternoon", "pom": "afternoon",
    "notte": "night", "not": "night", "notturno": "night",
}
_DAY_NORM = {
    "lunedì": "monday", "lunedi": "monday",
    "martedì": "tuesday", "martedi": "tuesday",
    "mercoledì": "wednesday", "mercoledi": "wednesday",
    "giovedì": "thursday", "giovedi": "thursday",
    "venerdì": "friday", "venerdi": "friday",
    "sabato": "saturday", "domenica": "sunday",
}


def _normalize(data: dict) -> dict:
    """
    Normalizza i valori prodotti dall'LLM:
    - converte nomi in italiano → inglese
    - rinomina 'emergency' → 'emergency_availability'
    """
    ns = lambda v: _SHIFT_NORM.get(v.lower(), v.lower())
    nd = lambda v: _DAY_NORM.get(v.lower(), v.lower())
    data["preferred_shifts"]   = [ns(s) for s in data.get("preferred_shifts", [])]
    data["avoid_shifts"]       = [ns(s) for s in data.get("avoid_shifts", [])]
    data["preferred_days_off"] = [nd(d) for d in data.get("preferred_days_off", [])]
    if "emergency" in data and "emergency_availability" not in data:
        data["emergency_availability"] = data.pop("emergency")
    return data


# ── Preferenze neutre ─────────────────────────────────────────────────────────

def make_neutral_worker(worker_id: str, role: str) -> dict:
    """
    Worker con preferenze completamente neutre.
    Usato come baseline nel confronto: stessi lavoratori, nessuna preferenza.
    Non è un fallback dell'LLM — è una scelta esplicita per il confronto.
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


# ── Serializzazione preferenze ────────────────────────────────────────────────

def save_preferences(workers: list[Worker], path: str) -> None:
    """
    Salva le preferenze estratte in preferences.json.
    Questo file è il punto di disaccoppiamento tra Stage 1 e gli stadi successivi.
    """
    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([w.to_dict() for w in workers], f, indent=2, ensure_ascii=False)
    logger.info(f"Preferenze salvate in: {path}")


def load_preferences(path: str) -> list[Worker]:
    """Carica preferenze da un file JSON precedentemente salvato."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    workers = [Worker.from_dict(w) for w in data]
    logger.info(f"Preferenze caricate da: {path} ({len(workers)} lavoratori)")
    return workers


# ── Estrazione ────────────────────────────────────────────────────────────────

def _apply_backend(
    worker_id: str,
    role: str,
    statement: str,
    backend: Optional[LLMBackend],
    max_retries: int = 2,
) -> Worker:
    """
    Estrae le preferenze di un lavoratore usando il backend LLM.
    Non fa discovery — riceve il backend già risolto da extract_all_preferences.
    Se il backend è None o fallisce, restituisce preferenze neutre.
    """
    if backend is None:
        return Worker.from_dict(make_neutral_worker(worker_id, role))

    prompt = _build_prompt(worker_id, role, statement)
    for attempt in range(1, max_retries + 1):
        try:
            raw = backend.call(prompt)
            data = _normalize(_extract_json(raw))
            data["worker_id"] = worker_id
            data["role"] = role
            worker = Worker.from_dict(data)
            logger.info(
                f"[{worker_id}] Estratto via {backend.name} "
                f"(tentativo {attempt}): prefer={worker.preferred_shifts}, "
                f"avoid={worker.avoid_shifts}, night_tol={worker.night_tolerance:.1f}"
            )
            return worker
        except Exception as e:
            logger.warning(f"[{worker_id}] {backend.name} tentativo {attempt}/{max_retries}: {e}")

    logger.warning(f"[{worker_id}] LLM fallito dopo {max_retries} tentativi. Preferenze neutre.")
    return Worker.from_dict(make_neutral_worker(worker_id, role))


def extract_all_preferences(
    worker_inputs: list[dict],
    force_fallback: bool = False,
) -> list[Worker]:
    """
    Estrae le preferenze per tutti i lavoratori.
    La discovery del backend avviene UNA SOLA VOLTA prima del loop.
    """
    backend = None if force_fallback else _get_available_backend()
    if backend is None:
        msg = "LLM non richiesto" if force_fallback else "Nessun backend LLM disponibile"
        logger.warning(f"{msg} — tutti i Worker avranno preferenze neutre.")

    return [
        _apply_backend(
            worker_id=e["worker_id"],
            role=e["role"],
            statement=e["statement"],
            backend=backend,
        )
        for e in worker_inputs
    ]


def extract_and_save(
    workers_file: str,
    preferences_path: str,
    force_fallback: bool = False,
) -> list[Worker]:
    """
    Punto di ingresso principale per Stage 1.
    Legge gli statement, estrae le preferenze via LLM e le salva in preferences.json.

    Args:
        workers_file:       percorso al JSON con gli statement
        preferences_path:   dove salvare preferences.json
        force_fallback:     se True, usa preferenze neutre (per il baseline del confronto)

    Returns:
        Lista di Worker con preferenze estratte
    """
    with open(workers_file, encoding="utf-8") as f:
        data = json.load(f)

    use_case = data.get("use_case", "A").upper()
    workers = extract_all_preferences(data["workers"], force_fallback=force_fallback)
    save_preferences(workers, preferences_path)
    logger.info(f"Stage 1 completato: {len(workers)} lavoratori (use case {use_case})")
    return workers