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
    model: str = "llama3.2",
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
    def call(
        self,
        prompt: str,
        num_predict: Optional[int] = None,
        num_ctx: Optional[int] = None,
    ) -> str: ...
    @property
    @abstractmethod
    def name(self) -> str: ...


class OllamaBackend(LLMBackend):
    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        default_num_predict: int = 512,
        default_num_ctx: Optional[int] = None,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = 120
        self.default_num_predict = default_num_predict
        self.default_num_ctx = default_num_ctx

    @property
    def name(self) -> str:
        return f"Ollama/{self.model}"

    def is_available(self) -> bool:
        available, msg = check_llm_availability(self.model, self.base_url)
        if not available:
            logger.debug(f"[{self.name}] Non disponibile: {msg}")
        return available

    def call(
        self,
        prompt: str,
        num_predict: Optional[int] = None,
        num_ctx: Optional[int] = None,
    ) -> str:
        """
        `num_predict`/`num_ctx` sono opzionali e sovrascrivono i default
        dell'istanza SOLO per questa chiamata. Questo permette allo
        Stage 1 (una risposta breve per lavoratore) di usare i default
        leggeri, mentre lo Stage 2 (drafting/repair, che deve generare
        molte più righe di JSON) può richiedere un budget di token
        molto più ampio senza appesantire tutte le altre chiamate.
        """
        effective_num_predict = num_predict if num_predict is not None else self.default_num_predict
        options = {"temperature": 0.1, "num_predict": effective_num_predict}

        effective_num_ctx = num_ctx if num_ctx is not None else self.default_num_ctx
        if effective_num_ctx is not None:
            options["num_ctx"] = effective_num_ctx

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        response = requests.post(
            f"{self.base_url}/api/generate", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        # Ollama può rispondere con HTTP 200 e comunque segnalare un
        # errore (modello in errore, memoria insufficiente, ecc.) nel
        # campo "error", oppure con "response" vuoto se il modello non
        # ha generato nulla entro num_predict o il prompt eccede il
        # contesto (num_ctx).
        # Solleviamo un errore esplicito con la diagnosi reale.
        if "error" in data:
            raise RuntimeError(
                f"Ollama ({self.name}) ha restituito un errore: {data['error']}"
            )

        text = data.get("response", "")

        if not text.strip():
            raise RuntimeError(
                f"Ollama ({self.name}) ha risposto senza contenuto "
                f"(done={data.get('done')}, done_reason={data.get('done_reason')}). "
                "Possibile causa: il prompt supera il contesto del modello "
                "(num_ctx) oppure il modello non ha generato nulla entro "
                "num_predict. Prova ad aumentare num_ctx/num_predict o a "
                "usare un modello più piccolo/veloce."
            )

        return text


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
    return f"""Sei un assistente ospedaliero. Estrai le preferenze dal testo e restituisci SOLO un oggetto JSON valido.

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

REGOLE ASSEGNAZIONE:
- preferred_shifts: sottoinsieme di ["morning", "afternoon", "night"] (mattino, pomeriggio, notte) che il lavoratore PREFERISCE o AMA.
- avoid_shifts: sottoinsieme di ["morning", "afternoon", "night"] che il lavoratore ODIA o VUOLE EVITARE (es. "no notte" -> ["night"]).
- preferred_days_off: sottoinsieme di ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]. Inserisci SOLO il giorno specifico richiesto come LIBERO (es. "lunedì libero" -> ["monday"], "sabato e domenica liberi" -> ["saturday", "sunday"]).
  ⚠️ ATTENZIONE: Se il lavoratore è flessibile, disponibile sempre o non specifica giorni di riposo, la lista DEVE ESSERE VUOTA: []. Non inventare o aggiungere giorni a caso.
- night_tolerance: 0.0 (rifiuta notti), 0.25 (preferisce evitare), 0.5 (neutrale/non menzionato), 0.75 (disponibile), 1.0 (preferisce notti).
- holiday_tolerance: 0.0 (detesta festivi/domeniche) a 1.0 (disponibile/li preferisce). Default 0.5 se non menzionato.
- emergency: intero 0-5, numero massimo di straordinari/emergenze al mese dichiarati. Default 0 se non specificato.

Restituisci SOLO il JSON pulito. No blocchi markdown (no ```json), no spiegazioni prima o dopo.

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
    Sanitizza l'output del JSON creando un dizionario NUOVO di zecca.
    Elimina i bug di sovrascrittura e leak di memoria tra i lavoratori.
    """
    # 1. Se il dato è corrotto o nullo, restituiamo un dizionario vuoto sicuro
    if not data or not isinstance(data, dict):
        return {}

    # 2. CREIAMO UN DIZIONARIO COMPLETAMENTE NUOVO (Isolamento dei puntatori)
    clean_profile = {
        "worker_id": str(data.get("worker_id", "")).strip(),
        "role": str(data.get("role", "")).strip(),
        "preferred_shifts": [],
        "avoid_shifts": [],
        "preferred_days_off": [],
        "night_tolerance": 0.5,
        "holiday_tolerance": 0.5,
        "emergency": 0
    }

    # 3. Estrazione e sanitizzazione dei Turni (con liste locali nuove)
    valid_shifts = ["morning", "afternoon", "night"]

    raw_pref = data.get("preferred_shifts", [])
    if isinstance(raw_pref, list):
        clean_profile["preferred_shifts"] = [str(s).lower().strip() for s in raw_pref if
                                             str(s).lower().strip() in valid_shifts]

    raw_avoid = data.get("avoid_shifts", [])
    if isinstance(raw_avoid, list):
        clean_profile["avoid_shifts"] = [str(s).lower().strip() for s in raw_avoid if
                                         str(s).lower().strip() in valid_shifts]

    # 4. Estrazione e sanitizzazione INDIPENDENTE dei Giorni Off
    valid_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    raw_days = data.get("preferred_days_off", [])

    if isinstance(raw_days, list):
        days_extracted = [str(d).lower().strip() for d in raw_days if str(d).lower().strip() in valid_days]
        # Rimuove i duplicati mantenendo l'ordine
        clean_profile["preferred_days_off"] = list(dict.fromkeys(days_extracted))

    # 5. Parsing numerico sicuro (evita che stringhe o errori blocchino il dato)
    try:
        clean_profile["night_tolerance"] = max(0.0, min(1.0, float(data.get("night_tolerance", 0.5))))
    except (ValueError, TypeError):
        clean_profile["night_tolerance"] = 0.5

    try:
        clean_profile["holiday_tolerance"] = max(0.0, min(1.0, float(data.get("holiday_tolerance", 0.5))))
    except (ValueError, TypeError):
        clean_profile["holiday_tolerance"] = 0.5

    try:
        # Controlliamo sia "emergency" che "emergency_availability"
        em_val = data.get("emergency", data.get("emergency_availability", 0))
        clean_profile["emergency"] = max(0, min(5, int(float(em_val))))
    except (ValueError, TypeError):
        clean_profile["emergency"] = 0

    # Restituiamo il profilo atomico e clonato
    return clean_profile


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