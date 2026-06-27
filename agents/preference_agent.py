"""
agents/preference_agent.py

Stage 1: raccoglie le preferenze dei lavoratori in linguaggio naturale
e le formalizza in oggetti Worker tramite un LLM.

Scelta progettuale — modalità operative:

    1. LLM mode (default): chiama LLaMA via Ollama.
       Il modello interpreta liberamente il testo del lavoratore e
       produce JSON strutturato. È la modalità principale e quella
       che soddisfa la specifica del progetto (comprensione del
       linguaggio naturale tramite IA).

    2. Neutral fallback (--fallback): se Ollama non è disponibile
       o si forza il fallback, il Worker riceve preferenze completamente
       neutre. OR-Tools ottimizza solo sui vincoli istituzionali,
       distribuendo i turni in modo equo senza bias sulle preferenze
       individuali.

Controllo startup:
    check_llm_availability() verifica che:
    1. Ollama sia in esecuzione (porta 11434)
    2. Il modello richiesto (llama3) sia già stato scaricato (pull)
    Deve essere chiamata all'avvio del sistema prima di processare
    qualsiasi worker.
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
    Verifica che Ollama sia in esecuzione e che il modello richiesto
    sia già stato scaricato localmente.

    Esegue due controlli distinti:
      1. Raggiungibilità del server Ollama (porta 11434 di default)
      2. Presenza del modello nella lista dei modelli locali

    Questo è più rigoroso di un semplice ping: un server Ollama può
    essere attivo ma non avere ancora il modello scaricato, nel qual
    caso la prima chiamata di inferenza fallirebbe con timeout.

    Returns:
        (disponibile: bool, messaggio: str)
        Il messaggio descrive la causa del fallimento se disponibile=False,
        o conferma il successo se disponibile=True.
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
    except requests.ConnectionError:
        return False, (
            f"Ollama non raggiungibile su {base_url}. "
            "Avviare il server con: ollama serve"
        )
    except requests.Timeout:
        return False, f"Timeout connessione a Ollama ({base_url})."
    except requests.RequestException as e:
        return False, f"Errore connessione Ollama: {e}"

    if response.status_code != 200:
        return False, f"Ollama risponde con HTTP {response.status_code}."

    try:
        available_models = [m["name"] for m in response.json().get("models", [])]
    except (ValueError, KeyError):
        return False, "Risposta di Ollama malformata (impossibile leggere la lista modelli)."

    model_found = any(m.startswith(model) for m in available_models)
    if not model_found:
        models_str = ", ".join(available_models) if available_models else "(nessuno)"
        return False, (
            f"Modello '{model}' non trovato in Ollama. "
            f"Modelli disponibili: {models_str}. "
            f"Scaricare il modello con: ollama pull {model}"
        )

    return True, f"Modello '{model}' disponibile su {base_url}."


# ── Interfaccia astratta LLM ──────────────────────────────────────────────────

class LLMBackend(ABC):
    """
    Interfaccia astratta per un backend LLM.

    Rispetta il principio Open/Closed: aperto all'estensione
    (aggiungere nuovi backend), chiuso alla modifica (il chiamante
    non cambia al variare del backend).
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Restituisce True se il backend è raggiungibile e il modello è presente."""
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

    Il metodo is_available() usa check_llm_availability() per verificare
    sia la raggiungibilità del server sia la presenza del modello,
    non solo un ping generico.
    """

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
            "options": {
                "temperature": 0.1,
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

BACKENDS: list[LLMBackend] = [
    OllamaBackend(model="llama3")
]


def _get_available_backend() -> Optional[LLMBackend]:
    """
    Scorre la lista dei backend e restituisce il primo disponibile.
    La verifica include sia il server sia il modello scaricato.
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

    Il prompt è scritto in italiano perché la dichiarazione del lavoratore
    è in italiano. Indicare esplicitamente la lingua riduce gli errori di
    interpretazione nei modelli medio-piccoli (es. confondere "mattino"
    con un nome proprio invece di "morning").

    I valori dei campi JSON restano in inglese perché sono identificatori
    usati dal codice (shift_names, role, preferred_days_off in formato
    inglese). La mappatura italiano→inglese è esplicitata nelle regole.

    La temperatura bassa (0.1) e l'istruzione finale "SOLO l'oggetto JSON"
    limitano l'aggiunta di testo non richiesto prima o dopo il JSON.
    """
    return f"""Sei un assistente di pianificazione turni per un ospedale.
Estrai le preferenze di turno dalla dichiarazione del lavoratore (scritta in italiano)
e restituisci un unico oggetto JSON valido che corrisponda esattamente a questo schema:

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

Regole per i campi:
- preferred_shifts / avoid_shifts: sottoinsiemi di ["morning", "afternoon", "night"]
  (mattino = morning, pomeriggio = afternoon, notte = night)
- preferred_days_off: sottoinsiemi di ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
  (lunedì = monday, martedì = tuesday, mercoledì = wednesday, giovedì = thursday,
   venerdì = friday, sabato = saturday, domenica = sunday)
- night_tolerance: da 0.0 (detesta i notturni) a 1.0 (indifferente o li preferisce)
- holiday_tolerance: da 0.0 (detesta i festivi) a 1.0 (indifferente ai festivi)
- emergency_availability: intero, numero massimo di turni straordinari al mese dichiarati dal lavoratore
- Restituisci SOLO l'oggetto JSON. Nessun markdown, nessuna spiegazione, nessun preambolo.

Dichiarazione del lavoratore: "{statement}"

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
    Tenta di estrarre le preferenze usando il backend LLM fornito.
    Se il backend è None o fallisce tutti i tentativi, restituisce
    un Worker con preferenze neutre.

    Scelta progettuale: questa funzione NON fa discovery — riceve
    il backend già risolto da extract_all_preferences. Questo garantisce
    che la chiamata di rete per verificare la disponibilità dell'LLM
    avvenga UNA SOLA VOLTA per batch, non una volta per lavoratore.
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
            logger.info(
                f"[{worker_id}] Preferenze estratte via {backend.name} "
                f"(tentativo {attempt}): prefer={worker.preferred_shifts}, "
                f"avoid={worker.avoid_shifts}, night_tol={worker.night_tolerance:.1f}"
            )
            return worker
        except Exception as e:
            logger.warning(
                f"[{worker_id}] {backend.name} tentativo {attempt}/{max_retries}: {e}"
            )

    logger.warning(
        f"[{worker_id}] LLM fallito dopo {max_retries} tentativi. Preferenze neutre."
    )
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
        force_fallback: se True, salta l'LLM e usa preferenze neutre
    """
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