"""
input/model_draft_parser.py

Legge e parsa il file di input istituzionale (model_draft_use_case_*.txt)
che descrive turni, forza lavoro e vincoli hard.

Scelta progettuale: il file di input istituzionale è separato dalle
preferenze dei lavoratori per una ragione semantica precisa — i vincoli
istituzionali (legge, contratto collettivo) sono fissi e dettati
dall'ospedale, mentre le preferenze sono individuali e raccolte per
ogni ciclo di scheduling. Questa separazione rispecchia la realtà
operativa: il responsabile HR carica il model draft una volta, poi
raccoglie le preferenze dai lavoratori.

Il parser è puramente deterministico: nessun LLM coinvolto.
La struttura del file è semplice e formale — non c'è ambiguità
sintattica che giustifichi l'uso di un modello linguistico.

Formato atteso: file .txt con sezioni [SECTION_NAME] e righe
"chiave | valore | commento_opzionale" o "chiave = valore".
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Tuple


@dataclass
class ShiftDefinition:
    """Definizione di un tipo di turno letta dal model draft."""
    name: str           # "morning", "afternoon", "night"
    start_time: str     # "08:00"
    end_time: str       # "14:00"
    duration_hours: int
    weight: int         # 1 per MAT/POM, 2 per NOT


@dataclass
class WorkforceSpec:
    """Specifica della forza lavoro letta dal model draft."""
    total_workers: int
    roles: Dict[str, int]  # {"standard": 13} o {"standard": 13, "specialized": 7}


@dataclass
class CoverageRequirement:
    """Requisito di copertura minima per turno."""
    shift_type: str
    min_standard: int
    min_specialized: int  # 0 per Use Case A


@dataclass
class HardConstraints:
    """Vincoli hard letti dal model draft."""
    max_hours_per_week: int
    shifts_per_month: int
    rest_days_after_night: int
    max_shifts_per_day: int
    no_consecutive_shifts: bool


@dataclass
class ModelDraft:
    """
    Rappresentazione completa del model draft istituzionale.
    È il tipo restituito dal parser e consumato dal Drafting Agent.
    """
    start_date: date
    end_date: date
    shifts: List[ShiftDefinition] = field(default_factory=list)
    workforce: WorkforceSpec = None
    coverage: List[CoverageRequirement] = field(default_factory=list)
    constraints: HardConstraints = None

    @property
    def use_case(self) -> str:
        """Inferisce il use case dalla composizione della forza lavoro."""
        roles = self.workforce.roles if self.workforce else {}
        return "B" if "specialized" in roles and roles["specialized"] > 0 else "A"

    @property
    def shift_names(self) -> List[str]:
        return [s.name for s in self.shifts]

    @property
    def shift_weights(self) -> Dict[str, int]:
        return {s.name: s.weight for s in self.shifts}

    @property
    def shift_durations(self) -> Dict[str, int]:
        return {s.name: s.duration_hours for s in self.shifts}


def _parse_date(s: str) -> date:
    return date.fromisoformat(s.strip())


def _parse_key_value(line: str) -> Tuple[str, str]:
    """Parsa una riga 'chiave = valore' ignorando commenti inline."""
    line = line.split("#")[0].strip()
    key, _, value = line.partition("=")
    return key.strip(), value.strip()


def _parse_pipe_row(line: str) -> List[str]:
    """Parsa una riga 'campo | campo | ...' ignorando commenti."""
    line = line.split("#")[0].strip()
    return [part.strip() for part in line.split("|")]


def parse_model_draft(filepath: str) -> ModelDraft:
    """
    Legge e parsa il file di input istituzionale.

    Args:
        filepath: percorso al file .txt (es. input/model_draft_use_case_a.txt)

    Returns:
        Oggetto ModelDraft con tutte le specifiche istituzionali.

    Raises:
        ValueError: se il file manca di sezioni obbligatorie o ha dati malformati.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    # Raggruppa le righe per sezione
    sections: Dict[str, List[str]] = {}
    current_section = None

    for line in raw_lines:
        line = line.rstrip()
        # Linee vuote e commenti puri
        if not line or line.strip().startswith("#"):
            continue
        # Intestazione di sezione
        if line.strip().startswith("[") and line.strip().endswith("]"):
            current_section = line.strip()[1:-1]
            sections[current_section] = []
        elif current_section is not None:
            # Ignora commenti inline
            clean = line.split("#")[0].strip()
            if clean:
                sections[current_section].append(clean)

    # ── SCHEDULING_HORIZON ─────────────────────────────────────────────────────
    horizon = sections.get("SCHEDULING_HORIZON", [])
    start_date = end_date = None
    for line in horizon:
        key, value = _parse_key_value(line)
        if key == "start_date":
            start_date = _parse_date(value)
        elif key == "end_date":
            end_date = _parse_date(value)

    if not start_date or not end_date:
        raise ValueError("SCHEDULING_HORIZON: start_date e end_date obbligatori.")

    # ── SHIFTS ─────────────────────────────────────────────────────────────────
    shift_defs = []
    for line in sections.get("SHIFTS", []):
        parts = _parse_pipe_row(line)
        if len(parts) < 5:
            continue
        shift_defs.append(ShiftDefinition(
            name=parts[0],
            start_time=parts[1],
            end_time=parts[2],
            duration_hours=int(parts[3]),
            weight=int(parts[4]),
        ))

    # ── WORKFORCE ──────────────────────────────────────────────────────────────
    workforce = None
    for line in sections.get("WORKFORCE", []):
        key, value = _parse_key_value(line)
        if key == "total_workers":
            total = int(value)
        elif key == "roles":
            roles = {}
            for part in value.split(","):
                role, _, count = part.strip().partition(":")
                roles[role.strip()] = int(count.strip())
    workforce = WorkforceSpec(total_workers=total, roles=roles)

    # ── COVERAGE_REQUIREMENTS ──────────────────────────────────────────────────
    coverage = []
    for line in sections.get("COVERAGE_REQUIREMENTS", []):
        parts = _parse_pipe_row(line)
        if len(parts) == 2:
            # Use Case A: solo min_workers totale
            coverage.append(CoverageRequirement(
                shift_type=parts[0],
                min_standard=int(parts[1]),
                min_specialized=0,
            ))
        elif len(parts) == 3:
            # Use Case B: min_standard + min_specialized
            coverage.append(CoverageRequirement(
                shift_type=parts[0],
                min_standard=int(parts[1]),
                min_specialized=int(parts[2]),
            ))

    # ── HARD_CONSTRAINTS ───────────────────────────────────────────────────────
    hc_data = {}
    for line in sections.get("HARD_CONSTRAINTS", []):
        parts = _parse_pipe_row(line)
        if len(parts) >= 2:
            hc_data[parts[0]] = parts[1]

    constraints = HardConstraints(
        max_hours_per_week=int(hc_data.get("max_hours_per_week", 36)),
        shifts_per_month=int(hc_data.get("shifts_per_month", 25)),
        rest_days_after_night=int(hc_data.get("rest_days_after_night", 2)),
        max_shifts_per_day=int(hc_data.get("max_shifts_per_day", 1)),
        no_consecutive_shifts=hc_data.get("no_consecutive_shifts", "true").lower() == "true",
    )

    return ModelDraft(
        start_date=start_date,
        end_date=end_date,
        shifts=shift_defs,
        workforce=workforce,
        coverage=coverage,
        constraints=constraints,
    )
