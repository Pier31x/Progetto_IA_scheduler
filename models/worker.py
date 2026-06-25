"""
models/worker.py

Rappresenta un lavoratore con il suo ruolo e le sue preferenze.

Scelta progettuale: le preferenze sono mantenute separate dalla logica
di scheduling. Worker è un oggetto dati puro (no logica OR-Tools),
in modo da poter essere serializzato/deserializzato da JSON liberamente.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Worker:
    """
    Entità che rappresenta un lavoratore ospedaliero.

    Attributi:
        worker_id: identificatore univoco (es. "W01")
        role: "standard" o "specialized" (rilevante per Use Case B)
        preferred_shifts: turni preferiti tra "morning", "afternoon", "night"
        avoid_shifts: turni che il lavoratore vuole evitare
        preferred_days_off: giorni di riposo preferiti (es. ["sunday"])
        night_tolerance: valore in [0.0, 1.0] — quanto il lavoratore
            tollera i turni notturni (0 = li evita, 1 = indifferente)
        holiday_tolerance: analoga per i giorni festivi
        emergency_availability: numero massimo di disponibilità
            straordinarie al mese dichiarate dal lavoratore
    """

    worker_id: str
    role: str  # "standard" | "specialized"
    preferred_shifts: List[str] = field(default_factory=list)
    avoid_shifts: List[str] = field(default_factory=list)
    preferred_days_off: List[str] = field(default_factory=list)
    night_tolerance: float = 0.5
    holiday_tolerance: float = 0.5
    emergency_availability: int = 0

    def __post_init__(self):
        """Validazione dei campi al momento della creazione."""
        valid_roles = {"standard", "specialized"}
        if self.role not in valid_roles:
            raise ValueError(f"Ruolo '{self.role}' non valido. Attesi: {valid_roles}")

        valid_shifts = {"morning", "afternoon", "night"}
        for s in self.preferred_shifts + self.avoid_shifts:
            if s not in valid_shifts:
                raise ValueError(f"Turno '{s}' non valido. Attesi: {valid_shifts}")

        if not (0.0 <= self.night_tolerance <= 1.0):
            raise ValueError("night_tolerance deve essere in [0.0, 1.0]")

        if not (0.0 <= self.holiday_tolerance <= 1.0):
            raise ValueError("holiday_tolerance deve essere in [0.0, 1.0]")

    def to_dict(self) -> dict:
        """Serializzazione verso JSON (usata dal Preference Agent)."""
        return {
            "worker_id": self.worker_id,
            "role": self.role,
            "preferred_shifts": self.preferred_shifts,
            "avoid_shifts": self.avoid_shifts,
            "preferred_days_off": self.preferred_days_off,
            "night_tolerance": self.night_tolerance,
            "holiday_tolerance": self.holiday_tolerance,
            "emergency_availability": self.emergency_availability,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Worker":
        """Deserializzazione da JSON (usata dopo l'output del Preference Agent)."""
        return cls(
            worker_id=data["worker_id"],
            role=data["role"],
            preferred_shifts=data.get("preferred_shifts", []),
            avoid_shifts=data.get("avoid_shifts", []),
            preferred_days_off=data.get("preferred_days_off", []),
            night_tolerance=data.get("night_tolerance", 0.5),
            holiday_tolerance=data.get("holiday_tolerance", 0.5),
            emergency_availability=data.get("emergency_availability", 0),
        )
