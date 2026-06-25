"""
models/shift.py

Rappresenta un singolo turno all'interno del calendario.

Scelta progettuale: il turno è modellato come entità immutabile con
attributi derivati (durata, peso). Questo evita di ridefinire la logica
di peso (notte = doppio) in più punti del codice — centralizzarla qui
garantisce coerenza tra il solver e le metriche di fairness.
"""

from dataclasses import dataclass
from datetime import date


# Lista canonica dei tipi di turno (usata come riferimento in tutto il progetto)
SHIFT_TYPES: list[str] = ["morning", "afternoon", "night"]

# Turni disponibili con le relative durate in ore
SHIFT_DURATIONS: dict[str, int] = {
    "morning": 6,    # 08:00 - 14:00
    "afternoon": 6,  # 14:00 - 20:00
    "night": 12,     # 20:00 - 08:00 (giorno successivo)
}

# La notte vale 2 "unità turno" ai fini del conteggio dei 25 turni mensili
SHIFT_WEIGHTS: dict[str, int] = {
    "morning": 1,
    "afternoon": 1,
    "night": 2,
}


@dataclass(frozen=True)
class Shift:
    """
    Rappresenta un turno in un giorno specifico.

    Attributi:
        day: data del turno
        shift_type: "morning", "afternoon" o "night"

    È frozen (immutabile) perché uno shift è un fatto del calendario,
    non qualcosa che cambia durante l'ottimizzazione.
    """

    day: date
    shift_type: str  # "morning" | "afternoon" | "night"

    def __post_init__(self):
        if self.shift_type not in SHIFT_DURATIONS:
            raise ValueError(f"Tipo turno '{self.shift_type}' non valido.")

    @property
    def duration_hours(self) -> int:
        """Durata del turno in ore."""
        return SHIFT_DURATIONS[self.shift_type]

    @property
    def weight(self) -> int:
        """
        Peso del turno ai fini del conteggio mensile (25 turni/mese).
        La notte pesa 2 perché dura il doppio e comporta vincoli aggiuntivi.
        """
        return SHIFT_WEIGHTS[self.shift_type]

    @property
    def is_night(self) -> bool:
        return self.shift_type == "night"

    @property
    def day_of_week(self) -> str:
        """Nome del giorno della settimana in inglese (per matching con preferenze)."""
        return self.day.strftime("%A").lower()  # es. "sunday"

    def __repr__(self) -> str:
        return f"Shift({self.day.isoformat()}, {self.shift_type})"
