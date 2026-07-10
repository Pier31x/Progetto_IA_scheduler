from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Assignment:
    worker_id: str
    day: date
    shift_type: str