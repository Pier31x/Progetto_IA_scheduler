"""
models/schedule.py

Rappresenta lo schedule completo prodotto dal solver.

Scelta progettuale: Schedule è il "risultato" del sistema — l'oggetto
che attraversa tutti gli stadi dopo la risoluzione. Contiene sia le
assegnazioni che i metadati di fairness, in modo che il Verification
Agent e il Refinement Agent possano operare su di esso senza dover
re-interrogare il solver.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Tuple

from models.worker import Worker
from models.shift import Shift


# Tipo alias per chiarezza: (worker_id, day, shift_type) -> assegnato (bool)
Assignment = Dict[Tuple[str, date, str], bool]


@dataclass
class Schedule:
    """
    Schedule mensile completo.

    Attributi:
        assignments: dizionario sparso delle assegnazioni.
            Chiave: (worker_id, day, shift_type)
            Valore: True se il lavoratore è assegnato a quel turno
        workers: lista dei lavoratori coinvolti
        shifts: lista di tutti i turni del periodo
        satisfaction_scores: punteggi di soddisfazione normalizzati
            per ogni lavoratore, calcolati dal Verification Agent
    """

    assignments: Assignment = field(default_factory=dict)
    workers: List[Worker] = field(default_factory=list)
    shifts: List[Shift] = field(default_factory=list)
    satisfaction_scores: Dict[str, float] = field(default_factory=dict)

    def is_assigned(self, worker_id: str, day: date, shift_type: str) -> bool:
        """Restituisce True se il lavoratore è assegnato al turno dato."""
        return self.assignments.get((worker_id, day, shift_type), False)

    def get_worker_shifts(self, worker_id: str) -> List[Shift]:
        """Restituisce tutti i turni assegnati a un lavoratore."""
        return [
            shift for shift in self.shifts
            if self.is_assigned(worker_id, shift.day, shift.shift_type)
        ]

    def get_shift_workers(self, day: date, shift_type: str) -> List[str]:
        """Restituisce gli ID dei lavoratori assegnati a un turno specifico."""
        return [
            w.worker_id for w in self.workers
            if self.is_assigned(w.worker_id, day, shift_type)
        ]

    def least_satisfied_worker(self) -> str:
        """
        Restituisce l'ID del lavoratore con il punteggio di soddisfazione
        più basso. Usato dal Refinement Agent per identificare il target
        del miglioramento (criterio Maximin).
        """
        if not self.satisfaction_scores:
            raise ValueError("I punteggi di soddisfazione non sono ancora stati calcolati.")
        return min(self.satisfaction_scores, key=self.satisfaction_scores.get)

    def min_satisfaction(self) -> float:
        """Soddisfazione minima attuale (usata come baseline nel refinement)."""
        if not self.satisfaction_scores:
            raise ValueError("I punteggi di soddisfazione non sono ancora stati calcolati.")
        return min(self.satisfaction_scores.values())

    def summary(self) -> str:
        """Rappresentazione testuale sintetica dello schedule per il logging."""
        lines = ["=== Schedule Summary ==="]
        for worker in self.workers:
            assigned = self.get_worker_shifts(worker.worker_id)
            score = self.satisfaction_scores.get(worker.worker_id, "N/A")
            lines.append(
                f"  {worker.worker_id} ({worker.role}): "
                f"{len(assigned)} turni | soddisfazione: {score:.2f}"
                if isinstance(score, float) else
                f"  {worker.worker_id} ({worker.role}): {len(assigned)} turni"
            )
        return "\n".join(lines)
