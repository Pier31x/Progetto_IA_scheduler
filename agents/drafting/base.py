"""
agents/drafting/base.py

Interfaccia comune per tutti i Drafting Agent.

Scelta progettuale
------------------
Lo SmartScheduler supporta più strategie di generazione del piano:

- SolverDraftingAgent
    usa OR-Tools CP-SAT per costruire e risolvere il modello.

- LLMDraftingAgent
    delega la costruzione della schedule ad un Large Language Model.

Entrambi espongono la stessa API, permettendo al resto della pipeline
(main.py, refinement_agent.py, verification_agent.py) di rimanere
completamente indipendente dalla strategia scelta.

Questo rende possibile confrontare sperimentalmente approcci simbolici
e LLM senza modificare gli altri componenti del sistema.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from input.model_draft_parser import ModelDraft
from models.schedule import Schedule
from models.worker import Worker


class DraftingAgent(ABC):
    """
    Interfaccia comune dei Drafting Agent.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome descrittivo dell'agente."""

    @abstractmethod
    def solve(
        self,
        workers: List[Worker],
        draft: ModelDraft,
        least_satisfied_id: Optional[str] = None,
        boost_factor: int = 3,
        violation_feedback: Optional[List[str]] = None,
        model_export_path: Optional[str] = None,
    ) -> Optional[Schedule]:
        """
        Genera una schedule.

        Parameters
        ----------
        workers
            Lavoratori con preferenze formalizzate.

        draft
            Model Draft istituzionale.

        least_satisfied_id
            Worker da privilegiare durante il refinement.

        boost_factor
            Peso usato nel refinement.

        violation_feedback
            Violazioni rilevate dal Verification Agent
            durante un eventuale ciclo precedente.

        model_export_path
            Percorso in cui esportare il modello parziale.

        Returns
        -------
        Schedule | None
        """