"""
agents/drafting/llm_drafting.py

Stage 2 — Drafting Agent basato su LLM.

----------------------------------------------------
Il flusso è:

    LLM propone assegnazioni guidate dalle preferenze
        ↓
    Ogni proposta viene validata (hard-constraint-safe) e "bloccata"
        ↓
    (si ripete per un numero limitato di iterazioni, accumulando
     altre proposte, mostrando all'LLM cosa è già bloccato)
        ↓
    Le assegnazioni bloccate vengono passate come vincoli di
    uguaglianza (pin) al solver OR-Tools (`SolverDraftingAgent`)
        ↓
    Il solver — esatto, con backtracking — COMPLETA il resto
    GARANTENDO la fattibilità

L'LLM quindi non deve mai garantire fattibilità: propone solo scelte
plausibili in base alle preferenze dichiarate. Il "lavoro pesante" di
soddisfare tutti i vincoli hard è demandato interamente al solver, che
lo fa nativamente e in modo esatto.

Refinement e repair mirato (Stage 4 e ripetizioni di Stage 2)
----------------------------------------------------------------
Quando `least_satisfied_id` o `violation_feedback` sono impostati, il
"perimetro libero" su cui l'LLM propone si restringe (il worker target,
o i worker coinvolti nelle violazioni). Le uniche assegnazioni
"bloccate" (pin) sono le poche proposte dall'LLM per quei worker — il
solver ottimizza comunque l'INTERA schedule (tutti i worker), con un
boost sulle preferenze del worker target se impostato.

Nota di design importante: NON blocchiamo l'intera schedule degli
altri worker così com'era prima. Farlo lascerebbe un solo worker (o un
piccolo sottoinsieme) a dover soddisfare da solo tutti i vincoli hard
(riposo, ore settimanali, quota mensile) che prima erano risolti
congiuntamente da tutti — molto più fragile, e verificato
empiricamente portare a `INFEASIBLE` anche in casi semplici. Lasciare
il solver libero di rioptimizzare tutto insieme è esattamente il
percorso già validato con lo Scenario B (solo OR-Tools).
"""

from __future__ import annotations

from datetime import timedelta
import logging
from typing import List, Optional, Set

from agents.drafting.base import DraftingAgent
from agents.drafting.prompt_builder import (
    build_more_assignments_prompt,
    extract_involved_workers,
)
from agents.drafting.schedule_parser import ScheduleParser
from agents.drafting.solver_drafting import SolverDraftingAgent

# Riutilizziamo il backend già implementato nello Stage 1
from agents.preference_agent import _get_available_backend

from input.model_draft_parser import ModelDraft
from models.schedule import Schedule
from models.worker import Worker
from agents.verification_agent import VerificationReport

logger = logging.getLogger(__name__)


class LLMDraftingAgent(DraftingAgent):

    def __init__(
        self,
        max_llm_iterations: int = 5,
        llm_num_predict: int = 1536,
        llm_num_ctx: int = 4096,
    ):
        self.backend = _get_available_backend()
        self.max_llm_iterations = max_llm_iterations
        self.llm_num_predict = llm_num_predict
        self.llm_num_ctx = llm_num_ctx
        # Il completamento garantito è sempre demandato al solver esatto.
        self._solver_agent = SolverDraftingAgent()

    @property
    def name(self) -> str:
        return "LLM Drafting Agent"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def solve(
        self,
        workers: List[Worker],
        draft: ModelDraft,
        least_satisfied_id: Optional[str] = None,
        boost_factor: int = 3,
        violation_feedback: Optional[VerificationReport] = None,
        model_export_path: Optional[str] = None,
        current_schedule: Optional[Schedule] = None,
    ) -> Optional[Schedule]:
        """
        Produce una schedule GARANTITA fattibile (salvo che il modello
        con le assegnazioni bloccate risulti INFEASIBLE — evento raro,
        dato che ogni proposta è validata prima di essere bloccata).

        Parametri
        ---------
        least_satisfied_id:
            Se impostato, l'LLM propone SOLO per questo worker (con
            priorità alle sue preferenze). Il solver ottimizza comunque
            l'INTERA schedule (come nello Scenario B), applicando un
            boost sulle preferenze di questo worker — le uniche
            assegnazioni "bloccate" sono le poche proposte dall'LLM per
            lui, non l'intera schedule degli altri worker (bloccare
            tutti gli altri lascerebbe un solo worker a dover
            soddisfare da solo tutti i vincoli hard, molto più fragile
            e a rischio INFEASIBLE — verificato empiricamente).
        violation_feedback:
            Se impostato (un `VerificationReport`), l'LLM propone SOLO
            per i worker coinvolti nelle violazioni; stesso principio:
            il solver ottimizza comunque l'intero problema.
        current_schedule:
            Se fornita, viene mostrata all'LLM come contesto (per
            evitare proposte ovviamente ridondanti) ma NON viene mai
            usata per bloccare rigidamente le assegnazioni degli altri
            worker — solo il solver decide la schedule finale completa.
        """

        worker_map = {w.worker_id: w for w in workers}

        # ── Perimetro libero (su cui l'LLM propone) ────────────────
        focus_ids: Optional[Set[str]]

        if least_satisfied_id is not None:
            focus_ids = {least_satisfied_id}

        elif violation_feedback is not None:
            focus_ids = extract_involved_workers(workers, violation_feedback)

        else:
            focus_ids = None  # drafting da zero: tutti i worker sono liberi

        # ── Assegnazioni bloccate: SOLO le proposte dell'LLM per i
        # worker a fuoco. Non blocchiamo mai l'intera schedule degli
        # altri worker: il solver deve restare libero di riottimizzare
        # tutto insieme, altrimenti un singolo worker "a fuoco" può
        # trovarsi a dover soddisfare da solo vincoli (riposo, ore,
        # quota mensile) che prima erano risolti congiuntamente da
        # tutti i 13 — molto più fragile e a rischio INFEASIBLE.
        locked_schedule = Schedule(assignments=[], workers=workers, shifts=[])

        current_units = {w.worker_id: 0 for w in workers}

        # ── Iterazioni LLM: propone, si valida, si blocca ───────────
        if self.backend is None:

            logger.warning(
                "[%s] Nessun backend LLM disponibile: nessuna assegnazione "
                "guidata dalle preferenze verrà proposta. Il solver OR-Tools "
                "completerà l'intera schedule da solo.",
                self.name,
            )

        else:

            for iteration in range(1, self.max_llm_iterations + 1):

                logger.info(
                    "[%s] Iterazione LLM %d/%d",
                    self.name,
                    iteration,
                    self.max_llm_iterations,
                )

                prompt = build_more_assignments_prompt(
                    workers=workers,
                    draft=draft,
                    locked_assignments=locked_schedule.assignments,
                    focus_worker_ids=focus_ids,
                    fairness_target=least_satisfied_id,
                )

                try:
                    raw = self.backend.call(
                        prompt,
                        num_predict=self.llm_num_predict,
                        num_ctx=self.llm_num_ctx,
                    )
                    parsed = ScheduleParser.from_json(raw, workers, draft)

                except Exception as e:
                    logger.warning(
                        "[%s] Iterazione LLM %d fallita: %s",
                        self.name,
                        iteration,
                        e,
                    )
                    continue

                accepted = 0
                rejected = 0

                for a in parsed.assignments:

                    if focus_ids is not None and a.worker_id not in focus_ids:
                        continue

                    worker = worker_map.get(a.worker_id)
                    if worker is None:
                        continue

                    if self._already_assigned(locked_schedule, a.worker_id, a.day):
                        continue

                    if not self._assignment_is_safe(
                        worker, a.day, a.shift_type, locked_schedule, current_units, draft
                    ):
                        rejected += 1
                        continue

                    locked_schedule.assignments.append(a)
                    current_units[a.worker_id] += draft.shift_weights[a.shift_type]
                    accepted += 1

                logger.info(
                    "[%s] Iterazione LLM %d: %d proposte accettate, %d scartate "
                    "(non hard-safe o duplicate). Totale bloccate finora: %d",
                    self.name,
                    iteration,
                    accepted,
                    rejected,
                    len(locked_schedule.assignments),
                )

        logger.info(
            "[%s] %d assegnazioni guidate dalle preferenze bloccate "
            "(tutte proposte dall'LLM in questa chiamata). Il resto della "
            "schedule (tutti i worker, inclusi quelli non a fuoco) viene "
            "ottimizzato/completato liberamente dal solver OR-Tools, che "
            "garantisce la fattibilità.",
            self.name,
            len(locked_schedule.assignments),
        )

        # ── Completamento GARANTITO fattibile via solver esatto ─────
        # NB: NON passiamo pin per i worker fuori da `focus_ids`: il
        # solver deve restare libero di rioptimizzare l'intero problema
        # insieme (esattamente come nello Scenario B), altrimenti un
        # singolo worker "a fuoco" rischia di dover soddisfare da solo
        # vincoli che richiedono la cooperazione di tutti e 13.
        return self._solver_agent.solve(
            workers=workers,
            draft=draft,
            least_satisfied_id=least_satisfied_id,
            boost_factor=boost_factor,
            model_export_path=model_export_path,
            pinned_assignments=locked_schedule.assignments,
            timeout_seconds=20
        )

    ####################################################################
    # Hard-constraint guard (valida le proposte dell'LLM prima di
    # bloccarle nel modello — NON reimplementa i vincoli per riempire
    # lo schedule, solo per filtrare le proposte dell'LLM)
    ####################################################################

    @staticmethod
    def _count_shifts_on_day(
        schedule: Schedule,
        worker_id: str,
        day,
    ) -> int:
        return sum(
            1
            for a in schedule.assignments
            if a.worker_id == worker_id and a.day == day
        )

    @staticmethod
    def _week_index(day, draft: ModelDraft) -> int:
        return (day - draft.start_date).days // 7

    def _worker_week_hours(
        self,
        schedule: Schedule,
        worker_id: str,
        week_idx: int,
        draft: ModelDraft,
    ) -> float:
        total = 0.0
        for a in schedule.assignments:
            if a.worker_id != worker_id:
                continue
            if self._week_index(a.day, draft) != week_idx:
                continue
            total += draft.shift_durations[a.shift_type]
        return total

    @staticmethod
    def _has_night_within_rest_window(
        schedule: Schedule,
        worker_id: str,
        day,
        draft: ModelDraft,
    ) -> bool:
        k_max = draft.constraints.rest_days_after_night
        for k in range(1, k_max + 1):
            prev_day = day - timedelta(days=k)
            if any(
                a.worker_id == worker_id
                and a.day == prev_day
                and a.shift_type == "night"
                for a in schedule.assignments
            ):
                return True
        return False

    @staticmethod
    def _creates_forward_rest_conflict(
        schedule: Schedule,
        worker_id: str,
        day,
        shift: str,
        draft: ModelDraft,
    ) -> bool:
        """
        Se il turno da aggiungere è una notte, controlla IN AVANTI: uno
        qualunque dei prossimi `rest_days_after_night` giorni ha già
        un'assegnazione per questo worker? Se sì, aggiungere questa
        notte violerebbe RETROATTIVAMENTE il riposo-dopo-notte.
        """
        if shift != "night":
            return False
        k_max = draft.constraints.rest_days_after_night
        for k in range(1, k_max + 1):
            next_day = day + timedelta(days=k)
            if any(
                a.worker_id == worker_id and a.day == next_day
                for a in schedule.assignments
            ):
                return True
        return False

    @staticmethod
    def _creates_consecutive_conflict(
        schedule: Schedule,
        worker_id: str,
        day,
        shift: str,
    ) -> bool:
        """
        Replica del vincolo C2 del solver OR-Tools
        (`solver/constraints.add_no_consecutive_shifts`):
        pomeriggio il giorno `d` + mattina il giorno `d+1` è vietato per
        lo stesso lavoratore. Controlliamo entrambe le direzioni.
        """

        if shift == "morning":
            prev_day = day - timedelta(days=1)
            if any(
                a.worker_id == worker_id
                and a.day == prev_day
                and a.shift_type == "afternoon"
                for a in schedule.assignments
            ):
                return True

        if shift == "afternoon":
            next_day = day + timedelta(days=1)
            if any(
                a.worker_id == worker_id
                and a.day == next_day
                and a.shift_type == "morning"
                for a in schedule.assignments
            ):
                return True

        return False

    def _assignment_is_safe(
        self,
        worker: Worker,
        day,
        shift: str,
        schedule: Schedule,
        current_units: dict,
        draft: ModelDraft,
    ) -> bool:
        """
        Ritorna True solo se aggiungere `(worker, day, shift)` allo
        schedule NON introduce alcuna violazione hard nota. Usato SOLO
        per decidere se una proposta dell'LLM è sicura da bloccare —
        tutti i vincoli hard sul resto della schedule sono comunque
        applicati (e garantiti) dal solver OR-Tools a valle.
        """

        # 1. un turno al giorno
        if self._count_shifts_on_day(
            schedule, worker.worker_id, day
        ) >= draft.constraints.max_shifts_per_day:
            return False

        # 2. riposo dopo la notte (controllo all'indietro)
        if self._has_night_within_rest_window(
            schedule, worker.worker_id, day, draft
        ):
            return False

        # 2b. riposo dopo la notte (controllo in avanti)
        if self._creates_forward_rest_conflict(
            schedule, worker.worker_id, day, shift, draft
        ):
            return False

        # 3. turni consecutivi
        if self._creates_consecutive_conflict(
            schedule, worker.worker_id, day, shift
        ):
            return False

        # 4. ore settimanali
        week_idx = self._week_index(day, draft)
        projected_hours = (
            self._worker_week_hours(schedule, worker.worker_id, week_idx, draft)
            + draft.shift_durations[shift]
        )
        if projected_hours > draft.constraints.max_hours_per_week:
            return False

        # 5. carico mensile (unità-turno)
        projected_units = current_units[worker.worker_id] + draft.shift_weights[shift]
        if projected_units > draft.constraints.shifts_per_month:
            return False

        return True

    @staticmethod
    def _already_assigned(
        schedule: Schedule,
        worker_id: str,
        day,
    ) -> bool:

        return any(
            a.worker_id == worker_id
            and a.day == day
            for a in schedule.assignments
        )