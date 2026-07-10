"""
agents/drafting/schedule_parser.py

Converte output JSON del LLM in Schedule compatibile.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import List

from input.model_draft_parser import ModelDraft
from models.schedule import Schedule
from models.shift import Shift
from models.assignment import Assignment
from models.worker import Worker

logger = logging.getLogger(__name__)


class ScheduleParser:

    @staticmethod
    def _parse_llm_json(llm_output: str) -> dict:
        """
        Estrae il JSON dalla risposta del LLM.

        Gli LLM instruction-tuned non sempre rispettano "restituisci
        SOLO JSON": a volte anteponogono testo libero o avvolgono il
        JSON in un blocco markdown ```json ... ```. Qui proviamo prima
        un parsing diretto, poi un'estrazione tollerante del primo
        blocco {...} nel testo, prima di arrenderci.

        In caso di fallimento totale, l'errore include un estratto
        della risposta grezza: senza questo, un log mostrerebbe solo
        "Expecting value: line 1 column 1 (char 0)" per qualunque
        risposta non-JSON (vuota o con preambolo), rendendo impossibile
        distinguere i due casi.
        """
        # Caso 1: l'intera risposta è già JSON valido
        try:
            #print("DEBUGGGGGGGGGGGGGG", llm_output)
            return json.loads(llm_output)
        except json.JSONDecodeError:
            pass

        # Caso 2: JSON annidato in testo libero o markdown — cerchiamo
        # il blocco {...} più esterno (greedy: dalla prima { all'ultima }).
        match = re.search(r"\{.*\}", llm_output, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        snippet = llm_output.strip()[:3000]
        snippet_repr = repr(snippet) if snippet else "(risposta completamente vuota)"
        raise ValueError(f"Invalid JSON from LLM. Risposta ricevuta: {snippet_repr}")

    @staticmethod
    def from_json(
        llm_output: str,
        workers: List[Worker],
        draft: ModelDraft,
    ) -> Schedule:

        data = ScheduleParser._parse_llm_json(llm_output)

        worker_map = {w.worker_id: w for w in workers}

        valid_days = set()
        d = draft.start_date
        while d <= draft.end_date:
            valid_days.add(d)
            d += timedelta(days=1)

        assignments: List[Assignment] = []
        seen = set()

        for item in data.get("assignments", []):
            worker_id = item.get("worker")
            shift_type = item.get("shift")
            day_str = item.get("day")

            if not worker_id or not shift_type or not day_str:
                continue

            if worker_id not in worker_map:
                continue

            if shift_type not in draft.shift_names:
                continue

            try:
                day = datetime.strptime(day_str, "%Y-%m-%d").date()
            except Exception:
                continue

            if day not in valid_days:
                continue

            key = (worker_id, day, shift_type)

            if key in seen:
                continue

            seen.add(key)
            assignments.append(
                Assignment(worker_id=worker_id, day=day, shift_type=shift_type)
            )

        shifts = []
        d = draft.start_date

        while d <= draft.end_date:
            for s in draft.shift_names:
                shifts.append(Shift(day=d, shift_type=s))
            d += timedelta(days=1)

        return Schedule(
            assignments=assignments,
            workers=workers,
            shifts=shifts,
        )