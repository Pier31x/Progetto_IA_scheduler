"""
solver/fairness.py

Calcola i punteggi di soddisfazione per ogni lavoratore e le metriche
di fairness sullo schedule prodotto dal solver.

Scelta progettuale: la fairness è calcolata FUORI dal solver, in Python
puro, per il REPORTING (Stage 3/4, UI). Questo per due ragioni:
1. È più leggibile e verificabile separare la metrica dal modello di
   ottimizzazione.
2. Permette di cambiare la funzione di scoring senza riscrivere il modello.

L'obiettivo di OTTIMIZZAZIONE vero e proprio (in `solver_drafting.py`)
usa invece `best_worst_pref_score` — la stessa funzione usata qui per
la normalizzazione — così il "minimo" che il solver massimizza
coincide ESATTAMENTE con il "minimo" che viene poi riportato da
`compute_satisfaction_scores`. Nessun disallineamento tra le due.

Funzione di soddisfazione:
    satisfaction(w) = Σ_{d,s} x[w,d,s] · pref_score(w, d, s)

Normalizzata in [0, 1] per comparabilità tra lavoratori con preferenze
di intensità diversa.

Correzione rispetto alla versione originale:
    pref_score() computava day_name ma non lo usava mai — preferred_days_off
    non veniva quindi considerato nel punteggio. Corretta l'omissione.
    compute_min_max_possible() usava un giorno di riferimento fisso (lunedì);
    aggiornato per iterare su tutti i 7 giorni della settimana, così che
    il best/worst case tengano conto dei giorni liberi preferiti.
"""

from datetime import date, timedelta
from typing import Dict, List, Tuple

from models.worker import Worker
from models.schedule import Schedule
from models.shift import SHIFT_TYPES

from datetime import date, timedelta
from typing import List, Tuple

# Settimana di riferimento per il calcolo del range di normalizzazione.
# Deve contenere tutti e 7 i giorni della settimana; 2026-12-07 è lunedì.
_REF_MONDAY = date(2026, 12, 7)
_REF_WEEK: List[date] = [_REF_MONDAY + timedelta(days=i) for i in range(7)]

# COSTANTE UNIVERSALE IN INGLESE PER EVITARE BUG DI LINGUA DEL PC
DAYS_EN = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
SHIFT_TYPES = ["morning", "afternoon", "night"]


def pref_score(worker: Worker, day: date, shift_type: str, is_overtime: bool = False) -> float:
    """
    Calcola il contributo normalizzato [-1.0, +1.0] di un turno alla soddisfazione del lavoratore.
    """
    score = 0.0

    # Usa l'indice numerico del giorno per pescare la stringa in inglese
    day_name = DAYS_EN[day.weekday()]

    # 1. Turni Preferiti / Da Evitare
    if shift_type in worker.preferred_shifts:
        score += 1.0
    if shift_type in worker.avoid_shifts:
        score -= 1.0

    # 2. Tolleranza Notturna
    if shift_type == "night":
        score -= (1.0 - worker.night_tolerance)

    # 3. Giorni di riposo desiderati
    if day_name in worker.preferred_days_off:
        score -= 0.5

    # 4. Tolleranza Festivi / Domeniche
    is_holiday = day_name == "sunday"
    if is_holiday:
        score -= (1.0 - worker.holiday_tolerance)

    # 5. Propensione agli straordinari (Emergency 0-5)
    if is_overtime:
        penalty_factor = 1.0 - (worker.emergency_availability / 5.0)
        score -= penalty_factor

    # Taglio rigoroso tra -1.0 e +1.0
    return max(-1.0, min(1.0, score))


def best_worst_pref_score(worker: Worker) -> Tuple[float, float]:
    """
    Range [best, worst] di `pref_score` garantito privo di bug linguistici.
    """
    all_scores = [
        pref_score(worker, d, s)
        for d in _REF_WEEK
        for s in SHIFT_TYPES
    ]
    return max(all_scores), min(all_scores)


def compute_raw_satisfaction(worker: Worker, schedule: Schedule) -> float:
    """
    Calcola la soddisfazione grezza (non normalizzata) di un lavoratore.
    È la somma dei pref_score su tutti i turni assegnati.
    """
    total = 0.0
    for shift in schedule.get_worker_shifts(worker.worker_id):
        total += pref_score(worker, shift.day, shift.shift_type)
    return total


def compute_min_max_possible(worker: Worker, schedule: Schedule) -> tuple[float, float]:
    """
    Calcola i valori min e max REALI di soddisfazione per un lavoratore,
    basandosi sulle sue preferenze effettive.

    La normalizzazione usa il best/worst pref_score ottenibile
    (`best_worst_pref_score`) moltiplicato per il numero di turni
    assegnati.

    Aggiornamento rispetto alla versione originale:
        Si itera su tutti e 7 i giorni di una settimana di riferimento
        per catturare sia il best case (turno preferito in giorno libero no)
        sia il worst case (turno evitato in giorno di riposo preferito).

    Lavoratore completamente indifferente (tutti gli score uguali):
        Restituisce un range artificiale per evitare divisione per zero
        nella normalizzazione.
    """
    assigned_shifts = schedule.get_worker_shifts(worker.worker_id)
    n = len(assigned_shifts)

    if n == 0:
        return 0.0, 0.0

    best_score, worst_score = best_worst_pref_score(worker)

    max_possible = n * best_score
    min_possible = n * worst_score

    if max_possible == min_possible:
        return min_possible - 1.0, max_possible + 1.0

    return min_possible, max_possible


def compute_satisfaction_scores(workers: List[Worker], schedule: Schedule) -> Dict[str, float]:
    """
    Calcola i punteggi di soddisfazione normalizzati in [0, 1] per tutti
    i lavoratori e li salva nello schedule.

    La normalizzazione è necessaria per il criterio Maximin: confrontare
    punteggi grezzi tra lavoratori con preferenze di intensità diversa
    sarebbe fuorviante.

    Returns:
        Dizionario {worker_id: score_normalizzato}
    """
    scores = {}
    for worker in workers:
        raw = compute_raw_satisfaction(worker, schedule)
        min_p, max_p = compute_min_max_possible(worker, schedule)

        if max_p == min_p:
            normalized = 1
        else:
            normalized = (raw - min_p) / (max_p - min_p)

        scores[worker.worker_id] = round(normalized, 4)

    schedule.satisfaction_scores = scores
    return scores


def is_fairness_improvement(
    old_scores: Dict[str, float],
    new_scores: Dict[str, float],
    target_worker_id: str,
) -> bool:
    """
    Valida se il nuovo schedule rappresenta un miglioramento Maximin
    rispetto al precedente.

    Condizioni per accettare il nuovo schedule:
    1. Il lavoratore target (least satisfied) migliora il suo score.
    2. Nessun altro lavoratore scende sotto il minimo globale precedente.

    Implementazione del criterio Rawlsiano: il miglioramento è accettato
    solo se non peggiora il minimo tra gli altri lavoratori.

    Nota: con l'obiettivo maximin ESATTO ora usato dal solver
    (`solver_drafting._add_maximin_fairness_objective`), una singola
    chiamata a `solve()` produce già il miglior minimo possibile — è
    normale e corretto che questa funzione, chiamata dal Refinement
    Agent su una schedule già maximin-ottima, non trovi ulteriori
    miglioramenti e il loop converga in una o zero iterazioni.
    """
    old_min = min(old_scores.values())
    new_target_score = new_scores.get(target_worker_id, 0.0)
    old_target_score = old_scores.get(target_worker_id, 0.0)

    if new_target_score <= old_target_score:
        return False

    for wid, new_score in new_scores.items():
        if wid == target_worker_id:
            continue
        if new_score < old_min:
            return False

    return True


def compute_equity_metrics(schedule: Schedule, workers: List[Worker] = None) -> Dict[str, any]:
    """
    Calcola le metriche essenziali sullo schedule per i widget di app.py.
    Traccia solo i valori minimi, massimi e le medie per un controllo immediato.
    """
    if workers is None:
        if hasattr(schedule, 'workers') and schedule.workers:
            workers = schedule.workers
        elif hasattr(schedule, 'get_all_workers'):
            workers = schedule.get_all_workers()
        else:
            worker_ids = set(assignment.worker_id for assignment in schedule.assignments)
            workers = [Worker(worker_id=wid, role="standard") for wid in worker_ids]

    n_workers = len(workers)
    if n_workers == 0:
        return {
            "min_nights": 0, "max_nights": 0, "discrepancy_nights": 0, "night_counts": {},
            "min_holidays": 0, "max_holidays": 0, "holiday_counts": {},
            "total_overtime_shifts": 0, "emergency_alignment": 1.0, "overtime_counts": {},
            "min_satisfaction": 0.0, "avg_satisfaction": 0.0
        }

    # 1. INIZIALIZZAZIONE SICURA: Mettiamo a 0 i contatori per TUTTI i lavoratori esistenti
    night_counts = {w.worker_id: 0 for w in workers}
    holiday_counts = {w.worker_id: 0 for w in workers}
    overtime_counts = {w.worker_id: 0 for w in workers}
    worker_map = {w.worker_id: w for w in workers}

    # 2. POPOLAMENTO DIRETTO DAL CALENDARIO (Leggiamo la lista globale delle assegnazioni)
    # Questo evita di usare get_worker_shifts() se non è affidabile.
    for ass in schedule.assignments:
        wid = ass.worker_id

        # Se l'assegnazione si riferisce a un lavoratore della nostra lista
        if wid in night_counts:
            # Conteggio Notti
            if ass.shift_type.lower() == "night":
                night_counts[wid] += 1

            # Conteggio Festivi (Domeniche) usando l'attributo .day dell'oggetto turno
            if ass.day.strftime("%A").lower() == "sunday":
                holiday_counts[wid] += 1

            # Conteggio Straordinari
            if getattr(ass, 'is_overtime', False):
                overtime_counts[wid] += 1

    # 3. ESTRAZIONE METRICHE RIGIDE
    min_nights, max_nights = min(night_counts.values()), max(night_counts.values())
    min_holidays, max_holidays = min(holiday_counts.values()), max(holiday_counts.values())
    total_overtime = sum(overtime_counts.values())

    # Calcolo allineamento straordinari (invariato)
    assigned_well = sum(count * worker_map[wid].emergency_availability for wid, count in overtime_counts.items())
    max_possible_weight = total_overtime * 5
    emergency_alignment = (assigned_well / max_possible_weight) if max_possible_weight > 0 else 1.0

    # Calcolo soddisfazione (invariato)
    scores = compute_satisfaction_scores(workers, schedule)
    min_sat = min(scores.values()) if scores else 0.0
    avg_sat = sum(scores.values()) / len(scores) if scores else 0.0

    # 4. RESTITUIAMO TUTTO, COMPRESI I DIZIONARI POPOLATI
    return {
        "min_nights": int(min_nights),
        "max_nights": int(max_nights),
        "discrepancy_nights": int(max_nights - min_nights),
        "night_counts": night_counts,  # <--- Ora conterrà i dati reali!

        "min_holidays": int(min_holidays),
        "max_holidays": int(max_holidays),
        "holiday_counts": holiday_counts,  # <--- Ora conterrà i dati reali!

        "total_overtime_shifts": int(total_overtime),
        "emergency_alignment": round(emergency_alignment, 2),
        "overtime_counts": overtime_counts,  # <--- Ora conterrà i dati reali!

        "min_satisfaction": round(min_sat, 4),
        "avg_satisfaction": round(avg_sat, 4)
    }


def compute_preference_violations(workers: list[Worker], schedule: Schedule) -> dict:
    """
    Conta quante volte il sistema assegna turni esplicitamente evitati
    o lavora nei giorni di riposo preferiti, sia per singolo lavoratore che globalmente.
    """
    violations_dict = {}
    total_avoid_violations = 0
    total_day_off_violations = 0

    for w in workers:
        shifts = schedule.get_worker_shifts(w.worker_id)

        # Conteggio violazioni sui turni da evitare
        avoid_assigned = sum(1 for s in shifts if s.shift_type in w.avoid_shifts)

        # Conteggio violazioni sui giorni liberi preferiti
        day_off_violated = sum(1 for s in shifts if s.day.strftime("%A").lower() in w.preferred_days_off)

        # Accumulo per i totali globali
        total_avoid_violations += avoid_assigned
        total_day_off_violations += day_off_violated

        # Mappatura del singolo lavoratore
        violations_dict[w.worker_id] = {
            "avoid_shifts_violations": avoid_assigned,
            "days_off_violations": day_off_violated,
            "total_violations": avoid_assigned + day_off_violated
        }

    # Struttura finale corretta attesa da app.py
    return {
        "per_worker": violations_dict,
        "global_avoid_violations": total_avoid_violations,
        "global_day_off_violations": total_day_off_violations,
        "global_total_violations": total_avoid_violations + total_day_off_violations
    }