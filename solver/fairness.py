"""
solver/fairness.py

Calcola i punteggi di soddisfazione per ogni lavoratore e le metriche
di fairness sullo schedule prodotto dal solver.

Scelta progettuale: la fairness è calcolata FUORI dal solver, in Python
puro. Questo per due ragioni:
1. Il solver OR-Tools ottimizza un obiettivo scalare; il criterio Maximin
   richiede un approccio iterativo più leggibile in Python.
2. Separare il calcolo della soddisfazione dal modello di ottimizzazione
   permette di cambiare la funzione di scoring senza riscrivere il modello.

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
from typing import Dict, List

from models.worker import Worker
from models.schedule import Schedule
from models.shift import SHIFT_TYPES

# Settimana di riferimento per il calcolo del range di normalizzazione.
# Deve contenere tutti e 7 i giorni della settimana; 2026-12-07 è lunedì.
_REF_MONDAY = date(2026, 12, 7)
_REF_WEEK: List[date] = [_REF_MONDAY + timedelta(days=i) for i in range(7)]


def pref_score(worker: Worker, day: date, shift_type: str) -> float:
    """
    Calcola il contributo di un singolo turno alla soddisfazione del lavoratore.

    Scala: [-1.0, +1.0]
        +1.0  → turno preferito su un giorno qualsiasi
         0.0  → turno neutro su un giorno qualsiasi
        -0.5  → turno neutro su un giorno di riposo preferito
        -1.0  → turno da evitare (o combinazione di penalità che satura il clamp)

    Contributi:
      +1.0  se shift_type ∈ preferred_shifts
      -1.0  se shift_type ∈ avoid_shifts
      -(1 - night_tolerance)  se shift_type == "night"
             (0.0 con tol=1.0, -1.0 con tol=0.0)
      -0.5  se day_of_week ∈ preferred_days_off
             (il lavoratore lavora in un giorno che avrebbe voluto libero)

    Tutti i contributi sono sommati e poi clamped in [-1, +1].

    Nota sulla versione originale:
        day_name veniva calcolato ma non usato, rendendo preferred_days_off
        ininfluente sul punteggio. Questo è il punto corretto dove applicarla.
    """
    score = 0.0
    day_name = day.strftime("%A").lower()  # es. "sunday"

    if shift_type in worker.preferred_shifts:
        score += 1.0

    if shift_type in worker.avoid_shifts:
        score -= 1.0

    if shift_type == "night":
        score -= (1.0 - worker.night_tolerance)

    if day_name in worker.preferred_days_off:
        score -= 0.5

    return max(-1.0, min(1.0, score))


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

    La normalizzazione usa il best/worst pref_score ottenibile su tutti
    i (giorno della settimana, tipo di turno) possibili, moltiplicato
    per il numero di turni assegnati.

    Aggiornamento rispetto alla versione originale:
        La versione precedente usava un giorno di riferimento fisso (lunedì),
        il che era equivalente a ignorare preferred_days_off nel calcolo
        del range — il worst case non includeva la penalità per lavorare
        in un giorno di riposo preferito.
        Ora si itera su tutti e 7 i giorni di una settimana di riferimento
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

    all_scores = [
        pref_score(worker, d, s)
        for d in _REF_WEEK
        for s in SHIFT_TYPES
    ]

    best_score  = max(all_scores)
    worst_score = min(all_scores)

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
            normalized = 0.5
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
    Calcola le metriche aggregate e puntuali oggettive/soggettive sullo schedule.
    Soddisfa tutti i requisiti di log e i widget/grafici di app.py.
    """
    # 1. Recupero dei lavoratori se non passati direttamente
    if workers is None:
        if hasattr(schedule, 'workers') and schedule.workers:
            workers = schedule.workers
        elif hasattr(schedule, 'get_all_workers'):
            workers = schedule.get_all_workers()
        else:
            worker_ids = set(assignment.worker_id for assignment in schedule.assignments)
            workers = [Worker(worker_id=wid, role="standard") for wid in worker_ids]

    # 2. Calcolo metriche OGGETTIVE (Mappatura dei turni notturni per ID)
    night_counts_dict = {}
    for w in workers:
        shifts = schedule.get_worker_shifts(w.worker_id)
        nights = sum(1 for s in shifts if s.shift_type.lower() == "night")
        night_counts_dict[w.worker_id] = nights

    night_values = list(night_counts_dict.values())
    n_workers = len(night_values)

    if n_workers == 0:
        return {
            "std_nights": 0.0, "gini_nights": 0.0,
            "min_nights": 0.0, "max_nights": 0.0,
            "min_satisfaction": 0.0, "avg_satisfaction": 0.0,
            "night_counts": {}
        }

    min_nights = min(night_values)
    max_nights = max(night_values)

    # Deviazione Standard delle Notti
    avg_nights = sum(night_values) / n_workers
    var_nights = sum((x - avg_nights) ** 2 for x in night_values) / n_workers
    std_nights = var_nights ** 0.5

    # Indice di Gini delle Notti
    sorted_nights = sorted(night_values)
    if sum(sorted_nights) == 0:
        gini_nights = 0.0
    else:
        cumulative_sum = sum((i + 1) * x for i, x in enumerate(sorted_nights))
        gini_nights = (2 * cumulative_sum) / (n_workers * sum(sorted_nights)) - (n_workers + 1) / n_workers

    # 3. Calcolo metriche SOGGETTIVE (Soddisfazione normalizzata)
    scores = compute_satisfaction_scores(workers, schedule)
    values = list(scores.values())
    avg_sat = sum(values) / len(values) if values else 0.0
    min_sat = min(values) if values else 0.0

    violations_data = compute_preference_violations(workers, schedule)

    return {
        "std_nights": round(std_nights, 4),
        "gini_nights": round(gini_nights, 4),
        "min_nights": int(min_nights),
        "max_nights": int(max_nights),
        "min_satisfaction": round(min_sat, 4),
        "avg_satisfaction": round(avg_sat, 4),
        "night_counts": night_counts_dict,
        "violations": violations_data
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
        total_total_day_off_violations = total_day_off_violations + day_off_violated  # Rinomina per chiarezza se serve, o usa l'accumulatore sotto:
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
