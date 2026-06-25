"""
ortools/fairness.py

Calcola i punteggi di soddisfazione per ogni lavoratore e le metriche
di fairness sullo schedule prodotto dal solver.

Scelta progettuale: la fairness è calcolata FUORI dal solver, in Python
puro. Questo per due ragioni:
1. Il solver OR-Tools ottimizza un obiettivo scalare; il criterio Maximin
   richiede un approccio iterativo che è più leggibile e controllabile
   in Python.
2. Separare il calcolo della soddisfazione dal modello di ottimizzazione
   permette di cambiare la funzione di scoring senza riscrivere il modello.

Funzione di soddisfazione:
    satisfaction(w) = Σ_{d,s} x[w,d,s] · pref_score(w, d, s)

Normalizzata in [0, 1] per comparabilità tra lavoratori con preferenze
di intensità diversa.
"""

from datetime import date
from typing import Dict, List

from models.worker import Worker
from models.schedule import Schedule
from models.shift import SHIFT_TYPES


def pref_score(worker: Worker, day: date, shift_type: str) -> float:
    """
    Calcola il contributo di un singolo turno alla soddisfazione del lavoratore.

    Scala: [-1.0, +1.0]
        +1.0  → turno preferito
         0.0  → turno neutro
        -1.0  → turno da evitare (con modulazione per la tolleranza)

    Per i turni notturni e festivi, la penalità è modulata dalla tolleranza
    dichiarata: un lavoratore con night_tolerance=0.8 sarà meno penalizzato
    da una notte rispetto a uno con night_tolerance=0.2.
    """
    score = 0.0
    day_name = day.strftime("%A").lower()  # es. "sunday"

    # Bonus per turno preferito
    if shift_type in worker.preferred_shifts:
        score += 1.0

    # Penalità per turno da evitare
    if shift_type in worker.avoid_shifts:
        score -= 1.0

    # Penalità specifica per la notte, modulata dalla tolleranza
    if shift_type == "night":
        # (1 - tolerance) → più è bassa la tolleranza, più alta la penalità
        score -= (1.0 - worker.night_tolerance)

    # Bonus per giorno di riposo preferito rispettato
    # (non applicabile qui: questo è calcolato sui giorni liberi, non sui turni)

    return max(-1.0, min(1.0, score))  # clamp in [-1, 1]


def compute_raw_satisfaction(worker: Worker, schedule: Schedule) -> float:
    """
    Calcola la soddisfazione grezza (non normalizzata) di un lavoratore.

    È la somma pesata dei pref_score su tutti i turni assegnati.
    """
    total = 0.0
    for shift in schedule.get_worker_shifts(worker.worker_id):
        total += pref_score(worker, shift.day, shift.shift_type)
    return total


def compute_min_max_possible(worker: Worker, schedule: Schedule) -> tuple[float, float]:
    """
    Calcola i valori min e max REALI di soddisfazione per un lavoratore,
    basandosi sulle sue preferenze effettive — non su un caso peggiore/migliore
    assoluto uguale per tutti.

    Problema della versione precedente: usare min=-2n e max=+n per tutti
    schiacciava verso il basso i lavoratori con preferenze neutre, rendendoli
    apparentemente meno soddisfatti di lavoratori con preferenze forti che
    ricevevano esattamente gli stessi turni. Questo rendeva il confronto
    Maximin non significativo.

    Soluzione: il min e max sono calcolati come il pref_score peggiore e
    migliore che questo specifico lavoratore potrebbe ricevere su n turni,
    dati i suoi valori di preferred_shifts, avoid_shifts e night_tolerance.

    - max_possible: n turni tutti del tipo che massimizza pref_score per lui
    - min_possible: n turni tutti del tipo che minimizza pref_score per lui
    """
    assigned_shifts = schedule.get_worker_shifts(worker.worker_id)
    n = len(assigned_shifts)

    if n == 0:
        return 0.0, 0.0

    from datetime import date as date_type
    ref_day = date_type(2026, 12, 7)  # giorno di riferimento neutro (non festivo)

    # Calcola il pref_score per ogni tipo di turno per questo lavoratore
    scores_per_type = {s: pref_score(worker, ref_day, s) for s in SHIFT_TYPES}

    best_score  = max(scores_per_type.values())
    worst_score = min(scores_per_type.values())

    max_possible = n * best_score
    min_possible = n * worst_score

    # Caso degenere: lavoratore completamente indifferente (tutti gli score uguali)
    # → restituiamo un range artificiale per evitare divisione per zero
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
            # Caso degenere: nessuna preferenza espressa → score neutro
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
    2. Nessun altro lavoratore scende sotto il minimo precedente.

    Questa è l'implementazione diretta del criterio Rawlsiano descritto
    nel documento: "il miglioramento è accettato solo se non peggiora
    il minimo tra gli altri lavoratori".
    """
    old_min = min(old_scores.values())
    new_target_score = new_scores.get(target_worker_id, 0.0)
    old_target_score = old_scores.get(target_worker_id, 0.0)

    # Condizione 1: il target migliora
    if new_target_score <= old_target_score:
        return False

    # Condizione 2: nessun altro scende sotto il vecchio minimo
    for wid, new_score in new_scores.items():
        if wid == target_worker_id:
            continue
        if new_score < old_min:
            return False

    return True