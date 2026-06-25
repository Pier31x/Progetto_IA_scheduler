"""
output/schedule_output.py

Formatta e stampa lo schedule finale in modo leggibile.

Scelta progettuale: la logica di presentazione è separata dalla logica
di business. Questo modulo non sa nulla di OR-Tools o di come è stato
generato lo schedule — riceve un oggetto Schedule e lo visualizza.
Questo rende semplice aggiungere nuovi formati di output (es. CSV, HTML)
senza toccare gli agenti.
"""

from datetime import date, timedelta
from models.schedule import Schedule
from models.shift import SHIFT_TYPES
from config.scenario import SCHEDULE_START, SCHEDULE_END


def print_schedule(schedule: Schedule) -> None:
    """Stampa lo schedule in formato tabellare giorno × turno."""
    days = [
        SCHEDULE_START + timedelta(days=i)
        for i in range((SCHEDULE_END - SCHEDULE_START).days + 1)
    ]

    # Header
    print("\n" + "=" * 80)
    print("SMARTSCHEDULER — SCHEDULE FINALE")
    print("=" * 80)

    shift_labels = {"morning": "MAT", "afternoon": "POM", "night": "NOT"}

    for day in days:
        day_str = day.strftime("%a %d/%m")
        print(f"\n{day_str}")
        for shift_type in SHIFT_TYPES:
            assigned = schedule.get_shift_workers(day, shift_type)
            if assigned:
                workers_str = ", ".join(assigned)
                print(f"  {shift_labels[shift_type]}: {workers_str}")

    print("\n" + "=" * 80)
    print("PUNTEGGI DI SODDISFAZIONE")
    print("=" * 80)
    if schedule.satisfaction_scores:
        sorted_scores = sorted(
            schedule.satisfaction_scores.items(),
            key=lambda x: x[1]
        )
        for wid, score in sorted_scores:
            bar = "█" * int(score * 20)
            print(f"  {wid:>4}: {score:.3f} |{bar}")
    else:
        print("  (non calcolati)")

    print(f"\n  Min: {schedule.min_satisfaction():.3f} "
          f"— Lavoratore: {schedule.least_satisfied_worker()}")
    print("=" * 80 + "\n")


def export_to_csv(schedule: Schedule, filepath: str) -> None:
    """
    Esporta lo schedule in formato CSV.
    Colonne: worker_id, date, shift_type
    """
    import csv
    rows = []
    for (wid, day, shift), assigned in schedule.assignments.items():
        if assigned:
            rows.append({
                "worker_id": wid,
                "date": day.isoformat(),
                "shift_type": shift,
            })

    rows.sort(key=lambda r: (r["date"], r["shift_type"], r["worker_id"]))

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["worker_id", "date", "shift_type"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Schedule esportato in: {filepath}")
