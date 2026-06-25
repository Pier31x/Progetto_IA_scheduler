"""
app.py

Interfaccia web Streamlit per SmartScheduler.

Scelta progettuale: Streamlit è preferibile a tkinter perché:
1. Interfaccia nel browser — più presentabile
2. Componenti (tabelle, grafici, HTML) già integrati
3. Nessuna gestione manuale dei thread

Avvio:
    streamlit run app.py
"""

import logging
import os
import sys
from datetime import date, timedelta

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

st.set_page_config(page_title="SmartScheduler", page_icon="🏥", layout="wide")

SHIFT_COLORS = {
    "morning":   {"bg": "#FFF3CD", "border": "#F0A500", "label": "🌅 MAT"},
    "afternoon": {"bg": "#D1ECF1", "border": "#17A2B8", "label": "🌤 POM"},
    "night":     {"bg": "#E2D9F3", "border": "#6F42C1", "label": "🌙 NOT"},
}

# ── Log handler ───────────────────────────────────────────────────────────────

class StreamlitLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append(record)
    def get_log_text(self):
        return "\n".join(f"[{r.levelname}] {self.format(r)}" for r in self.records)

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_file(path):
    try:
        return open(path, encoding="utf-8").read()
    except FileNotFoundError:
        return ""

def save_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def all_days(start, end):
    d, days = start, []
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    return days

# ── Calendario HTML ───────────────────────────────────────────────────────────

def render_calendar(schedule) -> str:
    """
    Genera un calendario mensile in HTML puro.

    Scelta progettuale: usiamo st.html() con HTML/CSS inline invece
    di una libreria esterna. Questo evita dipendenze aggiuntive e
    ci dà controllo completo sul layout.

    Il calendario mostra 7 colonne (Lun–Dom). Ogni cella contiene
    i turni del giorno con i lavoratori assegnati, colorati per tipo.
    """
    from config.scenario import SCHEDULE_START, SCHEDULE_END
    days = all_days(SCHEDULE_START, SCHEDULE_END)

    # Raggruppa assegnazioni per giorno
    day_data: dict[date, dict[str, list[str]]] = {}
    for (wid, day, shift), assigned in schedule.assignments.items():
        if assigned:
            day_data.setdefault(day, {}).setdefault(shift, []).append(wid)

    # Costruiamo settimane (righe del calendario)
    # Troviamo il lunedì della prima settimana
    first_monday = days[0] - timedelta(days=days[0].weekday())
    last_sunday  = days[-1] + timedelta(days=(6 - days[-1].weekday()))
    weeks = []
    cur = first_monday
    while cur <= last_sunday:
        weeks.append([cur + timedelta(days=i) for i in range(7)])
        cur += timedelta(weeks=1)

    day_names = ["Lun", "Mar", "Mer", "Gio", "Ven", "Sab", "Dom"]

    css = """
    <style>
    .cal-wrap { font-family: Arial, sans-serif; overflow-x: auto; }
    .cal-table { border-collapse: collapse; width: 100%; table-layout: fixed; }
    .cal-table th {
        background: #1A3A6B; color: white; text-align: center;
        padding: 8px 4px; font-size: 13px; border: 1px solid #ddd;
    }
    .cal-cell {
        vertical-align: top; border: 1px solid #ddd;
        padding: 4px; min-height: 90px; background: #fff;
    }
    .cal-cell.other-month { background: #f9f9f9; }
    .cal-day-num {
        font-size: 12px; font-weight: bold; color: #555;
        margin-bottom: 4px; text-align: right;
    }
    .cal-day-num.today { color: #1A3A6B; }
    .shift-block {
        border-radius: 4px; padding: 3px 5px;
        margin-bottom: 3px; font-size: 10px;
        border-left: 3px solid;
        line-height: 1.4;
    }
    .shift-label { font-weight: bold; margin-bottom: 2px; }
    .worker-list { color: #333; }
    </style>
    """

    rows_html = ""
    for week in weeks:
        rows_html += "<tr>"
        for day in week:
            in_range = SCHEDULE_START <= day <= SCHEDULE_END
            cell_class = "cal-cell" if in_range else "cal-cell other-month"
            rows_html += f'<td class="{cell_class}">'
            rows_html += f'<div class="cal-day-num">{day.day}</div>'

            if in_range and day in day_data:
                for shift_type in ["morning", "afternoon", "night"]:
                    workers_on = day_data[day].get(shift_type, [])
                    if not workers_on:
                        continue
                    c = SHIFT_COLORS[shift_type]
                    workers_str = ", ".join(workers_on)
                    rows_html += f"""
                    <div class="shift-block"
                         style="background:{c['bg']};border-color:{c['border']}">
                        <div class="shift-label">{c['label']}</div>
                        <div class="worker-list">{workers_str}</div>
                    </div>"""

            rows_html += "</td>"
        rows_html += "</tr>"

    header_html = "".join(f"<th>{d}</th>" for d in day_names)

    return f"""
    {css}
    <div class="cal-wrap">
      <table class="cal-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """

# ── Legenda ───────────────────────────────────────────────────────────────────

def render_legend() -> str:
    items = "".join(
        f"""<span style="
            display:inline-flex; align-items:center; gap:6px;
            background:{c['bg']}; border-left:4px solid {c['border']};
            border-radius:4px; padding:4px 10px; font-size:13px;
            font-family:Arial; margin-right:10px;">
            {c['label']}
        </span>"""
        for c in SHIFT_COLORS.values()
    )
    return f'<div style="margin-bottom:12px">{items}</div>'

# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    st.title("🏥 SmartScheduler")
    st.caption("Fair and Constraint-Aware Hospital Shift Scheduling")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configurazione")
        use_case = st.radio(
            "Use Case",
            ["A", "B"],
            format_func=lambda x: (
                "A — 13 lavoratori omogenei" if x == "A"
                else "B — 13 standard + 7 specializzati"
            ),
        )
        draft_path   = os.path.join(ROOT, "input", f"model_draft_use_case_{use_case.lower()}.txt")
        workers_path = os.path.join(ROOT, "input", f"workers_use_case_{use_case.lower()}.json")
        use_llm      = st.toggle("Usa LLaMA (richiede Ollama)", value=False)
        st.divider()
        st.caption("Output salvato in `output/` dopo l'esecuzione.")

    tab_input, tab_run, tab_results = st.tabs(["📄 Input", "▶️ Esecuzione", "📊 Risultati"])

    # ── TAB INPUT ─────────────────────────────────────────────────────────────
    with tab_input:
        st.subheader("Model Draft Istituzionale")
        st.caption("Turni, forza lavoro e vincoli legali. Salva prima di eseguire.")
        draft_content = st.text_area("model_draft.txt", value=load_file(draft_path), height=260, key="draft_ed")
        if st.button("💾 Salva model draft"):
            save_file(draft_path, draft_content)
            st.success("Salvato.")

        st.divider()
        st.subheader("Preferenze Lavoratori")
        st.caption(
            "Statement in linguaggio naturale. "
            "Con LLaMA puoi scrivere liberamente; "
            "col fallback usa keyword come *prefer*, *avoid*, *morning*, *saturday off*…"
        )
        workers_content = st.text_area("workers.json", value=load_file(workers_path), height=380, key="workers_ed")
        if st.button("💾 Salva preferenze"):
            save_file(workers_path, workers_content)
            st.success("Salvato.")

    # ── TAB ESECUZIONE ────────────────────────────────────────────────────────
    with tab_run:
        st.subheader("Avvia il sistema")
        mode = "LLaMA (Ollama)" if use_llm else "Rule-based fallback"
        st.info(
            f"**Use Case {use_case}** · Modalità: **{mode}**\n\n"
            f"- Draft: `{os.path.basename(draft_path)}`\n"
            f"- Workers: `{os.path.basename(workers_path)}`"
        )

        if st.button("▶ Avvia SmartScheduler", type="primary", use_container_width=True):
            handler = StreamlitLogHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            root_logger.setLevel(logging.INFO)
            os.chdir(ROOT)
            os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)

            prog = st.progress(0, text="Inizializzazione…")
            try:
                prog.progress(10, text="Stage 1 — Raccolta preferenze…")
                from agents.preference_agent import load_and_extract
                workers, _ = load_and_extract(workers_path, force_fallback=not use_llm)
                st.success(f"✅ Stage 1 — {len(workers)} lavoratori processati")

                prog.progress(30, text="Stage 2 — OR-Tools CP-SAT…")
                from input.model_draft_parser import parse_model_draft
                from agents.drafting_agent import solve
                draft = parse_model_draft(draft_path)
                model_out = os.path.join(ROOT, "output", "cp_model_partial.txt")
                schedule = solve(workers=workers, draft=draft, model_export_path=model_out)
                if schedule is None:
                    st.error("❌ Il solver non ha trovato soluzioni. Controlla i vincoli nel model draft.")
                    st.stop()
                st.success("✅ Stage 2 — Schedule generato")

                prog.progress(60, text="Stage 3 — Verifica vincoli…")
                from agents.verification_agent import verify, evaluate_fairness
                is_valid, violations = verify(schedule, draft)
                if not is_valid:
                    st.error(f"❌ {len(violations)} violazioni:")
                    for v in violations:
                        st.markdown(f"- {v}")
                    st.stop()
                evaluate_fairness(schedule)
                st.success(f"✅ Stage 3 — Valido · Soddisfazione minima: **{schedule.min_satisfaction():.3f}**")

                prog.progress(80, text="Stage 4 — Refinement Maximin…")
                from agents.refinement_agent import refine
                final = refine(schedule, draft)
                st.success(f"✅ Stage 4 — Soddisfazione minima finale: **{final.min_satisfaction():.3f}**")

                st.session_state["final_schedule"] = final
                st.session_state["use_case"] = use_case

                from output.schedule_output import export_to_csv
                export_to_csv(final, os.path.join(ROOT, "output", f"schedule_use_case_{use_case}.csv"))

                prog.progress(100, text="Completato.")
                st.balloons()
                st.info("📊 Vai al tab **Risultati** per visualizzare calendario e score.")

            except SystemExit:
                st.error("Il sistema ha terminato con errori. Vedi il log.")
            except Exception as e:
                st.error(f"Errore: {e}")
                raise
            finally:
                root_logger.removeHandler(handler)

            with st.expander("📋 Log completo"):
                st.code(handler.get_log_text(), language=None)

    # ── TAB RISULTATI ─────────────────────────────────────────────────────────
    with tab_results:
        if "final_schedule" not in st.session_state:
            st.info("Nessuno schedule disponibile. Vai al tab **Esecuzione** e avvia il sistema.")
            return

        import pandas as pd
        schedule  = st.session_state["final_schedule"]
        uc_label  = st.session_state.get("use_case", "A")
        scores    = schedule.satisfaction_scores

        # ── Metriche ──────────────────────────────────────────────────────────
        st.subheader("Punteggi di Soddisfazione")
        min_w = schedule.least_satisfied_worker()
        min_s = schedule.min_satisfaction()
        max_s = max(scores.values())
        avg_s = sum(scores.values()) / len(scores)

        c1, c2, c3 = st.columns(3)
        c1.metric("Minima", f"{min_s:.3f}", f"Worker: {min_w}")
        c2.metric("Massima", f"{max_s:.3f}")
        c3.metric("Media",   f"{avg_s:.3f}")

        score_df = pd.DataFrame(
            [{"Lavoratore": wid, "Soddisfazione": s}
             for wid, s in sorted(scores.items(), key=lambda x: x[1])]
        )
        st.bar_chart(score_df.set_index("Lavoratore"), color="#2E75B6")

        # ── Calendario ────────────────────────────────────────────────────────
        st.divider()
        st.subheader("📅 Calendario")
        st.html(render_legend())
        st.html(render_calendar(schedule))

        # ── Tabella pivot ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("Tabella Turni")
        from config.scenario import SCHEDULE_START, SCHEDULE_END

        rows = []
        for day in all_days(SCHEDULE_START, SCHEDULE_END):
            row = {"Giorno": day.strftime("%a %d/%m")}
            for s in ["morning", "afternoon", "night"]:
                assigned = schedule.get_shift_workers(day, s)
                row[s.capitalize()] = ", ".join(assigned) if assigned else "—"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Download ──────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Download")
        c1, c2 = st.columns(2)

        csv_path = os.path.join(ROOT, "output", f"schedule_use_case_{uc_label}.csv")
        if os.path.exists(csv_path):
            c1.download_button("⬇️ Schedule CSV", open(csv_path, "rb"),
                               file_name=os.path.basename(csv_path), mime="text/csv")

        model_path = os.path.join(ROOT, "output", "cp_model_partial.txt")
        if os.path.exists(model_path):
            c2.download_button("⬇️ CP-SAT Model", open(model_path, "rb"),
                               file_name="cp_model_partial.txt", mime="text/plain")


if __name__ == "__main__":
    main()