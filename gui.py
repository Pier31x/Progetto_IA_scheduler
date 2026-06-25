"""
app.py

Interfaccia web Streamlit per SmartScheduler.

Scelta progettuale: Streamlit è preferibile a tkinter per questo
contesto perché:
1. L'interfaccia è nel browser — più presentabile e familiare
2. I componenti (tabelle, grafici, log) sono già integrati
3. Non richiede gestione manuale dei thread per aggiornare la UI
4. È una dipendenza standard nell'ecosistema data science / AI

L'app non reimplementa nessuna logica del sistema: chiama le stesse
funzioni di main.py, ma intercetta i risultati intermedi per
visualizzarli in modo strutturato invece di stamparli su terminale.

Avvio:
    streamlit run app.py
"""

import io
import logging
import os
import sys
from datetime import date, timedelta

import streamlit as st

# ── Setup path ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Configurazione pagina ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartScheduler",
    page_icon="🏥",
    layout="wide",
)


# ── Intercettore log per Streamlit ────────────────────────────────────────────

class StreamlitLogHandler(logging.Handler):
    """
    Handler che reindirizza i log di Python verso un buffer
    leggibile da Streamlit.

    Scelta progettuale: invece di modificare i logger nei moduli
    esistenti, aggiungiamo un handler a livello root. Questo cattura
    tutti i log del sistema (agenti, solver, ecc.) senza toccare
    il codice di produzione.
    """
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def get_log_text(self) -> str:
        lines = []
        for r in self.records:
            level = r.levelname
            msg = self.format(r)
            lines.append(f"[{level}] {msg}")
        return "\n".join(lines)

    def clear(self):
        self.records = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_file_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def save_file_text(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def get_all_days(start: date, end: date):
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def build_schedule_dataframe(schedule):
    """Converte lo Schedule in una struttura tabellare per Streamlit."""
    import pandas as pd
    rows = []
    for (wid, day, shift), assigned in schedule.assignments.items():
        if assigned:
            rows.append({
                "Data": day.strftime("%a %d/%m"),
                "Turno": shift.capitalize(),
                "Lavoratore": wid,
                "Ruolo": next(
                    (w.role for w in schedule.workers if w.worker_id == wid), "?"
                ),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values(["Data", "Turno", "Lavoratore"]).reset_index(drop=True)


def build_pivot(schedule):
    """Pivot: righe = giorni, colonne = turni, valori = lista lavoratori."""
    import pandas as pd
    from config.scenario import SCHEDULE_START, SCHEDULE_END
    days = get_all_days(SCHEDULE_START, SCHEDULE_END)
    shift_order = ["morning", "afternoon", "night"]
    rows = []
    for day in days:
        row = {"Giorno": day.strftime("%a %d/%m")}
        for s in shift_order:
            workers_on = schedule.get_shift_workers(day, s)
            row[s.capitalize()] = ", ".join(workers_on) if workers_on else "—"
        rows.append(row)
    return pd.DataFrame(rows)


# ── UI principale ─────────────────────────────────────────────────────────────

def main():
    st.title("🏥 SmartScheduler")
    st.caption("Fair and Constraint-Aware Hospital Shift Scheduling")

    # ── Sidebar: configurazione ───────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configurazione")

        use_case = st.radio(
            "Use Case",
            options=["A", "B"],
            format_func=lambda x: (
                "A — 13 lavoratori omogenei" if x == "A"
                else "B — 13 standard + 7 specializzati"
            ),
        )

        draft_path = os.path.join(ROOT, "input", f"model_draft_use_case_{use_case.lower()}.txt")
        workers_path = os.path.join(ROOT, "input", f"workers_use_case_{use_case.lower()}.json")

        use_llm = st.toggle("Usa LLaMA (richiede Ollama)", value=False)
        force_fallback = not use_llm

        st.divider()
        st.markdown("**Output**")
        st.caption("I file vengono salvati in `output/` dopo l'esecuzione.")

    # ── Tab principali ────────────────────────────────────────────────────────
    tab_input, tab_run, tab_results = st.tabs([
        "📄 Input", "▶️ Esecuzione", "📊 Risultati"
    ])

    # ── TAB 1: Input ──────────────────────────────────────────────────────────
    with tab_input:
        st.subheader("Model Draft Istituzionale")
        st.caption(
            "Descrive turni, forza lavoro e vincoli legali. "
            "Modificabile direttamente — salva prima di eseguire."
        )
        draft_content = st.text_area(
            "model_draft.txt",
            value=load_file_text(draft_path),
            height=280,
            key="draft_editor",
        )
        if st.button("💾 Salva model draft"):
            save_file_text(draft_path, draft_content)
            st.success("Model draft salvato.")

        st.divider()
        st.subheader("Preferenze Lavoratori")
        st.caption(
            "Modifica gli statement in linguaggio naturale. "
            "Con LLaMA puoi scrivere liberamente; con il fallback usa keyword "
            "come *prefer*, *avoid*, *morning*, *night*, *saturday off*, ecc."
        )
        workers_content = st.text_area(
            "workers.json",
            value=load_file_text(workers_path),
            height=380,
            key="workers_editor",
        )
        if st.button("💾 Salva preferenze"):
            save_file_text(workers_path, workers_content)
            st.success("Preferenze salvate.")

    # ── TAB 2: Esecuzione ─────────────────────────────────────────────────────
    with tab_run:
        st.subheader("Avvia il sistema")

        mode_label = "LLaMA (Ollama)" if use_llm else "Rule-based fallback"
        st.info(
            f"**Use Case {use_case}** · Modalità preferenze: **{mode_label}**\n\n"
            f"- Draft: `{os.path.basename(draft_path)}`\n"
            f"- Workers: `{os.path.basename(workers_path)}`"
        )

        run_btn = st.button(
            "▶ Avvia SmartScheduler",
            type="primary",
            use_container_width=True,
        )

        log_placeholder = st.empty()

        if run_btn:
            # Setup log handler
            handler = StreamlitLogHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S"
            ))
            root_logger = logging.getLogger()
            root_logger.addHandler(handler)
            root_logger.setLevel(logging.INFO)

            os.chdir(ROOT)
            os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)

            progress = st.progress(0, text="Inizializzazione…")

            try:
                # Stage 1
                progress.progress(10, text="Stage 1 — Raccolta preferenze…")
                from agents.preference_agent import load_and_extract
                workers, _ = load_and_extract(workers_path, force_fallback=force_fallback)
                st.success(f"✅ Stage 1 completato — {len(workers)} lavoratori processati")

                # Stage 2
                progress.progress(30, text="Stage 2 — Generazione schedule (OR-Tools)…")
                from input.model_draft_parser import parse_model_draft
                from agents.drafting_agent import solve
                draft = parse_model_draft(draft_path)

                model_export = os.path.join(ROOT, "output", "cp_model_partial.txt")
                schedule = solve(
                    workers=workers,
                    draft=draft,
                    model_export_path=model_export,
                )
                if schedule is None:
                    st.error("❌ Il solver non ha trovato soluzioni. Verifica i vincoli nel model draft.")
                    st.stop()
                st.success("✅ Stage 2 completato — schedule generato")

                # Stage 3
                progress.progress(60, text="Stage 3 — Verifica vincoli e fairness…")
                from agents.verification_agent import verify, evaluate_fairness
                is_valid, violations = verify(schedule, draft)

                if not is_valid:
                    st.error(f"❌ Stage 3: {len(violations)} violazioni rilevate:")
                    for v in violations:
                        st.markdown(f"- {v}")
                    st.stop()

                evaluate_fairness(schedule)
                st.success(
                    f"✅ Stage 3 completato — tutti i vincoli soddisfatti · "
                    f"Soddisfazione minima: **{schedule.min_satisfaction():.3f}**"
                )

                # Stage 4
                progress.progress(80, text="Stage 4 — Refinement Maximin…")
                from agents.refinement_agent import refine
                final_schedule = refine(schedule, draft)
                st.success(
                    f"✅ Stage 4 completato — "
                    f"Soddisfazione minima finale: **{final_schedule.min_satisfaction():.3f}**"
                )

                # Salva risultati in session_state per il tab Risultati
                st.session_state["final_schedule"] = final_schedule
                st.session_state["draft"] = draft

                # Salva CSV
                from output.schedule_output import export_to_csv
                csv_path = os.path.join(ROOT, "output", f"schedule_use_case_{use_case}.csv")
                export_to_csv(final_schedule, csv_path)

                progress.progress(100, text="Completato.")
                st.info("📊 Vai al tab **Risultati** per visualizzare lo schedule.")

            except Exception as e:
                st.error(f"Errore imprevisto: {e}")
                raise
            finally:
                root_logger.removeHandler(handler)

            # Log completo
            with st.expander("📋 Log completo"):
                st.code(handler.get_log_text(), language=None)

    # ── TAB 3: Risultati ──────────────────────────────────────────────────────
    with tab_results:
        if "final_schedule" not in st.session_state:
            st.info("Nessuno schedule disponibile. Vai al tab **Esecuzione** e avvia il sistema.")
        else:
            import pandas as pd
            schedule = st.session_state["final_schedule"]
            draft = st.session_state["draft"]

            st.subheader("Punteggi di Soddisfazione")
            scores = schedule.satisfaction_scores
            if scores:
                score_df = pd.DataFrame(
                    [{"Lavoratore": wid, "Soddisfazione": score}
                     for wid, score in sorted(scores.items(), key=lambda x: x[1])]
                )
                st.bar_chart(score_df.set_index("Lavoratore"), color="#2E75B6")

                min_w = schedule.least_satisfied_worker()
                min_s = schedule.min_satisfaction()
                max_s = max(scores.values())
                avg_s = sum(scores.values()) / len(scores)

                col1, col2, col3 = st.columns(3)
                col1.metric("Soddisfazione minima", f"{min_s:.3f}", f"Worker: {min_w}")
                col2.metric("Soddisfazione massima", f"{max_s:.3f}")
                col3.metric("Media", f"{avg_s:.3f}")

            st.divider()
            st.subheader("Schedule Completo")

            view = st.radio(
                "Visualizzazione",
                ["Tabella pivot (giorno × turno)", "Lista assegnazioni"],
                horizontal=True,
            )

            if view == "Tabella pivot (giorno × turno)":
                pivot = build_pivot(schedule)
                st.dataframe(pivot, use_container_width=True, hide_index=True)
            else:
                df = build_schedule_dataframe(schedule)
                st.dataframe(df, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Download")
            col1, col2 = st.columns(2)

            csv_path = os.path.join(ROOT, "output", f"schedule_use_case_{use_case}.csv")
            if os.path.exists(csv_path):
                with open(csv_path, "rb") as f:
                    col1.download_button(
                        "⬇️ Schedule CSV",
                        data=f,
                        file_name=os.path.basename(csv_path),
                        mime="text/csv",
                    )

            model_path = os.path.join(ROOT, "output", "cp_model_partial.txt")
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    col2.download_button(
                        "⬇️ CP-SAT Model",
                        data=f,
                        file_name="cp_model_partial.txt",
                        mime="text/plain",
                    )


if __name__ == "__main__":
    main()