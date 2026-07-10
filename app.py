"""
app.py — Interfaccia web Streamlit per SmartScheduler.

Flusso UI:
    Tab Input:     modifica model draft e statements lavoratori
    Tab Stage 1:   estrai preferenze via LLM → salva preferences.json
    Tab Scheduling: carica preferences.json → produce schedule finale
    Tab Confronto: Scenario A (solo vincoli) vs B (vincoli+pref.+OR-Tools)
                   vs C (vincoli+pref.+LLM), preferenze estratte UNA sola
                   volta e condivise dai tre scenari → metriche di fairness
    Tab Demo:      visualizza risultati di Confronto GIA' PRECALCOLATI
                   (Use Case A o B) in modo istantaneo, senza rieseguire nulla
    Tab Risultati: visualizza l'ultimo schedule prodotto

Il file preferences.json è il punto di disaccoppiamento:
    - Stage 1 lo produce (una volta sola, richiede LLaMA)
    - Scheduling e Confronto lo consumano (veloci, nessuna chiamata LLM)

I risultati del Confronto vengono inoltre salvati su disco (pickle) per
Use Case, così da poterli richiamare istantaneamente dal tab Demo durante
una presentazione/esame, senza dover rieseguire l'intera pipeline.

Avvio: streamlit run app.py
"""

import json
import logging
import os
import pickle
import sys
from datetime import date, timedelta

import streamlit as st

from agents.drafting.llm_drafting import LLMDraftingAgent

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

st.set_page_config(page_title="SmartScheduler", page_icon="🏥", layout="wide")

SHIFT_COLORS = {
    "morning": {
        "label": "Mattina",
        "bg": "#1E3A8A",      # Blu Notte Intenso (Ottimo contrasto, professionale)
        "border": "#3B82F6",  # Blu Neon per il bordo sinistro
    },
    "afternoon": {
        "label": "Pomeriggio",
        "bg": "#B45309",    # Ambra/Arancione Bruciato (Caldo ma non accecante)
        "border": "#F59E0B", # Giallo Oro per il bordo sinistro
    },
    "night": {
        "label": "Notte",
        "bg": "#4C1D95",      # Viola Scuro Profondo (Rappresenta il turno notturno)
        "border": "#8B5CF6",  # Viola Fluido per il bordo sinistro
    }
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


def _attach_log_handler():
    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"
    ))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    return handler


def _detach_log_handler(handler):
    logging.getLogger().removeHandler(handler)


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

def prefs_path_for(uc: str) -> str:
    """Restituisce il percorso del JSON delle preferenze per uno Use Case specifico ('A' o 'B')."""
    return os.path.join(ROOT, "output", f"preferences_use_case_{uc.lower()}.json")

def prefs_path() -> str:
    """Restituisce il percorso del JSON delle preferenze specifico per lo Use Case corrente."""
    uc = st.session_state.get("use_case", "A")
    return prefs_path_for(uc)

def has_preferences() -> bool:
    """Verifica se esiste il file delle preferenze per lo Use Case corrente."""
    return os.path.exists(prefs_path())

def load_prefs_summary() -> list:
    """Carica le preferenze correnti dal file specifico dello Use Case attivo."""
    import json
    if not has_preferences():
        return []
    with open(prefs_path(), encoding="utf-8") as f:
        # Se il tuo file è una lista o ha una chiave interna, mantieni il tuo parsing originale.
        # Di solito: json.load(f) se salvi una lista, o json.load(f)["workers"]
        data = json.load(f)
        return data if isinstance(data, list) else data.get("workers", [])


def comparison_path_for(uc: str) -> str:
    """Percorso del file pickle con i risultati del Confronto A/B/C precalcolati per uno Use Case."""
    return os.path.join(ROOT, "output", f"comparison_use_case_{uc.lower()}.pkl")

def has_comparison(uc: str) -> bool:
    """Verifica se esistono risultati di Confronto già precalcolati e salvati per lo Use Case dato."""
    return os.path.exists(comparison_path_for(uc))

def save_comparison(uc: str, result) -> None:
    """Salva su disco i risultati del Confronto A/B/C per poterli richiamare istantaneamente in seguito."""
    os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
    with open(comparison_path_for(uc), "wb") as f:
        pickle.dump(result, f)

def load_comparison(uc: str):
    """Carica dal disco i risultati del Confronto A/B/C già precalcolati per lo Use Case dato."""
    with open(comparison_path_for(uc), "rb") as f:
        return pickle.load(f)


# ── Calendario HTML ───────────────────────────────────────────────────────────

def render_legend() -> str:
    """Genera la legenda dei turni compatibile sia con il tema chiaro che scuro."""
    # Definiamo colori ad alto contrasto con bordi netti per il tema scuro
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;background:{c["bg"]};'
        f'border-left:4px solid {c["border"]};border-radius:4px;padding:4px 10px;'
        f'font-size:13px;font-family:Arial;margin-right:10px;color:#FFFFFF;'  # Forza testo bianco per contrasto sul colore
        f'text-shadow: 1px 1px 1px rgba(0,0,0,0.8); font-weight:bold;">{c["label"]}</span>'
        for c in SHIFT_COLORS.values()
    )
    return f'<div style="margin-bottom:12px">{items}</div>'


def render_calendar(schedule) -> str:
    """
    Renderizza il calendario mantenendo lo stile e la struttura originale del progetto,
    ma applicando la nuova palette di colori intensi e ad alto contrasto.
    """
    from config.scenario import SCHEDULE_START, SCHEDULE_END

    days = all_days(SCHEDULE_START, SCHEDULE_END)

    html = """
    <style>
        .sched-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'Segoe UI', Arial, sans-serif;
            color: var(--text-color, #FFFFFF);
            background-color: transparent;
        }
        .sched-table th {
            background-color: rgba(128, 128, 128, 0.12);
            color: #4EA3E6; /* Azzurro coordinato per i giorni della settimana */
            padding: 10px;
            text-align: center;
            border: 1px solid rgba(128, 128, 128, 0.2);
            font-weight: bold;
            font-size: 13px;
        }
        .sched-table td {
            padding: 6px;
            text-align: left;
            border: 1px solid rgba(128, 128, 128, 0.2);
            vertical-align: top;
            height: 100px;
            width: 14.28%;
            background-color: rgba(128, 128, 128, 0.02);
        }
        .day-header {
            font-size: 12px;
            font-weight: bold;
            margin-bottom: 6px;
            color: var(--text-color, #FFFFFF);
            opacity: 0.9;
        }
        /* Stile per i singoli badge dei turni */
        .shift-item {
            border-radius: 4px; 
            padding: 4px 8px; 
            margin: 4px 0; 
            font-size: 11px;
            font-weight: 600; 
            color: #FFFFFF !important; /* Forza testo bianco nitido */
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8); /* Massima leggibilità su sfondi intensi */
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.15);
        }
    </style>
    <table class="sched-table">
        <tr>
            <th>Lun</th><th>Mar</th><th>Mer</th><th>Gio</th><th>Ven</th><th>Sab</th><th>Dom</th>
        </tr>
        <tr>
    """

    # Allineamento dinamico del primo giorno del mese
    start_idx = days.weekday() if hasattr(days, 'weekday') else days[0].weekday()
    for _ in range(start_idx):
        html += "<td></td>"

    current_idx = start_idx

    # Ciclo di popolazione delle celle con i dati dello schedule
    for day in days:
        if current_idx == 7:
            html += "</tr><tr>"
            current_idx = 0

        html += "<td>"
        html += f'<div class="day-header">{day.day}</div>'

        for s in ["morning", "afternoon", "night"]:
            assigned = schedule.get_shift_workers(day, s)
            if assigned:
                workers_str = ", ".join(assigned)
                c = SHIFT_COLORS[s]

                # Applichiamo lo stile originale arricchito con le nuove classi di contrasto
                html += (
                    f'<div class="shift-item" style="background:{c["bg"]}; border-left:4px solid {c["border"]};">'
                    f'<strong>{c["label"]}:</strong> {workers_str}'
                    f'</div>'
                )
        html += "</td>"
        current_idx += 1

    while current_idx < 7:
        html += "<td></td>"
        current_idx += 1

    html += "</tr></table>"
    return html


# ── Rendering condiviso dei risultati del Confronto A/B/C ─────────────────────

def render_comparison_results(result, uc: str, key_prefix: str) -> None:
    """
    Renderizza i risultati di un Confronto A/B/C (Sezioni 1-5), sia che provengano
    da un'esecuzione appena effettuata (tab Confronto), sia che siano stati
    caricati da un file pickle precalcolato (tab Demo).

    `uc` è lo Use Case a cui appartiene il risultato (serve per i download e per
    "Invia a Risultati"). `key_prefix` garantisce chiavi univoche ai widget quando
    la funzione viene richiamata da più tab nella stessa sessione.
    """
    import pandas as pd
    import plotly.express as px

    scenarios = [result.scenario_a, result.scenario_b, result.scenario_c]

    st.divider()
    st.header("🎯 Risultati del Confronto")

    # SEZIONE 1: metriche chiave affiancate (Soddisfazione ed Emergenze)
    cols = st.columns(3)
    for col, sc in zip(cols, scenarios):
        m = sc.metrics
        col.markdown(f"### {sc.label}")
        col.metric("Soddisfazione media", f"{m['avg_satisfaction']:.3f}")
        col.metric("Soddisfazione minima", f"{m['min_satisfaction']:.3f}")
        col.metric("Preferenze soddisfatte", f"{m['preference_satisfaction_pct']:.1f}%")
        #col.metric("Allineamento Straordinari", f"{m.get('emergency_alignment', 1.0) * 100:.1f}%")

    # SEZIONE 2: soddisfazione individuale per scenario
    st.subheader("📊 Confronto Soddisfazione Individuale dei Lavoratori")
    st.caption(
        "Le stesse preferenze reali (estratte una sola volta) sono usate "
        "per misurare la soddisfazione in tutti e tre gli scenari."
    )

    rows_sat = []
    for sc in scenarios:
        scores = sc.schedule.satisfaction_scores
        for w in result.workers:
            rows_sat.append({
                "Lavoratore": w.worker_id,
                "Scenario": sc.label,
                "Soddisfazione": scores.get(w.worker_id, 0.0),
            })

    df_sat = pd.DataFrame(rows_sat)
    fig_sat = px.bar(
        df_sat, x="Lavoratore", y="Soddisfazione", color="Scenario",
        barmode="group",
    )
    fig_sat.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig_sat, use_container_width=True, key=f"{key_prefix}_fig_sat")

    # SEZIONE 3: Distribuzione dei carichi di lavoro (Tabella Unica)
    st.subheader("🌙 ed 🏖️ Distribuzione di Notti, Festivi e Straordinari")
    st.caption("Pannello di controllo comparativo sui turni critici assegnati per ogni lavoratore.")

    rows_distribution = []
    for w in result.workers:
        for sc in scenarios:
            m = sc.metrics
            rows_distribution.append({
                "Lavoratore": w.worker_id,
                "Scenario": sc.label,
                "Turni Notturni": m["night_distribution"].get(w.worker_id, 0),
                "Turni Festivi": m.get("holiday_distribution", {}).get(w.worker_id, 0),
                # "Straordinari Assegnati": m.get("overtime_distribution", {}).get(w.worker_id, 0)
            })

    df_dist = pd.DataFrame(rows_distribution)
    st.dataframe(df_dist, use_container_width=True, hide_index=True, key=f"{key_prefix}_df_dist")

    # SEZIONE 4: Metriche strutturali ed equità assoluta (Semplificata)
    with st.expander("📈 Bilanciamento ed Equità dei Turni (KPI Semplici)"):
        rows_extra = []
        for sc in scenarios:
            m = sc.metrics
            rows_extra.append({
                "Scenario": sc.label,
                "Min Notti": m["min_nights"],
                "Max Notti": m["max_nights"],
                "Divario Notti": m.get("discrepancy_nights", m["max_nights"] - m["min_nights"]),
                "Min Festivi": m.get("min_holidays", "—"),
                "Max Festivi": m.get("max_holidays", "—"),
                "Totale Straordinari": m.get("total_overtime_shifts", "—"),
                "Worker Meno Soddisfatto": m["least_satisfied_worker"],
            })
        st.dataframe(pd.DataFrame(rows_extra), use_container_width=True, hide_index=True,
                     key=f"{key_prefix}_df_extra")

    # SEZIONE 5: invia uno scenario al tab Risultati
    st.divider()
    st.subheader("📤 Visualizza uno scenario nel tab Risultati")
    label_to_schedule = {sc.label: sc.schedule for sc in scenarios}
    chosen_label = st.selectbox(
        "Scegli lo scenario da visualizzare",
        list(label_to_schedule.keys()),
        key=f"{key_prefix}_compare_choice",
    )
    if st.button("Invia a Risultati", key=f"{key_prefix}_btn_send_to_results"):
        st.session_state["final_schedule"] = label_to_schedule[chosen_label]
        st.session_state["use_case"] = uc
        st.success(f"'{chosen_label}' inviato al tab Risultati.")

    # Download
    st.divider()
    pp = prefs_path_for(uc)
    if os.path.exists(pp):
        with open(pp, "rb") as f:
            st.download_button(
                "⬇️ preferences.json", data=f, file_name="preferences.json",
                mime="application/json", key=f"{key_prefix}_dl_prefs"
            )


# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    st.title("🏥 SmartScheduler")
    st.caption("Fair and Constraint-Aware Hospital Shift Scheduling")

    # 1. SINCRONIZZAZIONE DI SESSIONE: Inizializza o aggiorna lo Use Case globale
    if "use_case" not in st.session_state:
        st.session_state["use_case"] = "A"

    # Sidebar: solo selezione Use Case
    with st.sidebar:
        st.header("⚙️ Configurazione")

        # ── Use Case ─────────────────────────────────────────────
        use_case = st.radio(
            "Use Case",
            ["A", "B"],
            index=0 if st.session_state["use_case"] == "A" else 1,
            format_func=lambda x: (
                "A — 13 lavoratori omogenei"
                if x == "A"
                else "B — 13 standard + 7 specializzati"
            ),
            key="use_case_radio"
        )

        st.session_state["use_case"] = use_case

        # ── Drafting Agent ───────────────────────────────────────
        drafting_mode = st.radio(
            "Drafting Agent",
            ["Solver", "LLM"],
            index=0,
            help="Seleziona l'agente utilizzato nello Stage 2.",
            key="drafting_mode"
        )

        st.divider()

        # Percorsi file
        draft_path = os.path.join(
            ROOT,
            "input",
            f"model_draft_use_case_{use_case.lower()}.txt"
        )

        workers_path = os.path.join(
            ROOT,
            "input",
            f"workers_use_case_{use_case.lower()}.json"
        )

        # Stato preferences.json
        if has_preferences():
            st.success("✅ preferences.json presente")
        else:
            st.warning("⚠️ preferences.json assente\nEsegui Stage 1 prima.")

        # st.caption("Output salvato in `output/`")

    tabs = st.tabs([
        "📄 Input", "🤖 Preferenze", "📅 Scheduling", "📊 Risultati",
        "⚖️ Confronto", "🖥️ Demo",
    ])
    tab_input, tab_stage1, tab_scheduling, tab_results, tab_compare, tab_demo = tabs

    # ── TAB INPUT ─────────────────────────────────────────────────────────────
    with tab_input:
        st.subheader(f"Model Draft Istituzionale (Scenario {use_case})")
        st.caption("Turni, forza lavoro e vincoli legali. Salva prima di eseguire.")

        # Il parametro key cambia dinamicamente includendo il nome dello use_case.
        # Questo costringe Streamlit a distruggere e ricreare il componente leggendo il nuovo file.
        draft_content = st.text_area(
            f"model_draft_use_case_{use_case.lower()}.txt",
            value=load_file(draft_path),
            height=260,
            key=f"draft_ed_{use_case.lower()}"
        )
        if st.button("💾 Salva model draft", key=f"btn_save_draft_{use_case.lower()}"):
            save_file(draft_path, draft_content)
            st.success(f"File dello Scenario {use_case} salvato con successo.")

        st.divider()
        st.subheader(f"Statement Lavoratori (Scenario {use_case})")
        st.caption(
            "Scrivi gli statement liberamente in italiano o inglese. "
            "LLaMA interpreterà il testo e estrarrà le preferenze."
        )

        # Chiave dinamica applicata anche al file JSON degli statement
        workers_content = st.text_area(
            f"workers_use_case_{use_case.lower()}.json",
            value=load_file(workers_path),
            height=380,
            key=f"workers_ed_{use_case.lower()}"
        )
        if st.button("💾 Salva workers", key=f"btn_save_workers_{use_case.lower()}"):
            save_file(workers_path, workers_content)
            st.success(f"Dati dei lavoratori dello Scenario {use_case} salvati con successo.")

    # ── TAB STAGE 1 (estrazione preferenze) ───────────────────────────────────────────────────────────
    with tab_stage1:
        st.subheader("🤖 Estrazione Preferenze via LLM")
        st.markdown(
            "Legge gli statement dal file workers e chiama **LLaMA** per formalizzare "
            "le preferenze di ogni lavoratore. Il risultato viene salvato in "
            "`output/preferences.json` e riutilizzato da tutti gli stadi successivi "
            "(incluso il tab Confronto, che lo estrae una sola volta) "
            "senza chiamare l'LLM di nuovo."
        )

        if has_preferences():
            prefs = load_prefs_summary()
            st.success(
                f"✅ preferences.json già presente — {len(prefs)} lavoratori. "
                "Premi **Riesegui** se vuoi aggiornare le preferenze."
            )
            # Mostra anteprima
            import pandas as pd
            rows = []
            for w in prefs:
                rows.append({
                    "ID": w["worker_id"],
                    "Ruolo": w["role"],
                    "Preferisce": ", ".join(w.get("preferred_shifts", [])) or "—",
                    "Evita": ", ".join(w.get("avoid_shifts", [])) or "—",
                    "Giorni off": ", ".join(w.get("preferred_days_off", [])) or "—",
                    "Night tol.": w.get("night_tolerance", 0.5),
                    "Holiday tol.": w.get("holiday_tolerance", 0.5),
                    "Emergency Availability": w.get("emergency_availability", 0)
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        run_stage1  = col1.button("▶ Esegui Stage 1 (LLaMA)", type="primary", use_container_width=True)
        rerun_stage1 = col2.button("🔄 Riesegui Stage 1", use_container_width=True,
                                   disabled=not has_preferences())

        if run_stage1 or rerun_stage1:
            # Se il file dello Use Case ATTIVO esiste già e l'utente ha premuto il tasto di prima esecuzione
            if has_preferences() and run_stage1:
                st.info(
                    f"ℹ️ Le preferenze per lo Scenario {use_case} sono già presenti. "
                    "Usa il tasto **🔄 Riesegui Stage 1** a destra per sovrascriverle."
                )
            else:
                handler = _attach_log_handler()
                os.chdir(ROOT)
                os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
                prog = st.progress(0, text="Stage 1 — Connessione a LLaMA...")
                try:
                    from agents.preference_agent import extract_and_save
                    workers_out = []
                    import json as _json
                    with open(workers_path, encoding="utf-8") as _f:
                        worker_inputs = _json.load(_f)["workers"]

                    from agents.preference_agent import extract_all_preferences, save_preferences
                    backend_ref = [None]

                    # Mostra progresso per lavoratore
                    status = st.empty()
                    from agents.preference_agent import _get_available_backend, _apply_backend
                    backend = _get_available_backend()
                    if backend is None:
                        st.error("❌ Nessun backend LLM disponibile. Avvia Ollama e scarica un modello.")
                        st.stop()

                    for i, entry in enumerate(worker_inputs):
                        status.info(f"Lavoratore {i+1}/{len(worker_inputs)}: **{entry['worker_id']}**...")
                        prog.progress(int(100 * i / len(worker_inputs)),
                                      text=f"Stage 1 — {i+1}/{len(worker_inputs)} lavoratori")
                        w = _apply_backend(entry["worker_id"], entry["role"], entry["statement"], backend)
                        workers_out.append(w)

                    # Salva nel percorso dinamico specifico (preferences_use_case_a.json o b.json)
                    save_preferences(workers_out, prefs_path())
                    status.empty()
                    prog.progress(100, text="Stage 1 completato.")
                    st.success(f"✅ Preferenze estratte per {len(workers_out)} lavoratori dello Scenario {use_case} salvate.")
                    st.rerun()

                except Exception as e:
                    st.error(f"Errore: {e}")
                    raise
                finally:
                    _detach_log_handler(handler)

                with st.expander("📋 Log"):
                    st.code(handler.get_log_text(), language=None)

    # ── TAB SCHEDULING ────────────────────────────────────────────────────────
    with tab_scheduling:
        st.subheader("📅 Scheduling")

        st.markdown(
            "Carica le preferenze da `preferences.json` e produce lo schedule finale "
            "tramite OR-Tools CP-SAT + Refinement Maximin, oppure tramite Drafting "
            "LLM con riparazione iterativa. **Non chiama LLaMA per l'estrazione "
            "preferenze**, già fatta nello Stage 1."
        )

        if not has_preferences():
            st.warning("⚠️ Esegui prima **Stage 1** per estrarre le preferenze.")
        else:
            prefs = load_prefs_summary()
            st.info(
                f"Preferenze caricate: **{len(prefs)} lavoratori** da `preferences.json`\n\n"
                f"Draft: `{os.path.basename(draft_path)}`"
            )

            if st.button(
                    "▶ Avvia Scheduling",
                    type="primary",
                    use_container_width=True,
                    disabled=not has_preferences()
            ):

                # ── LOG HANDLER ─────────────────────────────
                handler = _attach_log_handler()
                os.chdir(ROOT)

                prog = st.progress(0, text="Carico preferenze...")

                try:
                    # ── IMPORT LOCALI ─────────────────────────
                    from agents.preference_agent import load_preferences
                    from input.model_draft_parser import parse_model_draft
                    from agents.drafting.solver_drafting import SolverDraftingAgent
                    from agents.drafting.llm_drafting import LLMDraftingAgent
                    from agents.verification_agent import verify, evaluate_fairness
                    from agents.refinement_agent import refine
                    from output.schedule_output import export_to_csv

                    # ── DATA LOAD ─────────────────────────────
                    workers = load_preferences(prefs_path())
                    draft = parse_model_draft(draft_path)

                    prog.progress(20, text="Stage 2 — Drafting...")

                    model_out = os.path.join(ROOT, "output", "cp_model_partial.txt")

                    # ── AGENT SELECTION ───────────────────────
                    mode_label = st.session_state.get("drafting_mode", "Solver")

                    if mode_label == "Solver":
                        drafting_agent = SolverDraftingAgent()
                    else:
                        drafting_agent = LLMDraftingAgent()

                    st.info(f"Drafting attivo: **{mode_label}**")

                    # ── STAGE 2 ───────────────────────────────
                    #print("-----DEBUG APP.PY (490)", workers)
                    schedule = drafting_agent.solve(
                        workers=workers,
                        draft=draft,
                        model_export_path=model_out
                    )

                    if schedule is None:
                        st.error("❌ Nessuna soluzione trovata dal Drafting Agent.")
                        with st.expander("📋 Log di questo tentativo", expanded=True):
                            st.code(handler.get_log_text(), language=None)
                        st.stop()

                    st.success("✅ Stage 2 — Schedule generato")

                    # ── STAGE 3 ───────────────────────────────
                    prog.progress(60, text="Stage 3 — Verifica vincoli...")

                    report = verify(schedule, draft)

                    if not report.feasible:
                        st.error(f"❌ {report.total_violations} violazioni:")
                        for v in report.all_violations:
                            st.markdown(f"- {v}")
                        if report.suggestions:
                            st.info("Suggerimenti del Verification Agent:")
                            for s in report.suggestions:
                                st.markdown(f"- {s}")
                        with st.expander("📋 Log di questo tentativo", expanded=True):
                            st.code(handler.get_log_text(), language=None)
                        st.stop()

                    evaluate_fairness(schedule)

                    st.success(
                        f"✅ Stage 3 — Valido · min soddisfazione: "
                        f"**{schedule.min_satisfaction():.3f}**"
                    )

                    # ── STAGE 4 ───────────────────────────────
                    prog.progress(80, text="Stage 4 — Refinement Maximin...")

                    final = refine(schedule, drafting_agent, draft)

                    st.success(
                        f"✅ Stage 4 — Min finale: **{final.min_satisfaction():.3f}**"
                    )

                    # ── SAVE ────────────────────────────────
                    st.session_state["final_schedule"] = final
                    st.session_state["use_case"] = use_case

                    export_to_csv(
                        final,
                        os.path.join(ROOT, "output", f"schedule_use_case_{use_case}.csv")
                    )

                    prog.progress(100, text="Completato.")
                    st.balloons()

                    st.info("📊 Vai al tab **Risultati** per visualizzare lo schedule.")

                except Exception as e:
                    st.error(f"Errore: {e}")
                    raise

                finally:
                    # ── LOG FINAL CAPTURE ─────────────────────
                    log_text = handler.get_log_text()
                    _detach_log_handler(handler)

                    st.session_state["last_logs"] = log_text

            # ── LOG VIEWER (STABILE FUORI DAL BUTTON) ─────
            if "last_logs" in st.session_state:
                with st.expander("📋 Log dell’ultima esecuzione", expanded=False):
                    st.code(st.session_state["last_logs"], language=None)

    # ── TAB CONFRONTO ─────────────────────────────────────────────────────────
    with tab_compare:
        st.subheader("⚖️ Confronto a tre scenari: A / B / C")
        st.markdown(
            "Usa le **steste preferenze**, estratte da LLaMA **una sola volta**, "
            "per tre run indipendenti che cambiano SOLO la strategia di drafting:\n\n"
            "- **Scenario A** — solo vincoli hard, nessuna preferenza nell'obiettivo\n"
            "- **Scenario B** — vincoli hard + preferenze + Drafting **OR-Tools**\n"
            "- **Scenario C** — vincoli hard + preferenze + Drafting **LLM** (con "
            "riparazione iterativa)\n\n"
            "Le metriche di fairness sono calcolate sugli stessi lavoratori "
            "(le stesse preferenze reali) in tutti e tre gli scenari, così il "
            "confronto isola l'unica variabile che cambia: il Drafting Agent."
        )

        if not has_preferences():
            st.warning("⚠️ Esegui prima **Stage 1** per estrarre le preferenze.")
        else:
            prefs = load_prefs_summary()
            st.info(
                f"Preferenze disponibili: **{len(prefs)} lavoratori** "
                "(condivise da tutti e tre gli scenari)"
            )

            if st.button("▶ Avvia Confronto A / B / C", type="primary",
                         use_container_width=True, disabled=not has_preferences()):
                handler = _attach_log_handler()
                os.chdir(ROOT)
                prog = st.progress(0, text="Scenario A — solo vincoli hard...")
                try:
                    from agents.evaluation_agent import run_three_way_comparison

                    result = run_three_way_comparison(
                        preferences_path=prefs_path(),
                        draft_file=draft_path,
                        output_dir=os.path.join(ROOT, "output"),
                    )
                    prog.progress(100, text="Confronto completato.")
                    st.session_state["comparison3"] = result
                    st.session_state["comparison3_uc"] = use_case

                    # Salva su disco così da poterlo richiamare istantaneamente
                    # dal tab Demo durante l'esame, senza rieseguire nulla.
                    save_comparison(use_case, result)

                    st.balloons()

                except Exception as e:
                    st.error(f"Errore: {e}")
                    raise
                finally:
                    _detach_log_handler(handler)

                with st.expander("📋 Log"):
                    st.code(handler.get_log_text(), language=None)

        if "comparison3" in st.session_state:
            render_comparison_results(
                st.session_state["comparison3"],
                uc=st.session_state.get("comparison3_uc", use_case),
                key_prefix="cmp",
            )

    # ── TAB DEMO (risultati precompilati) ────────────────────────────────────
    with tab_demo:
        st.subheader("🖥️ Demo (risultati precompilati)")
        st.markdown(
            "Mostra **istantaneamente** i risultati del Confronto A/B/C già "
            "calcolati in precedenza per lo Use Case scelto, esattamente come "
            "nel tab **Confronto**, senza rieseguire nulla.\n\n"
            "Per popolare questa scheda, nel tab **Confronto**, "
            "seleziona lo Use Case desiderato nella sidebar ed esegui il "
            "Confronto: il risultato viene salvato automaticamente e resterà "
            "disponibile qui anche dopo un riavvio dell'app."
        )

        demo_uc = st.radio(
            "Use Case da visualizzare",
            ["A", "B"],
            horizontal=True,
            key="demo_uc_choice",
            format_func=lambda x: (
                "A — 13 lavoratori omogenei"
                if x == "A"
                else "B — 13 standard + 7 specializzati"
            ),
        )

        if not has_comparison(demo_uc):
            st.warning(
                f"⚠️ Nessun risultato precompilato per lo Use Case **{demo_uc}**. "
                f"Vai al tab **Confronto**, seleziona lo Use Case {demo_uc} nella "
                "sidebar ed esegui il Confronto almeno una volta."
            )
        else:
            demo_result = load_comparison(demo_uc)
            st.success(f"✅ Risultati precompilati per lo Use Case {demo_uc} caricati.")
            render_comparison_results(demo_result, uc=demo_uc, key_prefix=f"demo_{demo_uc.lower()}")

    # ── TAB RISULTATI ─────────────────────────────────────────────────────────
    with tab_results:
        if "final_schedule" not in st.session_state:
            st.info("Nessuno schedule disponibile. Vai al tab **Scheduling** e avvia il sistema.")
            return

        import pandas as pd
        schedule = st.session_state["final_schedule"]
        uc_label = st.session_state.get("use_case", "A")
        scores   = schedule.satisfaction_scores

        st.subheader("Punteggi di Soddisfazione")
        min_w = schedule.least_satisfied_worker()
        c1, c2, c3 = st.columns(3)
        c1.metric("Minima",  f"{schedule.min_satisfaction():.3f}", f"Worker: {min_w}")
        c2.metric("Massima", f"{max(scores.values()):.3f}")
        c3.metric("Media",   f"{sum(scores.values())/len(scores):.3f}")

        score_df = pd.DataFrame(
            [{"Lavoratore": wid, "Soddisfazione": s}
             for wid, s in sorted(scores.items(), key=lambda x: x[1])]
        )
        st.bar_chart(score_df.set_index("Lavoratore"), color="#2E75B6")

        st.divider()
        st.subheader("📅 Calendario")
        st.html(render_legend())
        st.html(render_calendar(schedule))

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

        st.divider()
        st.subheader("Download")
        c1, c2 = st.columns(2)
        csv_path = os.path.join(ROOT, "output", f"schedule_use_case_{uc_label}.csv")
        if os.path.exists(csv_path):
            c1.download_button("⬇️ Schedule CSV", open(csv_path,"rb"),
                               file_name=os.path.basename(csv_path), mime="text/csv")
        model_path = os.path.join(ROOT, "output", "cp_model_partial.txt")
        if os.path.exists(model_path):
            c2.download_button("⬇️ CP-SAT Model", open(model_path,"rb"),
                               file_name="cp_model_partial.txt", mime="text/plain")


if __name__ == "__main__":
    main()