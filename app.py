"""
app.py — Interfaccia web Streamlit per SmartScheduler.

Flusso UI:
    Tab Input:     modifica model draft e statements lavoratori
    Tab Stage 1:   estrai preferenze via LLM → salva preferences.json
    Tab Scheduling: carica preferences.json → produce schedule finale
    Tab Confronto: Con preferenze LLM (preferenze LLM) vs Baseline senza preferenze (neutro) → metriche
    Tab Risultati: visualizza l'ultimo schedule prodotto

Il file preferences.json è il punto di disaccoppiamento:
    - Stage 1 lo produce (una volta sola, richiede LLaMA)
    - Scheduling e Confronto lo consumano (veloci, nessuna chiamata LLM)

Avvio: streamlit run app.py
"""

import json
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

def prefs_path() -> str:
    """Restituisce il percorso del JSON delle preferenze specifico per lo Use Case corrente."""
    uc = st.session_state.get("use_case", "A").lower()
    return os.path.join(ROOT, "output", f"preferences_use_case_{uc}.json")

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



# ── Calendario HTML ───────────────────────────────────────────────────────────

def render_calendar(schedule) -> str:
    from config.scenario import SCHEDULE_START, SCHEDULE_END
    days = all_days(SCHEDULE_START, SCHEDULE_END)

    day_data: dict[date, dict[str, list[str]]] = {}
    for (wid, day, shift), assigned in schedule.assignments.items():
        if assigned:
            day_data.setdefault(day, {}).setdefault(shift, []).append(wid)

    first_monday = days[0] - timedelta(days=days[0].weekday())
    last_sunday  = days[-1] + timedelta(days=(6 - days[-1].weekday()))
    weeks = []
    cur = first_monday
    while cur <= last_sunday:
        weeks.append([cur + timedelta(days=i) for i in range(7)])
        cur += timedelta(weeks=1)

    css = """<style>
    .cal-wrap{font-family:Arial,sans-serif;overflow-x:auto}
    .cal-table{border-collapse:collapse;width:100%;table-layout:fixed}
    .cal-table th{background:#1A3A6B;color:white;text-align:center;padding:8px 4px;font-size:13px;border:1px solid #ddd}
    .cal-cell{vertical-align:top;border:1px solid #ddd;padding:4px;min-height:90px;background:#fff}
    .cal-cell.out{background:#f9f9f9}
    .cal-day-num{font-size:12px;font-weight:bold;color:#555;margin-bottom:4px;text-align:right}
    .shift-block{border-radius:4px;padding:3px 5px;margin-bottom:3px;font-size:10px;border-left:3px solid;line-height:1.4}
    .shift-label{font-weight:bold;margin-bottom:2px}
    </style>"""

    rows = ""
    for week in weeks:
        rows += "<tr>"
        for day in week:
            in_range = SCHEDULE_START <= day <= SCHEDULE_END
            rows += f'<td class="cal-cell{" out" if not in_range else ""}">'
            rows += f'<div class="cal-day-num">{day.day}</div>'
            if in_range and day in day_data:
                for st_type in ["morning", "afternoon", "night"]:
                    wlist = day_data[day].get(st_type, [])
                    if wlist:
                        c = SHIFT_COLORS[st_type]
                        rows += (
                            f'<div class="shift-block" style="background:{c["bg"]};border-color:{c["border"]}">'
                            f'<div class="shift-label">{c["label"]}</div>'
                            f'<div>{", ".join(wlist)}</div></div>'
                        )
            rows += "</td>"
        rows += "</tr>"

    header = "".join(f"<th>{d}</th>" for d in ["Lun","Mar","Mer","Gio","Ven","Sab","Dom"])
    return f'{css}<div class="cal-wrap"><table class="cal-table"><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>'


def render_legend() -> str:
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;background:{c["bg"]};'
        f'border-left:4px solid {c["border"]};border-radius:4px;padding:4px 10px;'
        f'font-size:13px;font-family:Arial;margin-right:10px">{c["label"]}</span>'
        for c in SHIFT_COLORS.values()
    )
    return f'<div style="margin-bottom:12px">{items}</div>'


# ── UI ────────────────────────────────────────────────────────────────────────

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

        # Leggiamo direttamente dallo stato o salviamo nello stato al cambio
        use_case = st.radio(
            "Use Case",
            ["A", "B"],
            index=0 if st.session_state["use_case"] == "A" else 1,
            format_func=lambda x: (
                "A — 13 lavoratori omogenei" if x == "A"
                else "B — 13 standard + 7 specializzati"
            ),
            key="use_case_radio"  # chiave interna per il widget
        )

        # Aggiorna lo stato globale per gli altri Tab e gli Agenti di calcolo
        st.session_state["use_case"] = use_case

        # Generazione dinamica dei percorsi file basati sulla selezione corrente
        draft_path = os.path.join(ROOT, "input", f"model_draft_use_case_{use_case.lower()}.txt")
        workers_path = os.path.join(ROOT, "input", f"workers_use_case_{use_case.lower()}.json")

        st.divider()
        # Stato preferences.json
        if has_preferences():
            st.success("✅ preferences.json presente")
        else:
            st.warning("⚠️ preferences.json assente\nEsegui Stage 1 prima.")
        st.caption("Output salvato in `output/`")

    tabs = st.tabs(["📄 Input", "🤖 Stage 1 — Preferenze", "📅 Scheduling", "⚖️ Confronto", "📊 Risultati"])
    tab_input, tab_stage1, tab_scheduling, tab_compare, tab_results = tabs

    # ── TAB INPUT ─────────────────────────────────────────────────────────────
    with tab_input:
        st.subheader(f"Model Draft Istituzionale (Scenario {use_case})")
        st.caption("Turni, forza lavoro e vincoli legali. Salva prima di eseguire.")

        # FIX CRITICO: Il parametro key cambia dinamicamente includendo il nome dello use_case.
        # Questo costringe Streamlit a distruggere e ricreare il componente leggendo il nuovo file.
        draft_content = st.text_area(
            f"model_draft_use_case_{use_case.lower()}.txt",
            value=load_file(draft_path),
            height=260,
            key=f"draft_ed_{use_case.lower()}"  # <--- Chiave dinamica basata sullo Use Case
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

        # FIX CRITICO: Chiave dinamica applicata anche al file JSON degli statement
        workers_content = st.text_area(
            f"workers_use_case_{use_case.lower()}.json",
            value=load_file(workers_path),
            height=380,
            key=f"workers_ed_{use_case.lower()}"  # <--- Chiave dinamica basata sullo Use Case
        )
        if st.button("💾 Salva workers", key=f"btn_save_workers_{use_case.lower()}"):
            save_file(workers_path, workers_content)
            st.success(f"Dati dei lavoratori dello Scenario {use_case} salvati con successo.")

    # ── TAB STAGE 1 ───────────────────────────────────────────────────────────
    with tab_stage1:
        st.subheader("🤖 Stage 1 — Estrazione Preferenze via LLM")
        st.markdown(
            "Legge gli statement dal file workers e chiama **LLaMA** per formalizzare "
            "le preferenze di ogni lavoratore. Il risultato viene salvato in "
            "`output/preferences.json` e riutilizzato da tutti gli stadi successivi "
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
            "tramite OR-Tools CP-SAT + Refinement Maximin. **Non chiama LLaMA.**"
        )

        if not has_preferences():
            st.warning("⚠️ Esegui prima **Stage 1** per estrarre le preferenze.")
        else:
            prefs = load_prefs_summary()
            st.info(
                f"Preferenze caricate: **{len(prefs)} lavoratori** da `preferences.json`\n\n"
                f"Draft: `{os.path.basename(draft_path)}`"
            )

            if st.button("▶ Avvia Scheduling", type="primary",
                         use_container_width=True, disabled=not has_preferences()):
                handler = _attach_log_handler()
                os.chdir(ROOT)
                prog = st.progress(0, text="Carico preferenze...")
                try:
                    from agents.preference_agent import load_preferences
                    from input.model_draft_parser import parse_model_draft
                    from agents.drafting_agent import solve
                    from agents.verification_agent import verify, evaluate_fairness
                    from agents.refinement_agent import refine

                    workers = load_preferences(prefs_path())
                    draft   = parse_model_draft(draft_path)

                    prog.progress(20, text="Stage 2 — OR-Tools CP-SAT...")
                    model_out = os.path.join(ROOT, "output", "cp_model_partial.txt")
                    schedule = solve(workers=workers, draft=draft, model_export_path=model_out)
                    if schedule is None:
                        st.error("❌ Il solver non ha trovato soluzioni.")
                        st.stop()
                    st.success("✅ Stage 2 — Schedule generato")

                    prog.progress(60, text="Stage 3 — Verifica vincoli...")
                    is_valid, violations = verify(schedule, draft)
                    if not is_valid:
                        st.error(f"❌ {len(violations)} violazioni:")
                        for v in violations: st.markdown(f"- {v}")
                        st.stop()
                    evaluate_fairness(schedule)
                    st.success(f"✅ Stage 3 — Valido · min soddisfazione: **{schedule.min_satisfaction():.3f}**")

                    prog.progress(80, text="Stage 4 — Refinement Maximin...")
                    final = refine(schedule, draft)
                    st.success(f"✅ Stage 4 — Min finale: **{final.min_satisfaction():.3f}**")

                    st.session_state["final_schedule"] = final
                    st.session_state["use_case"] = use_case

                    from output.schedule_output import export_to_csv
                    export_to_csv(final, os.path.join(ROOT, "output", f"schedule_use_case_{use_case}.csv"))

                    prog.progress(100, text="Completato.")
                    st.balloons()
                    st.info("📊 Vai al tab **Risultati** per visualizzare lo schedule.")

                except Exception as e:
                    st.error(f"Errore: {e}")
                    raise
                finally:
                    _detach_log_handler(handler)

                with st.expander("📋 Log"):
                    st.code(handler.get_log_text(), language=None)

    # ── TAB CONFRONTO ─────────────────────────────────────────────────────────
    with tab_compare:
        st.subheader("⚖️ Confronto: LLM vs Preferenze Neutre")
        st.markdown(
            "Usa le stesse preferenze estratte da LLaMA per due run paralleli:\n"
            "- **Con preferenze LLM**: schedule con preferenze reali\n"
            "- **Baseline senza preferenze**: stesso modello, preferenze tutte neutre (baseline senza LLM)\n\n"
            "Il confronto sulle metriche oggettive dimostra quantitativamente "
            "il contributo dell'LLM."
        )

        if not has_preferences():
            st.warning("⚠️ Esegui prima **Stage 1** per estrarre le preferenze.")
        else:
            prefs = load_prefs_summary()
            st.info(f"Preferenze disponibili: **{len(prefs)} lavoratori**")

            if st.button("▶ Avvia Confronto", type="primary",
                         use_container_width=True, disabled=not has_preferences()):
                handler = _attach_log_handler()
                os.chdir(ROOT)
                prog = st.progress(0, text="Inizializzazione...")
                try:
                    from agents.evaluation_agent import run_comparison

                    prog.progress(10, text="Con preferenze LLM — Schedule con preferenze LLM...")
                    result = run_comparison(
                        preferences_path=prefs_path(),
                        draft_file=draft_path,
                        output_dir=os.path.join(ROOT, "output"),
                    )
                    prog.progress(100, text="Confronto completato.")
                    st.session_state["comparison"] = result
                    st.balloons()

                except Exception as e:
                    st.error(f"Errore: {e}")
                    raise
                finally:
                    _detach_log_handler(handler)

                with st.expander("📋 Log"):
                    st.code(handler.get_log_text(), language=None)

        if "comparison" in st.session_state:
            import pandas as pd
            import plotly.express as px  # Usiamo Plotly per eliminare il grafico a pila

            result = st.session_state["comparison"]
            ml = result.metrics_with_llm
            mn = result.metrics_neutral
            st.session_state["final_schedule"] = result.schedule_with_llm

            st.divider()
            st.header("🎯 Risultati del Confronto Globale")

            # SEZIONE 1: LE METRICHE CHIAVE (Soddisfazione e Violazioni prima di tutto)
            sat_llm = result.schedule_with_llm.satisfaction_scores
            sat_neu = result.schedule_neutral.satisfaction_scores
            avg_sat_llm = sum(sat_llm.values()) / len(sat_llm) if sat_llm else 0.0
            avg_sat_neu = sum(sat_neu.values()) / len(sat_neu) if sat_neu else 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Soddisfazione Media",
                      f"{avg_sat_llm:.3f}",
                      delta=f"{avg_sat_llm - avg_sat_neu:+.3f} vs Baseline (Senza LLM)")

            c2.metric("Violazioni con LLM",
                      f"{ml['violations']['global_total_violations']}",
                      delta=f"{ml['violations']['global_total_violations'] - mn['violations']['global_total_violations']}",
                      delta_color="inverse")

            c3.metric("Violazioni Senza LLM (Baseline)",
                      f"{mn['violations']['global_total_violations']}")

            # SEZIONE 2: IL GRAFICO DELLA SODDISFAZIONE (Barre affiancate)
            st.subheader("📊 Confronto Soddisfazione Individuale dei Lavoratori")
            st.caption(
                "Più la barra è alta, più il lavoratore è felice delle sue turnazioni. La baseline senza LLM è fissa a 0.5 (neutra).")

            rows_sat = []
            for w in result.workers_with_llm:
                rows_sat.append({"Lavoratore": w.worker_id, "Configurazione": "SmartScheduler (Con LLM)",
                                 "Soddisfazione": sat_llm.get(w.worker_id, 0.5)})
                rows_sat.append({"Lavoratore": w.worker_id, "Configurazione": "Baseline (Senza LLM)",
                                 "Soddisfazione": sat_neu.get(w.worker_id, 0.5)})

            df_sat = pd.DataFrame(rows_sat)
            fig_sat = px.bar(df_sat, x="Lavoratore", y="Soddisfazione", color="Configurazione",
                             barmode="group", color_discrete_map={"SmartScheduler (Con LLM)": "#2E75B6",
                                                                  "Baseline (Senza LLM)": "#A6A6A6"})
            fig_sat.update_layout(yaxis_range=[0, 1.05])
            st.plotly_chart(fig_sat, use_container_width=True)

            # SEZIONE 3: TABELLA DETTAGLIATA
            st.subheader("📋 Dettaglio Turni Notturni Assegnati")
            rows_table = []
            for w in result.workers_with_llm:
                wid = w.worker_id
                rows_table.append({
                    "Lavoratore": wid,
                    "Preferisce": ", ".join(w.preferred_shifts) or "—",
                    "Evita": ", ".join(w.avoid_shifts) or "—",
                    "Notti (Con LLM)": ml["night_counts"].get(wid, 0),
                    "Notti (Senza LLM)": mn["night_counts"].get(wid, 0),
                })
            st.dataframe(pd.DataFrame(rows_table), use_container_width=True, hide_index=True)

            # SEZIONE 4: METRICHE STRUTTURALI (Declassate in fondo per completezza accademica)
            with st.expander("📈 Visualizza Metriche di Bilanciamento Matematico (Opzionali)"):
                st.caption("Questi dati mostrano solo l'uniformità numerica dei turni, ignorando i desideri umani.")
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Dev. Standard Notti", f"{ml['std_nights']:.3f}",
                           delta=f"{ml['std_nights'] - mn['std_nights']:+.3f} vs neutro", delta_color="off")
                cc2.metric("Indice di Gini Notti", f"{ml['gini_nights']:.3f}",
                           delta=f"{ml['gini_nights'] - mn['gini_nights']:+.3f} vs neutro", delta_color="off")
                cc3.metric("Forbice Notti (Max-Min)", f"{ml['max_nights'] - ml['min_nights']}",
                           delta=f"{(ml['max_nights'] - ml['min_nights']) - (mn['max_nights'] - mn['min_nights'])} vs neutro",
                           delta_color="off")

            # Download
            st.divider()
            with open(prefs_path(), "rb") as f:
                st.download_button("⬇️ preferences.json", data=f, file_name="preferences.json", mime="application/json")

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