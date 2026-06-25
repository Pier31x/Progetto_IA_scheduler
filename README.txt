SmartScheduler — Hospital Shift Scheduling System
===================================================

REQUIREMENTS
------------
  pip install ortools requests streamlit

  Optional (for LLM mode):
    Install Ollama: https://ollama.com
    Pull model:     ollama pull llama3

USAGE — Interfaccia web (consigliato)
--------------------------------------
  streamlit run app.py

  Apre il browser su http://localhost:8501 con:
  - Tab Input:      modifica model draft e preferenze lavoratori
  - Tab Esecuzione: avvia il sistema con barra di progresso
  - Tab Risultati:  grafico soddisfazione, pivot schedule, download CSV

USAGE — Terminale
------------------
  python main.py --use-case A --fallback
  python main.py --use-case B --fallback
  python main.py --use-case A              # con Ollama + LLaMA

PROJECT STRUCTURE
-----------------
  app.py                     Interfaccia web Streamlit
  main.py                    Entry point CLI
  input/
    model_draft_*.txt        Input istituzionale (turni, vincoli)
    model_draft_parser.py    Parser del model draft
    workers_*.json           Preferenze lavoratori in NL
  agents/
    preference_agent.py      Stage 1: NL -> Worker
    drafting_agent.py        Stage 2: OR-Tools solve
    verification_agent.py    Stage 3: verifica simbolica
    refinement_agent.py      Stage 4: loop Maximin
  solver/
    model_builder.py         Costruzione modello CP-SAT
    constraints.py           Vincoli hard
    fairness.py              Scoring e criterio Maximin
  models/
    worker.py / shift.py / schedule.py
  config/
    scenario.py
  output/
    cp_model_partial.txt     Modello CP-SAT leggibile (generato)
    schedule_use_case_*.csv  Schedule finale (generato)
