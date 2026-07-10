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
    agents/
        /drafting
        base.py             Interfaccia per i drafter
        llm_drafting        Drafting che usa l'LLM
        prompt_builder      File usato da llm_drafting per gestire i prompt
        scheduler_drafting  Drafting che usa il solver
        scheduler_parser    Prende
    preference_agent.py      Stage 1: NL -> Worker
    drafting_agent.py        Stage 2: OR-Tools solve
    verification_agent.py    Stage 3: verifica simbolica
    refinement_agent.py      Stage 4: loop Maximin
  input/
    model_draft_parser.py    Parser del model draft (i vincoli hard)
    model_draft_use_case_*.txt
    workers_*.json           Preferenze lavoratori in NL
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

PER ESEGUIRE
---------------
streamlit run app.py (va eseguito su terminale)
