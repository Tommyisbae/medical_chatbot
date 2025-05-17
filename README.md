Medical Chatbot for Clinicians
==============================

A Python-based tool for licensed clinicians to collect patient history, analyze up to three chief complaints using the [SOCRATES framework](https://en.wikipedia.org/wiki/SOCRATES_(pain_assessment)), and generate SOAP notes with the 5Cs (Chief Complaint, Course, Cause/Aetiology, Complications, Care). Outputs include red flags, 3-5 differential diagnoses, and ICD-11 codes. Built with LangChain and Gemini 1.5 Flash, it’s targeting a React web app with PDF downloads and a blue-green UI.

**Features**: Clinician-only access, detailed SOCRATES (Site, Onset, Character, Radiation, Associated symptoms, Timing, Exacerbating/Relieving factors, Severity), 5Cs analysis, comprehensive history (PMH, meds, lifestyle, family, social), SOAP notes with red flags and differentials. Future: React frontend, PDF export.

**Prerequisites**: Python 3.13, Miniconda, Git, [Gemini API Key](https://aistudio.google.com/) (free tier, ~1M tokens/month).

**Installation**:

1.  Clone: git clone && cd medical\_chatbot
    
2.  Environment: conda create -n medical\_chatbot python=3.13 && conda activate medical\_chatbot && pip install -r requirements.txt
    
3.  API Key: echo "GEMINI\_API\_KEY=your\_api\_key\_here" > .env
    
4.  Run: python chatbot.py
    

**Usage**: Confirm clinician status (I’m a licensed clinician), enter complaints (e.g., Chest pain for 2 days), answer SOCRATES/history prompts. Get a SOAP note with Subjective (HPI, SOCRATES, history), Objective (red flags), Assessment (5Cs, differentials, ICD-11 like MD30).

**Testing**:

*   Scenario 1: Chest pain (2 days, substernal, sharp, radiates to left arm, dyspnea).
    
*   Scenario 2: Headache (3 days, bilateral, throbbing, nausea).
    
*   Scenario 3: Abdominal pain (1 week, post-meal, cramping, diarrhea).Run python chatbot.py, input scenario, check SOAP note.
    

**Notes**: Uses deprecated LangChain ConversationChain (0.2.7); migration to RunnableWithMessageHistory planned. Monitor Gemini API (~1.3 tokens/word) for free tier. .env excluded via .gitignore.

**Roadmap**: May-June 2025: React frontend, PDF downloads. July 2025: MVP with 8-10 scenarios. Future: Bayesian diagnostics, UI polish.

**License**: Clinician-use only. Contact developer for details.

**Contact**: File issues on [GitHub](https://github.com/).