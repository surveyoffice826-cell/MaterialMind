# MaterialMind
AI multi-agent app (CrewAI + Groq + Streamlit): upload a daily material report PDF, get
**received today / total received / remaining**, review and correct by hand, then approve.

## Agents
1. **PDF Extraction** (LLM) - reads rows from the PDF
2. **Normalizer** (LLM + Python) - matches names/units with the BOQ
3. **Stock Calculator** (pure Python) - all arithmetic, no LLM guessing
4. **Validator** (rules + LLM) - flags mistakes
5. **You** - edit the table and press Approve

Models: `llama-3.3-70b-versatile` (primary) with automatic fallback to `llama-3.1-8b-instant`.

## Files
`app.py` UI | `crew.py` runs agents + retry | `llm.py` Groq setup | `pdf_extractor.py`,
`material_normalizer.py`, `stock_calculator.py`, `report_validator.py` agents | `schemas.py` data models |
`boq.csv` sample required materials

## Run locally
```
pip install -r requirements.txt
echo GROQ_API_KEY=your_key > .env
streamlit run app.py
```

## Upload to GitHub
1. github.com -> New repository `materialmind` (Public or Private).
2. "Add file" -> "Upload files" -> drag all project files (not `.env`) -> Commit.

## Deploy on Streamlit
1. share.streamlit.io -> New app -> choose your repo, branch `main`, main file `app.py`.
2. Advanced settings -> Python **3.12**. API key: either paste it in the app sidebar each session, or save it once in Secrets: `GROQ_API_KEY = "your_key"`
3. Deploy.

## BOQ format
`material,unit,total_required` (CSV or Excel). Upload it in the sidebar for each project.

## Notes / limits
- Text PDFs only (scanned photos are not supported).
- Groq free tier has a small tokens-per-minute limit; long reports are cut at 8000 characters.
  If a request is rate-limited the app waits and retries, then switches to the fallback model.
- Streamlit Cloud may reset files on restart: download `records.csv` after each approval and
  restore it from the sidebar when needed.
- Unit conversion only for known pairs (kg/ton, cft/m3, ltr/m3, rft/mtr); others are flagged.
