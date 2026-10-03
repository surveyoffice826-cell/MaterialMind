"""Connects the agents into crews and runs them with retry + model fallback."""
import time

from crewai import Crew, Process

import material_normalizer
import pdf_extractor
import report_validator
from llm import MODELS, get_api_key, make_llm
from schemas import parse_rows

# (model, seconds to wait first): primary, primary again after a pause, then fallback model
ATTEMPTS = [(MODELS[0], 0), (MODELS[0], 30), (MODELS[1], 0)]


def _with_retry(job):
    if not get_api_key():
        raise RuntimeError("GROQ_API_KEY is missing (add it to .env or Streamlit Secrets).")
    last_error = None
    for model, wait in ATTEMPTS:
        if wait:
            time.sleep(wait)
        try:
            return job(make_llm(model)), model
        except Exception as exc:  # rate limit, bad JSON, model error -> try next option
            last_error = exc
    raise RuntimeError(f"Groq request failed after retries: {last_error}")


def extract_and_normalize(text, boq):
    """Agents 1 + 2. Returns (DataFrame of rows, model used)."""
    names = ", ".join(boq["material"].astype(str).tolist()[:150])

    def job(llm):
        a1, t1 = pdf_extractor.build(llm, text)
        a2, t2 = material_normalizer.build(llm, names, t1)
        crew = Crew(agents=[a1, a2], tasks=[t1, t2], process=Process.sequential, verbose=False)
        result = crew.kickoff()
        rows = parse_rows(getattr(result, "raw", None) or str(result))
        if not rows:
            raise ValueError("Agents returned no readable rows")
        return rows

    rows, model = _with_retry(job)
    return material_normalizer.post_clean(rows, boq), model


def run_validator(result_df, issues):
    """Agent 4 (LLM review). Returns (text, model used)."""
    def job(llm):
        agent, task = report_validator.build(llm, result_df.to_csv(index=False)[:3000], issues)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        result = crew.kickoff()
        return getattr(result, "raw", None) or str(result)

    return _with_retry(job)
