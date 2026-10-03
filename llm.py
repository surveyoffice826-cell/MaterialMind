"""Groq LLM setup with two stable models (primary + fallback)."""
import os

from dotenv import load_dotenv

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
load_dotenv()

from crewai import LLM  # noqa: E402

GROQ_URL = "https://api.groq.com/openai/v1"  # Groq's OpenAI-compatible endpoint
   PRIMARY_MODEL = "bara-model-ID"
   FALLBACK_MODEL = "chhota-model-ID"
MODELS = [PRIMARY_MODEL, FALLBACK_MODEL]


def get_api_key():
    """Read the key from env/.env locally or from Streamlit secrets when deployed."""
    try:
        import streamlit as st
        key = st.session_state.get("user_api_key")  # typed in the sidebar (this session only)
        if key:
            return key
    except Exception:
        st = None
    key = os.getenv("GROQ_API_KEY")
    if not key and st is not None:
        try:
            key = st.secrets["GROQ_API_KEY"]
        except Exception:
            key = None
    return key


def make_llm(model: str) -> LLM:
    key = get_api_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing (add it to .env or Streamlit Secrets).")
    return LLM(model=f"openai/{model}", base_url=GROQ_URL, api_key=key,
               temperature=0, max_tokens=2048)
