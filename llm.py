"""Groq LLM setup. Model names are auto-detected from your Groq account, so outdated names never break the app."""
import hashlib
import os
import re

from dotenv import load_dotenv

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
load_dotenv()

import requests  # noqa: E402
from crewai import LLM  # noqa: E402

GROQ_URL = "https://api.groq.com/openai/v1"  # Groq's OpenAI-compatible endpoint
DEFAULT_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]  # used only if auto-detect fails
SKIP = ("whisper", "tts", "guard", "orpheus", "compound", "safeguard", "embed", "playai")
_CACHE = {}


def get_api_key():
    """Key typed in the sidebar (this session only), else env/.env, else Streamlit Secrets."""
    st = None
    try:
        import streamlit as st
        key = st.session_state.get("user_api_key")
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


def _size(model_id: str) -> float:
    """Parameter size from the model name (70b -> 70, 8b -> 8)."""
    m = re.search(r"(\d+(?:\.\d+)?)b", model_id)
    return float(m.group(1)) if m else 0.0


def rank_models(ids):
    """Best big model first, then smaller/faster ones as fallbacks."""
    chat = [i for i in ids if not any(s in i for s in SKIP)]
    pool = [i for i in chat if "llama" in i] or chat
    if not pool:
        return []
    pool = sorted(pool, key=_size, reverse=True)
    return [pool[0]] + sorted(pool[1:], key=_size)


def get_models():
    """Ranked list of text models available for this API key."""
    key = get_api_key()
    if not key:
        return list(DEFAULT_MODELS)
    tag = hashlib.sha256(key.encode()).hexdigest()
    if tag not in _CACHE:
        try:
            r = requests.get(f"{GROQ_URL}/models", headers={"Authorization": f"Bearer {key}"}, timeout=15)
            r.raise_for_status()
            ids = [m["id"] for m in r.json().get("data", []) if m.get("active", True)]
            ranked = rank_models(ids)
        except Exception:
            ranked = []
        if not ranked:
            return list(DEFAULT_MODELS)
        _CACHE[tag] = ranked
    return list(_CACHE[tag])


def make_llm(model: str) -> LLM:
    key = get_api_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing (paste it in the sidebar, .env or Streamlit Secrets).")
    return LLM(model=f"openai/{model}", base_url=GROQ_URL, api_key=key,
               temperature=0, max_tokens=2048)
