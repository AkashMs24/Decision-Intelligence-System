# utils/llm_engine.py
"""
LLM engine — uses Groq's chat completion API.
Falls back gracefully with a descriptive message instead of crashing.
"""
import os
import time
import streamlit as st

# Model name — verified current as of Aug 2026.
# llama-3.1-8b-instant and llama-3.3-70b-versatile were deprecated by Groq
# (announced June 17, 2026). openai/gpt-oss-20b is their recommended
# fast/cheap replacement. Override via GROQ_MODEL secret/env if Groq
# retires this one too — check https://console.groq.com/docs/models.
DEFAULT_MODEL = "openai/gpt-oss-20b"


def _get_api_key() -> str | None:
    """Retrieve Groq API key from Streamlit Secrets or environment."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


def _get_model() -> str:
    """Retrieve Groq model name from Streamlit Secrets or environment, with a safe default."""
    try:
        return st.secrets["GROQ_MODEL"]
    except Exception:
        return os.getenv("GROQ_MODEL", DEFAULT_MODEL)


def ai(
    system_prompt: str,
    user_prompt: str,
    history: list[dict] | None = None,
    max_tokens: int = 1_024,
    retries: int = 2,
) -> str:
    """
    Send a chat request to Groq and return the response text.

    Parameters
    ----------
    system_prompt : str
        Instructions / persona for the assistant.
    user_prompt : str
        The user's message.
    history : list[dict] | None
        Prior conversation turns [{"role": ..., "content": ...}, ...].
    max_tokens : int
        Upper limit on response tokens (raised to 1 024 for richer answers).
    retries : int
        Number of automatic retries on rate-limit errors.

    Returns
    -------
    str
        The assistant's reply, or an error message starting with ❌ / ⚠️.
    """
    api_key = _get_api_key()
    if not api_key:
        return (
            "❌ Groq API key not found. "
            "Please add **GROQ_API_KEY** in your Streamlit Secrets (`.streamlit/secrets.toml`)."
        )

    model = _get_model()

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        # Keep only the last 10 turns to stay within context limits
        messages.extend(history[-10:])
    messages.append({"role": "user", "content": user_prompt})

    attempt = 0
    while attempt <= retries:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=model,   # fast & generous rate limits
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.5,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            err = str(exc).lower()
            if "429" in err or "rate limit" in err:
                if attempt < retries:
                    wait = 20 * (attempt + 1)   # 20 s, then 40 s
                    time.sleep(wait)
                    attempt += 1
                    continue
                return (
                    "⚠️ Groq rate limit reached. "
                    "Please wait a minute and try again — Groq's free tier has per-minute limits."
                )
            if "invalid api key" in err or "authentication" in err or "401" in err:
                return "❌ Invalid Groq API key. Please verify your key in Streamlit Secrets."
            if "503" in err or "unavailable" in err:
                return "⚠️ Groq service is temporarily unavailable. Please retry in a moment."
            if "decommissioned" in err or "model_decommissioned" in err or ("404" in err and "model" in err):
                return (
                    f"❌ The model '{model}' is no longer available on Groq. "
                    "Update GROQ_MODEL in your Streamlit Secrets — check "
                    "https://console.groq.com/docs/models for a current model ID."
                )
            return f"❌ Unexpected error: {str(exc)[:200]}"
        attempt += 1
    return "⚠️ Could not get a response after retries. Please try again."
