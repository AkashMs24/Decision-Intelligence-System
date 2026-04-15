# utils/llm_engine.py
"""
LLM engine — uses Groq's chat completion API.
Falls back gracefully with a descriptive message instead of crashing.
"""
import os
import time
import streamlit as st


def _get_api_key() -> str | None:
    """Retrieve Groq API key from Streamlit Secrets or environment."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


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
                model="llama-3.1-8b-instant",   # fast & generous rate limits
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

            return f"❌ Unexpected error: {str(exc)[:200]}"

        attempt += 1

    return "⚠️ Could not get a response after retries. Please try again."
