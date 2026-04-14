# utils/llm_engine.py
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def ai(system_prompt: str, user_prompt: str, history=None):
    """Clean Groq call - NO old fallback code at all"""

    # Get API key
    api_key = None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "❌ Missing Groq API Key.\n\nPlease add GROQ_API_KEY in Streamlit Secrets or .env file."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # fast model for testing
            messages=messages,
            max_tokens=600,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Groq Error:\n{str(e)[:200]}"
