# utils/llm_engine.py
import os
import streamlit as st

def ai(system_prompt: str, user_prompt: str, history=None):
    """Clean Groq call with better rate limit handling"""

    # Get API key from Streamlit Secrets first, then .env
    api_key = None
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "❌ Groq API key is missing. Please add GROQ_API_KEY in Streamlit Secrets."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        # Use smaller, faster model to reduce rate limit issues
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # Faster and higher limits
            messages=messages,
            max_tokens=600,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str or "rate limit" in error_str:
            return "⚠️ Groq rate limit reached. Please wait 30-60 seconds and try again."
        elif "invalid api key" in error_str or "authentication" in error_str:
            return "❌ Invalid Groq API key. Please check your key in Streamlit Secrets."
        else:
            return f"❌ Groq Error: {str(e)[:150]}"
