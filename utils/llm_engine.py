 utils/llm_engine.py
import os
import streamlit as st

def ai(system_prompt: str, user_prompt: str, history=None):
    """Clean Groq call - No old fallback logic"""

    # Get API key from Streamlit Secrets (preferred on cloud)
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "❌ GROQ_API_KEY is missing. Please add it in Streamlit Secrets."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=700,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Groq Error: {str(e)[:180]}"
