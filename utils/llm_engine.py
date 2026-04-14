# utils/llm_engine.py
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_groq_key():
    try:
        # For Streamlit Cloud
        return st.secrets["GROQ_API_KEY"]
    except:
        # For local .env
        return os.getenv("GROQ_API_KEY")

def ai(system_prompt: str, user_prompt: str, history=None):
    """Clean Groq call - NO old fallback logic"""
    
    api_key = get_groq_key()
    
    if not api_key:
        return "❌ Groq API key is missing. Please add it correctly in Streamlit Secrets or .env file."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",   # Best model
            messages=messages,
            max_tokens=700,
            temperature=0.4,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        error_msg = str(e).lower()
        if "invalid api key" in error_msg or "authentication" in error_msg:
            return "❌ Invalid Groq API key. Please check your key in Streamlit Secrets."
        elif "rate limit" in error_msg:
            return "⚠️ Rate limit reached. Please wait 10 seconds and try again."
        else:
            return f"❌ Groq error: {str(e)[:100]}...\n\nPlease check your API key."
