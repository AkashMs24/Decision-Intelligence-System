import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()   # for local development

def get_groq_key():
    """Get Groq key from Streamlit Secrets or .env"""
    # First try Streamlit Secrets (for cloud)
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        # Then try .env (for local)
        return os.getenv("GROQ_API_KEY")

def ai(system_prompt: str, user_prompt: str, history=None, max_tokens=700):
    api_key = get_groq_key()
    
    if not api_key:
        return "⚠️ Groq API key is missing. Please add it in .env (local) or Streamlit Secrets (cloud)."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        
        models = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"]
        
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        for model in models:
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.4
                )
                return r.choices[0].message.content.strip()
            except:
                continue
                
        return "⚠️ Groq models failed. Please try again."
        
    except Exception as e:
        return f"⚠️ Error: {str(e)[:80]}..."
