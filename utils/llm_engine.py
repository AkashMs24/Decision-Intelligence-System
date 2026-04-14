# utils/llm_engine.py
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_groq_key():
    """Support both local .env and Streamlit Cloud Secrets"""
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return os.getenv("GROQ_API_KEY")

def ai(system_prompt: str, user_prompt: str, history=None, max_tokens=700):
    api_key = get_groq_key()
    
    if not api_key:
        return "⚠️ Groq API key is missing. Please add GROQ_API_KEY in .env or Streamlit Secrets."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        # Try best models first
        models = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"]

        for model in models:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.4,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                err = str(e).lower()
                if "rate limit" in err or "429" in err:
                    return "⚠️ Rate limit reached. Please wait a moment and try again."
                if "not found" in err or "deprecated" in err:
                    continue  # try next model
                else:
                    break

        return "⚠️ Groq models failed. Falling back to basic insight."

    except ImportError:
        return "⚠️ Groq library not installed. Run: pip install groq"
    except Exception as e:
        return f"⚠️ Groq error: {str(e)[:120]}...\n\nPlease check your API key and internet."
