# utils/llm_engine.py
import os
import streamlit as st

def ai(system_prompt: str, user_prompt: str, history=None):
    """Clean Groq call - production ready"""

    # ✅ Safe API key loading (Cloud + Local)
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

    if not api_key:
        return "❌ GROQ_API_KEY is missing. Add it in Streamlit Secrets."

    try:
        # ✅ Safe import
        from groq import Groq
    except ImportError:
        return "❌ 'groq' package not installed. Run: pip install groq"

    try:
        client = Groq(api_key=api_key)

        messages = [{"role": "system", "content": system_prompt}]
        
        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # 🔥 best for your CEO assistant
            messages=messages,
            max_tokens=700,
            temperature=0.4,
        )

        # ✅ Logging (for debugging / verification)
        print("Groq API called successfully")

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Groq Error: {str(e)[:180]}"
