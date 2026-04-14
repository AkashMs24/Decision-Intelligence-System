# utils/llm_engine.py
import os
import streamlit as st


# ✅ Detect small talk (prevents unnecessary insights)
def is_small_talk(text: str) -> bool:
    small_talk = [
        "hi", "hello", "hey", "how are you", "what's up",
        "good morning", "good evening"
    ]
    return text.lower().strip() in small_talk


def ai(system_prompt: str, user_prompt: str, history=None):
    """CEO Assistant AI - Clean, Smart, Production Ready"""

    # ✅ Handle greetings separately
    if is_small_talk(user_prompt):
        return "👋 Hello! I'm your CEO Assistant. Ask me about revenue, churn, growth, or strategy."

    # ✅ Load API key (Cloud + Local safe)
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

    if not api_key:
        return "❌ GROQ_API_KEY is missing. Add it in Streamlit Secrets."

    # ✅ Safe import
    try:
        from groq import Groq
    except ImportError:
        return "❌ 'groq' package not installed. Run: pip install groq"

    try:
        client = Groq(api_key=api_key)

        # ✅ Strong system prompt (controls behavior)
        final_system_prompt = f"""
You are a CEO-level AI business assistant.

Behavior Rules:
- If the user asks casual questions → respond like a human (NO business insights)
- If the user asks about business/data → give insights based ONLY on provided data
- Do NOT hallucinate numbers
- Be concise, structured, and actionable
- Use bullet points when suggesting strategies

Context:
{system_prompt}
"""

        messages = [{"role": "system", "content": final_system_prompt}]

        # ✅ Add history if exists
        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": user_prompt})

        # ✅ Groq API call
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # 🔥 best for your use case
            messages=messages,
            max_tokens=700,
            temperature=0.4,
        )

        # ✅ Logging
        print("Groq API called successfully")

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Groq Error: {str(e)[:180]}"
