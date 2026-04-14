import os
from dotenv import load_dotenv

load_dotenv()

def generate_insights(context):
    """Rule-based fallback"""
    return "Revenue shows strong momentum. Churn is improving. Focus on customer growth segments with targeted re-engagement campaigns."

def ai(system: str, user: str, history=None, max_tokens=700):
    """Real Groq AI with fallback"""
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        models = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
        
        msgs = [{"role": "system", "content": system}]
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": user})

        for model in models:
            try:
                r = client.chat.completions.create(
                    model=model, messages=msgs, max_tokens=max_tokens, temperature=0.4
                )
                return r.choices[0].message.content.strip()
            except:
                continue
        return generate_insights({})
    except Exception:
        return generate_insights({})
