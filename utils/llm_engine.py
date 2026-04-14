import os
from dotenv import load_dotenv

# 🔥 LOAD ENV FROM ROOT
load_dotenv()

# -----------------------------
# 🧠 RULE-BASED INSIGHTS
# -----------------------------
def generate_insights(context):

    forecast = context.get("forecast", [])
    churn = context.get("churn", {}).get("feature_importance", {})
    anomalies = context.get("anomalies", [])

    if len(forecast) >= 2:
        growth = forecast[-1] - forecast[0]
        trend = "increasing 📈" if growth > 0 else "decreasing 📉"
    else:
        growth = 0
        trend = "stable"

    top_factor = max(churn, key=churn.get) if churn else "unknown"

    anomaly_msg = (
        f"{len(anomalies)} anomalies detected"
        if anomalies else "No anomalies detected"
    )

    return f"""
📊 Insights:
- Revenue trend is {trend}
- Expected growth: ₹{int(growth)}

⚠️ Risks:
- Churn driven mainly by: {top_factor}
- {anomaly_msg}

🚀 Recommendations:
- Focus on improving {top_factor}
- Optimize marketing strategies
- Monitor anomalies regularly
""".strip()


# -----------------------------
# 🤖 GROQ AI (FINAL FIXED)
# -----------------------------
def generate_ai_response(prompt):

    try:
        from groq import Groq
        import os

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            return "⚠️ AI not configured. Showing basic insights."

        client = Groq(api_key=api_key)

        # 🔥 TRY MODELS (fallback system)
        models = [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768"
        ]

        for model in models:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except:
                continue  # try next model

        # 🔥 IF ALL FAIL → fallback
        return "⚠️ AI temporarily unavailable. Showing business insights instead."

    except Exception:
        return "⚠️ AI error. Using rule-based insights."
