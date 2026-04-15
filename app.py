"""
DecisionIQ — Final Version
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

from pipeline import DecisionIQPipeline
from utils.llm_engine import ai

st.set_page_config(page_title="DecisionIQ", layout="wide", page_icon="🧠")

st.markdown("""
<style>
.main{background:#0e1117}
.block-container{padding-top:1.5rem;padding-bottom:2rem}
[data-testid="metric-container"]{background:#1c1f26;border:1px solid #2a2d36;border-radius:12px;padding:16px 20px}
.stTabs [data-baseweb="tab"]{border-radius:8px 8px 0 0;padding:8px 18px;font-size:.875rem}
.pill-high  {background:#3b1515;color:#f87171;border-radius:20px;padding:2px 10px;font-size:.75rem;font-weight:600}
.pill-medium{background:#3b2e0a;color:#fbbf24;border-radius:20px;padding:2px 10px;font-size:.75rem;font-weight:600}
.pill-low   {background:#0d3324;color:#34d399;border-radius:20px;padding:2px 10px;font-size:.75rem;font-weight:600}
</style>
""", unsafe_allow_html=True)

_BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#c9d1d9",
    margin=dict(l=10, r=10, t=40, b=10),
)

def L(height=280, **extra):
    return {**_BASE_LAYOUT, "height": height, **extra}

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("### ⚙️ Data Input")
    uploaded = st.file_uploader("Upload Business CSV", type=["csv"])
    st.divider()
    run_btn = st.button("▶ Run Analysis", use_container_width=True, type="primary")
    st.divider()
    st.markdown("**Pipeline**")
    st.caption("📈 Forecast  : XGBoost")
    st.caption("🔁 Churn     : RandomForest + CV")
    st.caption("🚨 Anomaly   : IsolationForest")
    st.caption("🤖 Insights  : Groq LLaMA-3 70B")
    st.caption(f"🕐 {datetime.now().strftime('%b %d, %H:%M')}")

# ====================== LOAD & RUN PIPELINE (with cache clear) ======================
if uploaded:
    with st.spinner("Processing uploaded data..."):
        pipeline = DecisionIQPipeline(uploaded)
else:
    pipeline = DecisionIQPipeline()

if run_btn or "results" not in st.session_state or st.session_state.get("last_file") != (uploaded.name if uploaded else "sample"):
    with st.spinner("Running analysis..."):
        st.session_state.results = pipeline.run()
        st.session_state.last_file = uploaded.name if uploaded else "sample"

R = st.session_state.results
is_business_data = pipeline.df.attrs.get('is_business_like', False)

# ====================== HEADER + KPIs ======================
st.markdown("## 🧠 Decision Intelligence System")
st.caption("AI-Powered Executive Dashboard · " + datetime.now().strftime("%A, %d %B %Y"))
st.divider()

if not is_business_data:
    st.error(f"⚠️ **Warning**: Dataset '{pipeline.filename}' is not a standard business/sales dataset. "
             "Revenue, Churn & Forecast values are synthetic.")

fp_val = max(R["forecast"])
fp_prev = max(R["forecast_prev"])
fp_d = (fp_val - fp_prev) / fp_prev * 100 if fp_prev else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("📈 Forecast Revenue", f"₹{fp_val/100_000:.1f}L", f"{fp_d:+.1f}% vs prev")
k2.metric("👥 Active Customers", f"{R['customers']:,}", f"{R['customers'] - R['customers_prev']:+d} vs 30d ago")
k3.metric("⚠️ Churn Risk", f"{R['churn']['rate']:.1f}%", f"{R['churn']['rate']-R['churn']['rate_prev']:+.1f}%", delta_color="inverse")
k4.metric("🚨 Anomalies", len(R["anomalies"]), f"{len([a for a in R['anomalies'] if a.get('Severity')=='High'])} high", delta_color="inverse")

st.divider()
st.info(f"**Dataset:** {pipeline.filename} | Rows: {len(pipeline.df)} | Business Suitable: {'✅ Yes' if is_business_data else '❌ No'}")

months = ["M+1","M+2","M+3","M+4"]

# ====================== TABS ======================
tab_fc, tab_ch, tab_an, tab_cmp, tab_wi, tab_ai = st.tabs([
    "📈 Forecast", "🔁 Churn", "🚨 Anomalies",
    "📊 Model Comparison", "🔮 What-If", "🤖 CEO Assistant"
])

# Forecast Tab
with tab_fc:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader(f"Revenue Forecast — {R['forecast_model']}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=months, y=R["upper"], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=months, y=R["lower"], mode="lines", fill="tonexty",
                                 fillcolor="rgba(59,130,246,0.12)", line=dict(width=0), name="95% CI"))
        fig.add_trace(go.Scatter(x=months, y=R["forecast_prev"], mode="lines+markers",
                                 name="Previous", line=dict(color="#4a5568", dash="dot", width=1.5)))
        fig.add_trace(go.Scatter(x=months, y=R["forecast"], mode="lines+markers",
                                 name="Forecast", line=dict(color="#3b82f6", width=2.5)))
        fig.update_layout(**L(300, yaxis=dict(tickprefix="₹", tickformat=".2s"),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02)))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"R² = {R['forecast_r2']:.3f} | Bands = ±1.5σ")
    with c2:
        st.subheader("AI Executive Insights")
        st.success(R["insights"])
        with st.expander("📊 Processed data (last 30 rows)"):
            st.dataframe(pipeline.df.tail(30), use_container_width=True)

# CEO Assistant Tab
with tab_ai:
    st.subheader("🤖 CEO Assistant — Groq LLaMA-3 70B")

    SYS_PROMPT = f"""You are an honest CEO-level AI business analyst.

Dataset: {pipeline.filename}
Rows: {len(pipeline.df)}
Suitable for Business Analysis: {'Yes' if is_business_data else 'No'}

Current Metrics:
- Revenue Forecast: {[f'₹{v/100000:.1f}L' for v in R.get('forecast', [])]}
- Churn Risk: {R['churn'].get('rate', 0):.1f}%
- Top Driver: {max(R['churn'].get('feature_importance', {}), key=R['churn'].get('feature_importance', {}).get, default='N/A')}

Be honest."""

    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "assistant", "content": "👋 Hello! I have the uploaded dataset. Ask me anything."}]

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask a business question …"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Calling Groq..."):
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat[:-1]]
            ans = ai(SYS_PROMPT, prompt, history)

        st.session_state.chat.append({"role": "assistant", "content": ans})
        with st.chat_message("assistant"):
            st.write(ans)

    if st.button("🗑 Clear chat"):
        st.session_state.chat = []
        st.rerun()

st.caption("All outputs generated from real uploaded or sample data — No static values")
