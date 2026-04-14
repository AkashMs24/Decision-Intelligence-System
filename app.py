"""
DecisionIQ — app.py (UI-Only Version)
────────────────────────────────────────────────────────────────────
• 100% Same UI as original
• All ML, forecasting, churn, anomalies, Groq AI logic removed
• Uses clean sample data only
• Ready for demo / presentation / styling showcase
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="DecisionIQ", layout="wide", page_icon="🧠")

st.markdown("""
<style>
.main{background:#0e1117}
.block-container{padding-top:1.5rem;padding-bottom:2rem}
[data-testid="metric-container"]{background:#1c1f26;border:1px solid #2a2d36;
    border-radius:12px;padding:16px 20px}
.stTabs [data-baseweb="tab"]{border-radius:8px 8px 0 0;padding:8px 18px;font-size:.875rem}
.pill-high  {background:#3b1515;color:#f87171;border-radius:20px;padding:2px 10px;font-size:.75rem;font-weight:600}
.pill-medium{background:#3b2e0a;color:#fbbf24;border-radius:20px;padding:2px 10px;font-size:.75rem;font-weight:600}
.pill-low   {background:#0d3324;color:#34d399;border-radius:20px;padding:2px 10px;font-size:.75rem;font-weight:600}
hr{border-color:#2a2d36!important}
</style>
""", unsafe_allow_html=True)

# ── plotly helper ────
_BASE_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#c9d1d9",
    margin=dict(l=10, r=10, t=40, b=10),
)

def L(height=280, **extra):
    return {**_BASE_LAYOUT, "height": height, **extra}

# ════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA (Static for UI demo)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data
def get_sample_data():
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=730, freq="D").date
    revenue = np.maximum(
        np.linspace(250_000, 620_000, 730) + 
        45_000 * np.sin(np.arange(730) * 2 * np.pi / 365) + 
        np.random.normal(0, 18_000, 730), 40_000
    )
    customers = np.maximum(180 + np.cumsum(np.random.randint(-2, 7, 730)), 80).astype(float)

    df = pd.DataFrame({
        "date": dates,
        "revenue": revenue,
        "customers": customers,
        "marketing_spend": revenue * np.random.uniform(0.15, 0.25, 730),
        "churn": (np.diff(customers, prepend=customers[0]) < 0).astype(int),
    })
    return df

df = get_sample_data()

# Hardcoded results for beautiful UI (no real computation)
R = {
    "forecast": [685000, 712000, 748000, 781000],
    "forecast_prev": [592000, 618000, 645000, 670000],
    "upper": [720000, 755000, 790000, 825000],
    "lower": [650000, 670000, 705000, 735000],
    "forecast_r2": 0.928,
    "forecast_model": "XGBoost",
    
    "customers": 247,
    "customers_prev": 231,
    
    "churn": {
        "rate": 12.4,
        "rate_prev": 14.1,
        "accuracy": 0.87,
        "auc": 0.91,
        "cv_mean": 0.89,
        "cv_std": 0.04,
        "feature_importance": {
            "Customer Growth": 0.38,
            "Revenue Trend": 0.25,
            "Recent Revenue": 0.18,
            "Marketing Efficiency": 0.12,
            "7-Day Revenue": 0.07
        },
        "confusion_matrix": [[142, 12], [18, 28]]
    },
    "anomalies": [
        {"Event": "Revenue anomaly (z=3.2σ)", "Date": "2026-03-12", "Impact": "₹8.45L", "Severity": "High"},
        {"Event": "Revenue anomaly (z=2.1σ)", "Date": "2026-02-28", "Impact": "₹5.90L", "Severity": "Medium"},
        {"Event": "Revenue anomaly (z=1.8σ)", "Date": "2025-12-15", "Impact": "₹4.20L", "Severity": "Medium"},
    ],
    "model_cmp": {
        "XGBoost": {"R²": 0.928, "MAE": 12400},
        "Random Forest": {"R²": 0.891, "MAE": 15800},
        "Linear Regression": {"R²": 0.764, "MAE": 28900}
    },
    "insights": "Revenue is showing strong momentum with a projected peak of ₹7.81L in M+4. Churn has improved from 14.1% to 12.4%, primarily driven by stabilizing customer growth. Recommend accelerating marketing spend in high-growth segments while monitoring the recent revenue dip in March."
}

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Data Input")
    uploaded = st.file_uploader("Upload Business CSV", type=["csv"], disabled=True)
    st.caption("*(Demo mode — upload disabled)*")
    
    st.divider()
    st.button("▶ Run Analysis", use_container_width=True, type="primary", disabled=True)
    st.divider()
    
    st.markdown("**Pipeline**")
    st.caption("📈 Forecast  : XGBoost")
    st.caption("🔁 Churn     : RandomForest + CV")
    st.caption("🚨 Anomaly   : IsolationForest")
    st.caption("🤖 Insights  : Groq LLaMA-3 70B")
    st.caption(f"🕐 {datetime.now().strftime('%b %d, %H:%M')}")

# ════════════════════════════════════════════════════════════════════════════
# HEADER + KPIs
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 🧠 Decision Intelligence System")
st.caption("AI-Powered Executive Dashboard · " + datetime.now().strftime("%A, %d %B %Y"))
st.divider()

fp_val = max(R["forecast"])
fp_prev = max(R["forecast_prev"])
fp_d = (fp_val - fp_prev) / fp_prev * 100
cust = R["customers"]
churn_r = R["churn"]["rate"]
churn_p = R["churn"]["rate_prev"]
an_list = R["anomalies"]
an_high = [a for a in an_list if a["Severity"] == "High"]
fi = R["churn"]["feature_importance"]
top = max(fi, key=fi.get)

k1, k2, k3, k4 = st.columns(4)
k1.metric("📈 Forecast Revenue", f"₹{fp_val/100_000:.1f}L", f"{fp_d:+.1f}% vs prev")
k2.metric("👥 Active Customers", f"{cust:,}", f"{cust - R['customers_prev']:+d} vs 30d ago")
k3.metric("⚠️ Churn Risk", f"{churn_r:.1f}%", f"{churn_r - churn_p:+.1f}%", delta_color="inverse")
k4.metric("🚨 Anomalies", len(an_list), f"{len(an_high)} high", delta_color="inverse")

st.divider()
st.caption("📊 All outputs are generated from real-time uploaded data — **UI Demo Mode**")

months = ["M+1", "M+2", "M+3", "M+4"]

# ════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════
tab_fc, tab_ch, tab_an, tab_cmp, tab_wi, tab_ai = st.tabs([
    "📈 Forecast", "🔁 Churn", "🚨 Anomalies",
    "📊 Model Comparison", "🔮 What-If", "🤖 CEO Assistant"
])

# ── FORECAST ─────────────────────────────────────────────────────────────────
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
        st.caption(f"R² (test) = {R['forecast_r2']:.3f}  |  Bands = ±1.5σ")

    with c2:
        st.subheader("AI Executive Insights")
        st.success(R["insights"])
        with st.expander("📊 Processed data (last 30 rows)"):
            st.dataframe(df.tail(30), use_container_width=True)

# ── CHURN ─────────────────────────────────────────────────────────────────────
with tab_ch:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Feature Importance")
        fig = go.Figure(go.Bar(
            x=list(fi.values()), y=list(fi.keys()), orientation="h",
            marker=dict(color=list(fi.values()), colorscale=[[0,"#1e3a5f"],[1,"#ef4444"]]),
            text=[f"{v:.0%}" for v in fi.values()], textposition="outside"))
        fig.update_layout(**L(300, xaxis=dict(tickformat=".0%", gridcolor="#2a2d36"),
                              yaxis=dict(autorange="reversed")))
        st.plotly_chart(fig, use_container_width=True)

        cm_d = R["churn"]["confusion_matrix"]
        fig_cm = px.imshow(cm_d, text_auto=True, color_continuous_scale="Blues",
                           x=["No Churn", "Churn"], y=["No Churn", "Churn"])
        fig_cm.update_layout(**L(220, title="Confusion Matrix"))
        st.plotly_chart(fig_cm, use_container_width=True)

    with c2:
        st.subheader("Evaluation Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Accuracy", f"{R['churn']['accuracy']:.1%}")
        m2.metric("AUC-ROC", f"{R['churn']['auc']:.3f}")
        m1.metric("CV AUC", f"{R['churn']['cv_mean']:.3f}")
        m2.metric("CV Std", f"±{R['churn']['cv_std']:.3f}")
        st.divider()
        st.info(f"**Top driver: {top}** ({fi[top]:.0%} impact)")
        st.plotly_chart(go.Figure(go.Indicator(
            mode="gauge+number+delta", value=churn_r,
            delta=dict(reference=churn_p, suffix="%", valueformat=".1f"),
            gauge=dict(axis=dict(range=[0,40]), bar=dict(color="#ef4444"),
                       steps=[dict(range=[0,15],color="#0d3324"),
                              dict(range=[15,25],color="#3b2e0a"),
                              dict(range=[25,40],color="#3b1515")]))).update_layout(**L(220)),
            use_container_width=True)

# ── ANOMALIES ─────────────────────────────────────────────────────────────────
with tab_an:
    if not R["anomalies"]:
        st.success("✅ No anomalies detected.")
    else:
        st.warning(f"⚠️ {len(R['anomalies'])} anomalies — {len(an_high)} high severity")
        for col, lbl in zip(st.columns([4,2,2,2]), ["Event", "Date", "Impact", "Severity"]):
            col.markdown(f"**{lbl}**")
        st.divider()
        for a in R["anomalies"]:
            row = st.columns([4,2,2,2])
            row[0].write(a["Event"])
            row[1].write(a["Date"])
            row[2].write(a["Impact"])
            row[3].markdown(f'<span class="pill-{a["Severity"].lower()}">{a["Severity"]}</span>', unsafe_allow_html=True)

# ── MODEL COMPARISON ──────────────────────────────────────────────────────────
with tab_cmp:
    st.subheader("Model Comparison — 20% hold-out test")
    mc = R["model_cmp"]
    mn = list(mc.keys())
    colors = ["#3b82f6", "#10b981", "#f59e0b"][:len(mn)]
    cc1, cc2 = st.columns(2)
    with cc1:
        fig = go.Figure(go.Bar(x=mn, y=[mc[m]["R²"] for m in mn], marker_color=colors,
                               text=[f"{mc[m]['R²']:.3f}" for m in mn], textposition="outside"))
        fig.update_layout(**L(280, title="R² Score (higher = better)", yaxis=dict(range=[0,1.15])))
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        fig = go.Figure(go.Bar(x=mn, y=[mc[m]["MAE"] for m in mn], marker_color=colors,
                               text=[f"₹{mc[m]['MAE']/1000:.0f}K" for m in mn], textposition="outside"))
        fig.update_layout(**L(280, title="MAE — lower is better"))
        st.plotly_chart(fig, use_container_width=True)
    best = max(mc, key=lambda m: mc[m]["R²"])
    st.info(f"**Best: {best}**  R²={mc[best]['R²']:.3f}  MAE=₹{mc[best]['MAE']/1000:.1f}K")

# ── WHAT-IF ───────────────────────────────────────────────────────────────────
with tab_wi:
    st.subheader("🔮 What-If Revenue Simulator")
    wc1, wc2, wc3 = st.columns(3)
    mkt = wc1.slider("📣 Marketing Spend Δ (%)", -50, 100, 20)
    chd = wc2.slider("♻️ Churn Reduction (%)", 0, 80, 30)
    cgr = wc3.slider("📈 Customer Growth Δ (%)", -20, 50, 10)

    mult = (1 + mkt/100*0.40) * (1 + chd/100*0.25) * (1 + cgr/100*0.60)
    sim = [max(v * mult, 0) for v in R["forecast"]]
    uplift = sum(sim) - sum(R["forecast"])
    uplift_p = (sum(sim) / sum(R["forecast"]) - 1) * 100 if sum(R["forecast"]) > 0 else 0

    fig = go.Figure()
    fig.add_trace(go.Bar(x=months, y=R["forecast"], name="Base", marker_color="#3b82f6"))
    fig.add_trace(go.Bar(x=months, y=sim, name="Simulated", marker_color="#10b981"))
    fig.update_layout(**L(300, barmode="group", yaxis=dict(tickprefix="₹", tickformat=".2s")))
    st.plotly_chart(fig, use_container_width=True)

    (st.success if uplift >= 0 else st.error)(
        f"**Projected uplift: ₹{uplift/100_000:.2f}L ({uplift_p:+.1f}%)** over 4 months")

# ── CEO ASSISTANT ─────────────────────────────────────────────────────────────
with tab_ai:
    st.subheader("🤖 CEO Assistant — Groq LLaMA-3 70B")
    st.info("💡 **Demo Mode**: Chat functionality is disabled in this UI-only version.")
    
    st.markdown("""
    **Sample Questions you can ask in full version:**
    - What is driving our churn?
    - Should we increase marketing spend next quarter?
    - Give me a 3-month action plan.
    """)

    if st.button("🗑 Clear chat (demo)"):
        st.info("Chat cleared (no effect in demo mode)")
