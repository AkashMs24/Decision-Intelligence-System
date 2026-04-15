"""
DecisionIQ — AI-Powered Executive Dashboard
============================================
All KPIs, charts, and labels adapt automatically to the uploaded dataset.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

from pipeline import DecisionIQPipeline
from utils.llm_engine import ai

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="DecisionIQ", layout="wide", page_icon="🧠")

st.markdown(
    """
    <style>
    .main { background: #0e1117; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    [data-testid="metric-container"] {
        background: #1c1f26;
        border: 1px solid #2a2d36;
        border-radius: 12px;
        padding: 16px 20px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        font-size: .875rem;
    }
    .pill-high   { background:#3b1515; color:#f87171;  border-radius:20px; padding:2px 10px; font-size:.75rem; font-weight:600; }
    .pill-medium { background:#3b2e0a; color:#fbbf24;  border-radius:20px; padding:2px 10px; font-size:.75rem; font-weight:600; }
    .pill-low    { background:#0d3324; color:#34d399;  border-radius:20px; padding:2px 10px; font-size:.75rem; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Shared layout defaults ─────────────────────────────────────────────────────
_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#c9d1d9",
    margin=dict(l=10, r=10, t=40, b=10),
)


def _layout(height: int = 280, **extra) -> dict:
    return {**_BASE, "height": height, **extra}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Data Input")
    uploaded = st.file_uploader("Upload Business CSV", type=["csv"])
    st.divider()

    run_btn = st.button("▶ Run Analysis", use_container_width=True, type="primary")
    st.divider()

    st.markdown("**Pipeline**")
    st.caption("📈 Forecast  : XGBoost / GradientBoosting")
    st.caption("🔁 Churn     : RandomForest (balanced, CV-5)")
    st.caption("🚨 Anomaly   : Z-score (>2.5 σ)")
    st.caption("🤖 Insights  : Groq LLaMA-3.1 8B")
    st.caption(f"🕐 {datetime.now().strftime('%b %d, %H:%M')}")

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & RUN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
current_file = uploaded.name if uploaded else "sample_data.csv"

# Rebuild pipeline whenever the file changes or the user hits Run
if (
    run_btn
    or "results" not in st.session_state
    or st.session_state.get("last_filename") != current_file
):
    with st.spinner("Running DecisionIQ analysis …"):
        pipeline = DecisionIQPipeline(uploaded)
        st.session_state.results        = pipeline.run()
        st.session_state.last_filename  = current_file
        st.session_state.pipeline       = pipeline

pipeline      = st.session_state.pipeline
R             = st.session_state.results
df            = pipeline.df
is_biz        = df.attrs.get("is_business_like", False)
dataset_type  = df.attrs.get("dataset_type", "generic")
filename      = pipeline.filename

# ── Dynamic labels based on dataset ───────────────────────────────────────────
rev_label   = "Sales / Revenue"     if is_biz else "Synthetic Revenue"
cust_label  = "Active Customers"    if is_biz else "Records"
months      = ["M+1", "M+2", "M+3", "M+4"]

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🧠 Decision Intelligence System")
st.caption("AI-Powered Executive Dashboard · " + datetime.now().strftime("%A, %d %B %Y"))
st.divider()

if not is_biz:
    st.error(
        f"⚠️ **Warning**: '{filename}' is not a standard business/sales dataset. "
        "Revenue, Churn & Forecast values shown are **synthetic** proxies."
    )

# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
fp_val  = max(R["forecast"])
fp_prev = max(R["forecast_prev"])
fp_d    = (fp_val - fp_prev) / fp_prev * 100 if fp_prev else 0

high_anomalies = len([a for a in R["anomalies"] if a.get("Severity") == "High"])
churn_delta    = R["churn"]["rate"] - R["churn"]["rate_prev"]

k1, k2, k3, k4 = st.columns(4)
k1.metric(f"📈 Peak Forecast {rev_label}",  f"₹{fp_val / 100_000:.1f}L",  f"{fp_d:+.1f}% vs prev")
k2.metric(f"👥 {cust_label}",               f"{R['customers']:,}",          f"{R['customers'] - R['customers_prev']:+d} vs 30d ago")
k3.metric("⚠️ Churn Risk",                  f"{R['churn']['rate']:.1f}%",   f"{churn_delta:+.1f}%", delta_color="inverse")
k4.metric("🚨 Anomalies Detected",          len(R["anomalies"]),             f"{high_anomalies} high", delta_color="inverse")

st.divider()
st.info(
    f"**Dataset:** {filename} | "
    f"**Rows:** {len(df)} | "
    f"**Type:** {dataset_type.replace('_', ' ').title()} | "
    f"**Business Suitable:** {'✅ Yes' if is_biz else '❌ No'}"
)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_fc, tab_ch, tab_an, tab_raw, tab_cmp, tab_wi, tab_ai = st.tabs([
    "📈 Forecast",
    "🔁 Churn",
    "🚨 Anomalies",
    "🗂 Raw Data",
    "📊 Model Comparison",
    "🔮 What-If",
    "🤖 CEO Assistant",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — FORECAST
# ─────────────────────────────────────────────────────────────────────────────
with tab_fc:
    c1, c2 = st.columns([3, 2])

    with c1:
        st.subheader(f"Revenue Forecast — {R['forecast_model']}")
        fig = go.Figure()

        # Confidence band
        fig.add_trace(go.Scatter(x=months, y=R["upper"], mode="lines",
                                 line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=months, y=R["lower"], mode="lines",
                                 fill="tonexty",
                                 fillcolor="rgba(59,130,246,0.12)",
                                 line=dict(width=0), name="95% CI"))

        # Previous period
        fig.add_trace(go.Scatter(x=months, y=R["forecast_prev"], mode="lines+markers",
                                 name="Previous Period",
                                 line=dict(color="#4a5568", dash="dot", width=1.5)))

        # Current forecast
        fig.add_trace(go.Scatter(x=months, y=R["forecast"], mode="lines+markers",
                                 name="Forecast",
                                 line=dict(color="#3b82f6", width=2.5)))

        fig.update_layout(
            **_layout(300,
                      yaxis=dict(tickprefix="₹", tickformat=".2s"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"R² = {R['forecast_r2']:.3f} | Bands = ±1.5σ residual")

        # Historic revenue line
        st.markdown("#### 📉 Historic Revenue Trend")
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Scatter(
            x=[str(d) for d in df["date"]],
            y=df["revenue"],
            mode="lines",
            name="Revenue",
            line=dict(color="#10b981", width=1.5),
        ))
        hist_fig.update_layout(
            **_layout(220, yaxis=dict(tickprefix="₹", tickformat=".2s"))
        )
        st.plotly_chart(hist_fig, use_container_width=True)

    with c2:
        st.subheader("🧠 AI Executive Insights")
        st.success(R["insights"])

        st.markdown("#### 📊 Processed Data (last 30 rows)")
        st.dataframe(df.tail(30), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CHURN
# ─────────────────────────────────────────────────────────────────────────────
with tab_ch:
    ch = R["churn"]
    c1, c2 = st.columns([2, 3])

    with c1:
        st.subheader("Churn Summary")
        st.metric("Current Churn Rate",  f"{ch['rate']:.1f}%",
                  f"{ch['rate'] - ch['rate_prev']:+.1f}% vs prev", delta_color="inverse")
        st.metric("Model Accuracy",  f"{ch['accuracy']*100:.1f}%")
        st.metric("AUC Score",       f"{ch['auc']:.3f}")
        st.metric("CV Mean Accuracy",f"{ch['cv_mean']*100:.1f}% ± {ch['cv_std']*100:.1f}%")
        st.caption(f"Model: {ch['model']}")

        # Confusion matrix
        cm = ch["confusion_matrix"]
        st.markdown("**Confusion Matrix**")
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No Churn", "Actual: Churn"],
            columns=["Pred: No Churn", "Pred: Churn"],
        )
        st.dataframe(cm_df, use_container_width=True)

    with c2:
        st.subheader("Feature Importance")
        fi = ch["feature_importance"]
        fi_fig = go.Figure(go.Bar(
            x=list(fi.values()),
            y=list(fi.keys()),
            orientation="h",
            marker_color=["#3b82f6", "#10b981", "#f59e0b"],
        ))
        fi_fig.update_layout(**_layout(260, xaxis_title="Importance"))
        st.plotly_chart(fi_fig, use_container_width=True)

        # Customer trend over time
        st.subheader("Customer Count Over Time")
        cust_fig = go.Figure(go.Scatter(
            x=[str(d) for d in df["date"]],
            y=df["customers"],
            mode="lines",
            line=dict(color="#a78bfa", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(167,139,250,0.1)",
        ))
        cust_fig.update_layout(**_layout(220))
        st.plotly_chart(cust_fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — ANOMALIES
# ─────────────────────────────────────────────────────────────────────────────
with tab_an:
    st.subheader("🚨 Revenue Anomalies (Z-score > 2.5σ)")

    if not R["anomalies"]:
        st.success("✅ No significant anomalies detected in the current dataset.")
    else:
        for a in R["anomalies"]:
            sev   = a.get("Severity", "Low")
            pill  = f'<span class="pill-{sev.lower()}">{sev}</span>'
            st.markdown(
                f"**{a['Event']}** — {a['Date']} — Impact: `{a['Impact']}` {pill}",
                unsafe_allow_html=True,
            )
        st.divider()
        st.dataframe(pd.DataFrame(R["anomalies"]), use_container_width=True)

    # Z-score scatter
    st.subheader("Z-Score Distribution")
    rev   = df["revenue"]
    mean  = rev.mean()
    std   = rev.std() or 1
    z_scores = (rev - mean) / std

    z_fig = go.Figure()
    z_fig.add_trace(go.Scatter(
        x=[str(d) for d in df["date"]],
        y=z_scores,
        mode="lines+markers",
        marker=dict(
            color=["#f87171" if abs(z) > 2.5 else "#3b82f6" for z in z_scores],
            size=5,
        ),
        line=dict(color="#3b82f6", width=1),
        name="Z-score",
    ))
    z_fig.add_hline(y=2.5,  line_dash="dot", line_color="#f87171",  annotation_text="+2.5σ")
    z_fig.add_hline(y=-2.5, line_dash="dot", line_color="#f87171",  annotation_text="-2.5σ")
    z_fig.update_layout(**_layout(260))
    st.plotly_chart(z_fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
with tab_raw:
    st.subheader(f"📋 Full Dataset — {filename}")
    col_a, col_b = st.columns(2)
    col_a.metric("Total Rows", len(df))
    col_b.metric("Columns",    len(df.columns))

    st.dataframe(df, use_container_width=True)

    # Revenue distribution
    st.subheader("Revenue Distribution")
    hist_fig = go.Figure(go.Histogram(
        x=df["revenue"],
        nbinsx=30,
        marker_color="#3b82f6",
        opacity=0.8,
    ))
    hist_fig.update_layout(**_layout(240, xaxis=dict(tickprefix="₹", tickformat=".2s")))
    st.plotly_chart(hist_fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
with tab_cmp:
    st.subheader("📊 Model Performance Comparison")

    metrics = {
        "Metric":    ["R² Score", "Churn Accuracy", "AUC", "CV Mean Acc"],
        "Value":     [
            f"{R['forecast_r2']:.3f}",
            f"{R['churn']['accuracy']*100:.1f}%",
            f"{R['churn']['auc']:.3f}",
            f"{R['churn']['cv_mean']*100:.1f}%",
        ],
        "Model":     [
            R["forecast_model"],
            R["churn"]["model"],
            R["churn"]["model"],
            R["churn"]["model"],
        ],
    }
    st.dataframe(pd.DataFrame(metrics), use_container_width=True)

    # Bar chart of forecast values
    st.subheader("Forecast vs Previous Period")
    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(x=months, y=R["forecast_prev"], name="Previous",
                             marker_color="#4a5568"))
    bar_fig.add_trace(go.Bar(x=months, y=R["forecast"],      name="Current",
                             marker_color="#3b82f6"))
    bar_fig.update_layout(
        **_layout(300, barmode="group",
                  yaxis=dict(tickprefix="₹", tickformat=".2s"))
    )
    st.plotly_chart(bar_fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 6 — WHAT-IF SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
with tab_wi:
    st.subheader("🔮 What-If Revenue Simulator")
    st.caption("Adjust sliders to see how changes in marketing spend affect the revenue forecast.")

    mkt_boost   = st.slider("Marketing Spend Multiplier", 0.5, 3.0, 1.0, 0.1)
    price_adj   = st.slider("Average Price Adjustment (%)", -30, 50, 0, 5)

    adj_forecast = [
        p * mkt_boost * (1 + price_adj / 100)
        for p in R["forecast"]
    ]

    wi_fig = go.Figure()
    wi_fig.add_trace(go.Scatter(x=months, y=R["forecast"], mode="lines+markers",
                                name="Baseline", line=dict(color="#3b82f6", width=2)))
    wi_fig.add_trace(go.Scatter(x=months, y=adj_forecast, mode="lines+markers",
                                name="Adjusted", line=dict(color="#10b981", width=2, dash="dash")))
    wi_fig.update_layout(
        **_layout(300, yaxis=dict(tickprefix="₹", tickformat=".2s"))
    )
    st.plotly_chart(wi_fig, use_container_width=True)

    uplift = sum(adj_forecast) - sum(R["forecast"])
    if uplift >= 0:
        st.success(f"📈 Projected uplift over 4 months: **₹{uplift/100_000:.2f}L**")
    else:
        st.warning(f"📉 Projected reduction over 4 months: **₹{abs(uplift)/100_000:.2f}L**")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 7 — CEO ASSISTANT (CHATBOT)
# ─────────────────────────────────────────────────────────────────────────────
with tab_ai:
    st.subheader("🤖 CEO Assistant — Groq LLaMA-3.1 8B")

    # Dynamic system prompt — updates whenever dataset changes
    top_driver = (
        max(R["churn"]["feature_importance"], key=R["churn"]["feature_importance"].get)
        if R["churn"]["feature_importance"] else "N/A"
    )

    SYS_PROMPT = f"""You are a warm, friendly, and highly capable personal AI assistant to a busy CEO.
You have access to the company's DecisionIQ analytics platform.

Current dataset: {filename}
Total records: {len(df)}
Dataset type: {dataset_type.replace('_', ' ').title()}
Business dataset: {'Yes' if is_biz else 'No (data is synthetic)'}

Live Metrics:
- 4-month Revenue Forecast: {[f'₹{v/100_000:.1f}L' for v in R.get('forecast', [])]}
- Forecast Model: {R['forecast_model']} | R²: {R['forecast_r2']:.3f}
- Churn Rate: {R['churn']['rate']:.1f}% | Top Driver: {top_driver}
- Active Customers: {R['customers']:,}
- Anomalies Detected: {len(R['anomalies'])} (High: {high_anomalies})

Personality rules:
1. Always be warm, helpful, and friendly.
2. If the user says hi/hello/hey or any greeting → respond casually and ask how you can help.
3. For business questions → reference the live metrics above and give sharp, actionable advice.
4. Never refuse to answer. If uncertain, say so honestly and still try to help.
5. Keep responses concise but thorough — no one-word answers.
6. Use bullet points for lists, plain text for conversational replies.
"""

    # Initialise chat history
    if "chat" not in st.session_state or st.session_state.get("chat_file") != current_file:
        st.session_state.chat      = [
            {"role": "assistant",
             "content": f"👋 Hey there! I'm your DecisionIQ assistant. "
                        f"I've just analysed **{filename}** — {len(df)} rows of data. "
                        f"Ask me anything about your business metrics!"}
        ]
        st.session_state.chat_file = current_file

    # Render conversation
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    if prompt := st.chat_input("Ask about your data, forecasts, churn, or anything else …"):
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.spinner("Thinking …"):
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat[:-1]
            ]
            reply = ai(SYS_PROMPT, prompt, history, max_tokens=1_024)

        st.session_state.chat.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

    if st.button("🗑 Clear Chat"):
        st.session_state.chat      = []
        st.session_state.chat_file = None
        st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("🧠 DecisionIQ · All outputs derived from real uploaded or sample data · No static placeholders")
