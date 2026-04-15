# pipeline.py
"""
DecisionIQ — central pipeline.
Loads data → preprocesses → runs models → generates AI insights.
"""
import pandas as pd

from utils.data_loader import load_data
from utils.data_preprocessor import transform_user_data
from models.forecasting import forecast_revenue
from models.churn import churn_analysis
from models.anomaly import detect_anomalies
from utils.llm_engine import ai


class DecisionIQPipeline:
    """
    Orchestrates data loading, transformation, and model execution.

    Parameters
    ----------
    uploaded_file : file-like object | None
        Streamlit UploadedFile. When None, loads the default sample CSV.
    """

    def __init__(self, uploaded_file=None):
        if uploaded_file is not None:
            self.filename = uploaded_file.name
            raw = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
        else:
            self.filename = "sample_data.csv"
            raw = load_data("data/sample_data.csv")

        self.df = transform_user_data(raw, self.filename)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> dict:
        """Execute all models and return a results dict for the dashboard."""
        fc = forecast_revenue(self.df)
        ch = churn_analysis(self.df)
        an = detect_anomalies(self.df)

        is_biz = self.df.attrs.get("is_business_like", False)
        top_driver = (
            max(ch["feature_importance"], key=ch["feature_importance"].get)
            if ch["feature_importance"]
            else "N/A"
        )

        # ── AI narrative insights ──────────────────────────────────────────────
        system_prompt = (
            "You are a senior business analyst and a friendly executive advisor. "
            "Be concise, honest, and data-driven. "
            "If the dataset is not business-suitable, say so clearly and briefly."
        )
        user_prompt = (
            f"Dataset: {self.filename} | Rows: {len(self.df)} | "
            f"Business Suitable: {'Yes' if is_biz else 'No'}\n"
            f"4-month Revenue Forecast: {[f'₹{v / 100_000:.1f}L' for v in fc['forecast']]}\n"
            f"Churn Rate: {ch['rate']:.1f}% | Top Driver: {top_driver}\n"
            f"Anomalies Detected: {len(an)} "
            f"(High: {sum(1 for a in an if a.get('Severity') == 'High')})\n"
            f"Forecast Model R²: {fc['r2']:.3f}\n\n"
            "Give 3-4 sharp bullet-point insights with actionable recommendations."
        )
        insights = ai(system_prompt, user_prompt)

        # ── Customer trend (vs 30 days ago) ───────────────────────────────────
        cust_now  = int(self.df["customers"].iloc[-1])
        cust_prev = (
            int(self.df["customers"].iloc[-31])
            if len(self.df) > 31
            else int(self.df["customers"].iloc[0])
        )

        return {
            # Forecast
            "forecast":       fc["forecast"],
            "forecast_prev":  fc["forecast_prev"],
            "upper":          fc["upper"],
            "lower":          fc["lower"],
            "forecast_r2":    fc["r2"],
            "forecast_model": fc["model"],
            # Churn
            "churn": ch,
            # Anomalies
            "anomalies": an,
            # AI
            "insights": insights,
            # Customers
            "customers":      cust_now,
            "customers_prev": cust_prev,
        }
