# pipeline.py
import pandas as pd
from utils.data_loader import load_data
from utils.data_preprocessor import transform_user_data
from models.forecasting import forecast_revenue
from models.churn import churn_analysis
from models.anomaly import detect_anomalies
from utils.llm_engine import ai

class DecisionIQPipeline:
    def __init__(self, uploaded_file=None):
        if uploaded_file is not None:
            self.filename = uploaded_file.name
            raw = pd.read_csv(uploaded_file, encoding="ISO-8859-1")
            self.df = transform_user_data(raw, self.filename)
        else:
            self.df = load_data()
            self.filename = "sample_data.csv"

    def run(self):
        fc = forecast_revenue(self.df)
        ch = churn_analysis(self.df)
        an = detect_anomalies(self.df)
        
        top = max(ch.get("feature_importance", {}), key=ch.get("feature_importance", {}).get, default="N/A")
        
        insights = ai(
            system_prompt="You are a senior business analyst. Be concise, honest about data quality, and highlight key opportunities/risks.",
            user_prompt=f"Dataset: {self.filename}\n"
                        f"Business Suitable: {'Yes' if self.df.attrs.get('is_business_like', False) else 'No'}\n"
                        f"Forecast (4 mo): {[f'₹{v/100000:.1f}L' for v in fc['forecast']]}\n"
                        f"Churn: {ch['rate']:.1f}% | Top driver: {top}\n"
                        f"Anomalies: {len(an)}"
        )
        
        return {
            "forecast": fc["forecast"],
            "forecast_prev": fc["forecast_prev"],
            "upper": fc["upper"],
            "lower": fc["lower"],
            "forecast_r2": fc["r2"],
            "forecast_model": fc["model"],
            "churn": ch,
            "anomalies": an,
            "insights": insights,
            "customers": int(self.df["customers"].iloc[-1]),
            "customers_prev": int(self.df["customers"].iloc[-30]) if len(self.df) > 30 else int(self.df["customers"].iloc[0]),
        }
