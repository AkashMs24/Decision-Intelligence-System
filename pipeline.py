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

        top = max(ch["feature_importance"], key=ch["feature_importance"].get)

        dataset_info = (f"Dataset: {self.filename}\n"
                        f"Rows: {len(self.df)}\n"
                        f"Original Columns: {self.df.attrs.get('original_columns', [])}")

        insights = ai(
            system_prompt="You are a senior business analyst. Be honest about data quality.",
            user_prompt=f"{dataset_info}\n\n"
                        f"Forecast (4 mo): {[f'₹{v/100000:.1f}L' for v in fc['forecast']]}\n"
                        f"Current Churn: {ch['rate']:.1f}% | Top driver: {top}\n"
                        f"Anomalies: {len(an)}\n"
                        f"Last revenue: ₹{self.df['revenue'].iloc[-1]/100000:.1f}L"
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
            "dataset_info": dataset_info
        }
