from utils.data_loader import load_data
from models.forecasting import forecast_revenue
from models.churn import churn_analysis
from models.anomaly import detect_anomalies
from utils.llm_engine import generate_insights


class DecisionIQPipeline:

    def __init__(self, df=None):
        """
        Initialize pipeline with optional external data
        """
        if df is not None:
            self.df = df
        else:
            self.df = load_data()

    # -------------------------
    # 🧹 CLEAN NUMPY TYPES
    # -------------------------
    def clean_data(self, data):
        import numpy as np

        if isinstance(data, dict):
            return {k: self.clean_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.clean_data(i) for i in data]
        elif isinstance(data, np.generic):
            return float(data)
        else:
            return data

    # -------------------------
    # 🚀 MAIN PIPELINE
    # -------------------------
    def run(self):

        # 1️⃣ Forecasting
        forecast = forecast_revenue(self.df)

        # 2️⃣ Churn Analysis
        churn = churn_analysis(self.df)

        # 3️⃣ Anomaly Detection
        anomalies = detect_anomalies(self.df)

        # 4️⃣ Clean outputs
        context = self.clean_data({
            "forecast": forecast,
            "churn": churn,
            "anomalies": anomalies
        })

        # 5️⃣ Generate Insights (rule-based engine)
        insights = generate_insights(context)

        # 6️⃣ Final structured output
        return {
            "forecast": context["forecast"],
            "churn": context["churn"],
            "anomalies": context["anomalies"],
            "insights": insights
        }