<div align="center">

# Decision Intelligence System

> Most dashboards tell you what happened.
> This one tells you what to do next.

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://decision-intelligence-system.streamlit.app/)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20RandomForest%20%7C%20IsolationForest-blue?style=flat-square)
![LLM](https://img.shields.io/badge/LLM-Groq%20LLaMA--3%2070B-purple?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)

</div>

## What this system does

Upload any business CSV. The system automatically:

- **Forecasts revenue** — XGBoost time-series model with confidence intervals
- **Predicts churn** — Random Forest + cross-validation with feature importance
- **Detects anomalies** — Isolation Forest flags unusual patterns with severity scoring
- **Generates executive insights** — Groq LLaMA-3 70B converts model outputs into plain business decisions
- **Simulates scenarios** — What-If analysis for marketing spend, churn reduction, customer growth

No hardcoded values. Every output is driven by the data you upload.

---

## Why this is different from a regular dashboard

| Regular dashboard | Decision Intelligence System |
|---|---|
| Shows what happened | Predicts what will happen |
| Static charts | Dynamic ML outputs |
| Analyst interprets | LLM generates recommendations |
| One dataset view | What-If scenario simulation |

---

## System architecture
CSV Upload
↓
Auto data preprocessing (column detection, standardization)
↓
ML Pipeline
├── Revenue Forecast     → XGBoost + confidence intervals (±1.5σ)
├── Churn Prediction     → Random Forest + AUC-ROC + CV Score
└── Anomaly Detection    → Isolation Forest (High / Medium / Low severity)
↓
LLM Layer (Groq LLaMA-3 70B)
↓
Executive insights + CEO Assistant chatbot
↓
Streamlit Dashboard

---

## Features

**ML models**
- Revenue forecasting with R² score and confidence bands
- Churn prediction with accuracy, AUC-ROC, CV score, and feature importance
- Anomaly detection with severity classification
- Model comparison dashboard — Linear Regression vs Random Forest vs XGBoost

**LLM layer**
- AI executive insights: converts model outputs into business-language decisions
- CEO Assistant chatbot: ask "Why is churn increasing?" and get answers using live data context

**Scenario simulation**
- Adjust marketing spend, churn rate, or customer growth
- Instantly see projected revenue impact

---

## Model performance

| Model | Task | Metric |
|---|---|---|
| XGBoost | Revenue forecasting | R² Score |
| Random Forest | Churn prediction | Accuracy + AUC-ROC + CV |
| Isolation Forest | Anomaly detection | Severity classification |

---

## Stack

`Python` `Streamlit` `XGBoost` `scikit-learn` `Pandas` `NumPy` `Plotly` `Groq API (LLaMA-3 70B)`

---

## Run locally

```bash
git clone https://github.com/AkashMs24/Decisioniq-ai-business-intelligence.git
cd Decisioniq-ai-business-intelligence
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Then run:

```bash
streamlit run app.py
```

---

## Use cases

- Startup founders who need data-driven decisions without a data team
- Marketing teams optimising spend and retention
- Business analysts replacing static reports with live ML outputs
- Finance teams forecasting revenue with uncertainty ranges

---

## What's next

- LSTM for deep learning time-series forecasting
- Real-time API data integration (no CSV upload needed)
- Multi-dataset support
- User authentication

---

## Related projects

- [Fraud Detection System](https://github.com/AkashMs24/Cost-Sensitive-Real-Time-Fraud-Detection-Decision-System) — XGBoost + SHAP + FastAPI
- [FarmVoice AI](https://github.com/AkashMs24/FarmVoice-AI) — NLP + Random Forest + SHAP for farmers
- [Employee Attrition XAI](https://github.com/AkashMs24/Employee-Attrition-Risk-Assessment-Using-Explainable-Machine-Learning)

---

Built by **Akash M S** · Presidency University, Bengaluru  
[LinkedIn](https://www.linkedin.com/in/akash-m-s-414a21297) · [GitHub](https://github.com/AkashMs24) · ms29akash@gmail.com
