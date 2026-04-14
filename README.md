# 🧠 Decision Intelligence System

### AI-Powered Executive Dashboard for Real-Time Business Decisions

🔗 **Live App:** https://decision-intelligence-system.streamlit.app/

---

## 🚀 Overview

**Decision Intelligence System** is an end-to-end AI-powered platform that transforms raw business data into **actionable executive insights**.

Unlike traditional dashboards that only describe *what happened*, this system focuses on **what to do next** using machine learning and AI. ([Qualtrics][1])

It enables decision-makers to:

* Forecast future revenue 📈
* Predict customer churn 🔁
* Detect anomalies in business trends 🚨
* Generate AI-driven strategic insights 🤖
* Simulate business scenarios using What-If analysis 🔮

---

## 🧠 Key Features

### 📊 Real-Time Data Processing

* Upload any business CSV
* Automatic data standardization
* Fully dynamic outputs (no static values)

---

### 📈 Revenue Forecasting

* Model: **XGBoost**
* Time-series feature engineering
* Confidence intervals (±1.5σ)
* Performance metric: **R² Score**

---

### 🔁 Churn Prediction

* Model: **Random Forest + Cross Validation**
* Metrics:

  * Accuracy
  * AUC-ROC
  * CV Score
* Feature importance for explainability

---

### 🚨 Anomaly Detection

* Model: **Isolation Forest**
* Detects unusual business patterns
* Severity classification (High / Medium / Low)

---

### 🤖 AI Executive Insights (LLM)

* Powered by **Groq LLaMA-3 70B**
* Converts model outputs → business decisions
* Context-aware recommendations

---

### 🔮 What-If Simulation

* Simulate:

  * Marketing spend changes
  * Churn reduction
  * Customer growth
* Predict revenue impact instantly

---

### 📊 Model Comparison Dashboard

* Compare:

  * Linear Regression
  * Random Forest
  * XGBoost
* Metrics: R², MAE

---

### 🤖 CEO Assistant (AI Chat)

* Ask questions like:

  * "Why is churn increasing?"
  * "How to increase revenue?"
* AI responds using **live data context**

---

## 🏗️ System Architecture

```
User Data (CSV)
      ↓
Data Preprocessing (Auto-detect columns)
      ↓
ML Pipeline
 ├── Forecast (XGBoost)
 ├── Churn (RandomForest)
 ├── Anomaly (IsolationForest)
      ↓
LLM Layer (Groq API)
      ↓
Executive Insights
      ↓
Streamlit Dashboard (UI)
```

---

## ⚙️ Tech Stack

| Category        | Tools                                  |
| --------------- | -------------------------------------- |
| Frontend        | Streamlit                              |
| ML Models       | XGBoost, RandomForest, IsolationForest |
| Data            | Pandas, NumPy                          |
| Visualization   | Plotly                                 |
| AI/LLM          | Groq (LLaMA-3)                         |
| Deployment      | Streamlit Cloud                        |
| Version Control | GitHub                                 |

---

## 📦 Installation

```bash
git clone https://github.com/YOUR_USERNAME/Decision-Intelligence-System.git
cd Decision-Intelligence-System
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔐 Environment Setup

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

---

## 📊 How It Works

Decision Intelligence combines:

* Data
* Machine Learning
* AI

to optimize decision-making and reduce uncertainty in business environments. ([ThoughtSpot][2])

Instead of static dashboards, this system:

* Predicts outcomes
* Simulates scenarios
* Recommends actions

---

## 🎯 Use Cases

* Business Analytics
* Startup Decision Support
* Marketing Optimization
* Customer Retention Strategy
* Financial Forecasting

---

## 📈 Future Improvements

* Deep Learning (LSTM forecasting)
* Real-time API integration
* Multi-dataset support
* User authentication system

---

## 👨‍💻 Author

**Akash M S**
B.Tech Data Science

---

## ⭐ Why This Project Stands Out

* End-to-end AI system (not just dashboard)
* Real-time data-driven decisions
* Combines ML + LLM (modern stack)
* Business-focused, not just technical

---

## 📌 Final Note

This project demonstrates how **Decision Intelligence systems bridge the gap between data and action**, turning analytics into real-world business impact. ([Kairntech][3])

---

⭐ If you like this project, consider giving it a star!

[1]: https://www.qualtrics.com/articles/strategy-research/decision-intelligence/?utm_source=chatgpt.com "The Ultimate Guide to Decision Intelligence (DI)"
[2]: https://www.thoughtspot.com/data-trends/ai/decision-intelligence?utm_source=chatgpt.com "What is Decision Intelligence? Top Examples and Benefits"
[3]: https://kairntech.com/blog/articles/decision-intelligence-platforms-what-they-are-and-why-they-matter/?utm_source=chatgpt.com "Decision intelligence platforms: What they are and why ..."
