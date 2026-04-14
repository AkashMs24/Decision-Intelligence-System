import pandas as pd

def transform_user_data(df):

    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Detect columns
    revenue_col = None
    customer_col = None
    date_col = None

    for col in df.columns:
        if "revenue" in col or "sales" in col or "amount" in col:
            revenue_col = col

    for col in df.columns:
        if "customer" in col or "user" in col or "client" in col:
            customer_col = col

    for col in df.columns:
        if "date" in col or "time" in col:
            date_col = col

    # Special case (Mall dataset)
    income_col = None
    score_col = None

    for col in df.columns:
        if "income" in col:
            income_col = col
        if "score" in col:
            score_col = col

    if income_col and score_col:
        df["revenue"] = df[income_col] * df[score_col]

    # General case
    if revenue_col and "revenue" not in df.columns:
        df["revenue"] = df[revenue_col]

    # Customers fallback
    df["customers"] = 1

    # Date handling
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["date"] = pd.date_range(start="2024-01-01", periods=len(df))

    # Validate revenue
    if "revenue" not in df.columns:
        raise ValueError("❌ Could not detect revenue column")

    # Add required columns
    df["marketing_spend"] = df["revenue"] * 0.2
    df["churn"] = 0

    return df[["date", "revenue", "customers", "marketing_spend", "churn"]]
