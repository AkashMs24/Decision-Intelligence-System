import pandas as pd
import numpy as np

def transform_user_data(raw: pd.DataFrame, filename: str = "Unknown") -> pd.DataFrame:
    df = raw.copy()
    original_columns = list(df.columns)
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Detect important columns
    date_col = next((c for c in df.columns if any(k in c for k in ["date", "time", "day", "month", "invoice", "period"])), None)
    rev_keywords = ["revenue", "sales", "amount", "total", "price", "value", "income", "spending", "purchase"]
    rev_col = next((c for c in df.columns if any(k in c for k in rev_keywords)), None)
    cust_col = next((c for c in df.columns if any(k in c for k in ["customer", "user", "client", "id"])), None)

    # Special case: Mall_Customers / Customer segmentation datasets
    if "income" in df.columns and "spending_score" in df.columns:
        df["revenue"] = df["income"] * (df["spending_score"] / 50.0)   # reasonable synthetic revenue
        df["customers"] = np.random.randint(80, 400, len(df))
        df.attrs['dataset_type'] = "customer_segmentation"
    elif rev_col:
        df["revenue"] = pd.to_numeric(df[rev_col], errors="coerce").fillna(df[rev_col].mean() if rev_col else 0)
        df["customers"] = pd.to_numeric(df.get(cust_col, 100), errors="coerce").fillna(100)
        df.attrs['dataset_type'] = "sales_revenue"
    else:
        # Generic fallback - create reasonable synthetic data
        df["revenue"] = np.random.uniform(5000, 150000, len(df))
        df["customers"] = np.random.randint(50, 500, len(df))
        df.attrs['dataset_type'] = "generic"

    # Date handling
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors='coerce').dt.date
    else:
        df["date"] = pd.date_range(start="2023-01-01", periods=len(df)).date

    # Required columns for models
    df["marketing_spend"] = df["revenue"] * 0.20
    df["churn"] = (df["customers"].pct_change() < 0).astype(int).fillna(0)

    final_df = df[["date", "revenue", "customers", "marketing_spend", "churn"]].reset_index(drop=True)

    # Attach metadata for LLM and warnings
    final_df.attrs['filename'] = filename
    final_df.attrs['num_rows'] = len(final_df)
    final_df.attrs['original_columns'] = original_columns
    final_df.attrs['detected_revenue_col'] = rev_col

    return final_df
