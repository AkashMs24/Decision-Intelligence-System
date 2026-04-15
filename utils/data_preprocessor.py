# utils/data_preprocessor.py
import pandas as pd
import numpy as np

def transform_user_data(raw: pd.DataFrame, filename: str = "Unknown") -> pd.DataFrame:
    df = raw.copy()
    original_columns = list(raw.columns)
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    # Business-like detection
    business_keywords = ["revenue", "sales", "amount", "price", "income", "spending", "purchase", "order", "fare", "total"]
    is_business_like = any(any(k in col for k in business_keywords) for col in df.columns)

    date_col = next((c for c in df.columns if any(k in c for k in ["date", "time", "day", "month", "invoice", "period"])), None)
    rev_col = next((c for c in df.columns if any(k in c for k in business_keywords)), None)

    # Smart conversion logic
    if "survived" in df.columns or "titanic" in filename.lower() or not is_business_like:
        # Non-business dataset (Titanic, Iris, etc.)
        df.attrs['dataset_type'] = "non_business"
        df["revenue"] = np.random.uniform(8000, 90000, len(df))   # synthetic but reasonable
        df["customers"] = np.random.randint(50, 350, len(df))
        df.attrs['is_business_like'] = False
    elif rev_col:
        # Real business dataset
        df["revenue"] = pd.to_numeric(df[rev_col], errors="coerce").fillna(0)
        df["customers"] = np.random.randint(50, 350, len(df))
        df.attrs['dataset_type'] = "business"
        df.attrs['is_business_like'] = True
    else:
        # Generic fallback
        df["revenue"] = np.random.uniform(10000, 120000, len(df))
        df["customers"] = np.random.randint(50, 350, len(df))
        df.attrs['dataset_type'] = "generic"
        df.attrs['is_business_like'] = False

    # Date handling
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors='coerce').dt.date
    else:
        df["date"] = pd.date_range(start="2023-01-01", periods=len(df)).date

    df["marketing_spend"] = df["revenue"] * 0.20
    df["churn"] = (df["customers"].pct_change() < 0).astype(int).fillna(0)

    final_df = df[["date", "revenue", "customers", "marketing_spend", "churn"]].reset_index(drop=True)

    # Metadata for dashboard and LLM
    final_df.attrs['filename'] = filename
    final_df.attrs['num_rows'] = len(final_df)
    final_df.attrs['original_columns'] = original_columns
    final_df.attrs['is_business_like'] = is_business_like

    return final_df
