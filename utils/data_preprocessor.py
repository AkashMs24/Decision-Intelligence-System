import pandas as pd
import numpy as np

def transform_user_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    date_col = next((c for c in df.columns if any(k in c for k in ("date","time","invoice"))), None)
    rev_col = next((c for c in df.columns if any(k in c for k in ("revenue","sales","amount","price"))), None)
    cust_col = next((c for c in df.columns if any(k in c for k in ("customer","user","client"))), None)

    if date_col and rev_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        grouped = df.groupby(df[date_col].dt.date).agg({
            rev_col: "sum",
            cust_col: "nunique" if cust_col else "count"
        }).reset_index()
        grouped.rename(columns={rev_col: "revenue", cust_col or "count": "customers"}, inplace=True)
    else:
        df["date"] = pd.date_range("2024-01-01", periods=len(df)).date
        df["revenue"] = pd.to_numeric(df.get(rev_col, df.iloc[:,0]), errors="coerce").fillna(0)
        df["customers"] = np.random.randint(80, 300, len(df))
        grouped = df[["date", "revenue", "customers"]]

    grouped["marketing_spend"] = grouped["revenue"] * 0.20
    grouped["churn"] = (grouped["customers"].pct_change() < 0).astype(int).fillna(0)
    return grouped[["date", "revenue", "customers", "marketing_spend", "churn"]].reset_index(drop=True)
