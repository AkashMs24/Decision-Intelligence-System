# utils/data_preprocessor.py
import pandas as pd
import numpy as np


# Keywords that indicate a revenue / sales column
_REVENUE_KEYWORDS = [
    "revenue", "sales", "amount", "price", "income",
    "spending", "purchase", "order", "fare", "total",
    "turnover", "gmv", "net",
]

_DATE_KEYWORDS = ["date", "time", "day", "month", "year", "invoice", "period", "week"]


def _find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Return the first column whose name contains any of the given keywords."""
    for col in df.columns:
        if any(k in col for k in keywords):
            return col
    return None


def transform_user_data(raw: pd.DataFrame, filename: str = "Unknown") -> pd.DataFrame:
    """
    Normalize any uploaded (or default) CSV into a standard 5-column frame:
        date | revenue | customers | marketing_spend | churn

    Attributes stored on the returned frame:
        filename, num_rows, original_columns, is_business_like, dataset_type
    """
    df = raw.copy()
    original_columns = list(raw.columns)

    # ── 1. Normalise column names ──────────────────────────────────────────────
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]

    # ── 2. Detect column roles ─────────────────────────────────────────────────
    date_col = _find_col(df, _DATE_KEYWORDS)
    rev_col  = _find_col(df, _REVENUE_KEYWORDS)

    # ── 3. Dataset-type classification ────────────────────────────────────────
    filename_lower = filename.lower()

    is_titanic = (
        "titanic" in filename_lower
        or "survived" in df.columns
        or "pclass" in df.columns
    )

    is_business_like = (
        rev_col is not None
        or any(any(k in c for k in _REVENUE_KEYWORDS) for c in df.columns)
        or "sample_data" in filename_lower          # always treat default sample as business
    )

    # ── 4. Build revenue series ───────────────────────────────────────────────
    if is_titanic:
        dataset_type = "non_business"
        is_business_like = False
        if "fare" in df.columns:
            df["revenue"] = pd.to_numeric(df["fare"], errors="coerce").fillna(50.0) * 1_000
        else:
            rng = np.random.default_rng(0)
            df["revenue"] = rng.uniform(8_000, 90_000, len(df))

    elif rev_col:
        dataset_type = "business"
        df["revenue"] = pd.to_numeric(df[rev_col], errors="coerce").fillna(
            pd.to_numeric(df[rev_col], errors="coerce").median()
        )
        # If median is also NaN (all bad rows) fall back to 50 000
        df["revenue"] = df["revenue"].fillna(50_000)

    else:
        dataset_type = "generic"
        rng = np.random.default_rng(42)
        df["revenue"] = rng.uniform(10_000, 120_000, len(df))

    # ── 5. Build customers series ─────────────────────────────────────────────
    cust_col = _find_col(df, ["customer", "client", "user", "passenger", "buyer"])
    if cust_col and cust_col != rev_col:
        df["customers"] = pd.to_numeric(df[cust_col], errors="coerce").fillna(200).astype(int)
    else:
        rng = np.random.default_rng(1)
        df["customers"] = rng.integers(80, 400, len(df))

    # ── 6. Build date series ──────────────────────────────────────────────────
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    else:
        df["date"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D").date

    # Drop rows where date parsing completely failed
    df = df.dropna(subset=["date"])

    # ── 7. Derived columns ────────────────────────────────────────────────────
    df["marketing_spend"] = df["revenue"] * 0.20
    df["churn"] = (df["customers"].pct_change() < 0).astype(int)
    df["churn"] = df["churn"].fillna(0).astype(int)

    # ── 8. Build final frame ──────────────────────────────────────────────────
    final_df = (
        df[["date", "revenue", "customers", "marketing_spend", "churn"]]
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Attach metadata
    final_df.attrs["filename"]         = filename
    final_df.attrs["num_rows"]         = len(final_df)
    final_df.attrs["original_columns"] = original_columns
    final_df.attrs["is_business_like"] = is_business_like
    final_df.attrs["dataset_type"]     = dataset_type

    return final_df
