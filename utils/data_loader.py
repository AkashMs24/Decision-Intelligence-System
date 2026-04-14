import pandas as pd

def load_data(path="data/sample_data.csv"):
    df = pd.read_csv(path, encoding='ISO-8859-1')

    print("DEBUG columns:", df.columns)  # 👈 IMPORTANT

    df['date'] = pd.to_datetime(df['date'])
    return df
