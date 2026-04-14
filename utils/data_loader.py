import pandas as pd

def load_data(path="data/sample_data.csv"):
    df = pd.read_csv(path, encoding='ISO-8859-1')
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df
