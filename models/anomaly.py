def detect_anomalies(df):

    mean = df['revenue'].mean()
    std = df['revenue'].std()

    df['z_score'] = (df['revenue'] - mean) / std

    # 🔥 Increase threshold
    anomalies = df[abs(df['z_score']) > 2.5]

    return anomalies[['date', 'revenue']].to_dict(orient='records')
