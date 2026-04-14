from sklearn.linear_model import LinearRegression
import numpy as np

def forecast_revenue(df):
    df = df.copy()
    df['time_index'] = np.arange(len(df))

    X = df[['time_index']]
    y = df['revenue']

    model = LinearRegression()
    model.fit(X, y)

    future = np.array([[len(df) + i] for i in range(3)])
    preds = model.predict(future)

    return preds.tolist()
