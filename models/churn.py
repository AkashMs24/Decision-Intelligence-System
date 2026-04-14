from sklearn.ensemble import RandomForestClassifier

def churn_analysis(df):
    X = df[['revenue', 'customers', 'marketing_spend']]
    y = df['churn']

    model = RandomForestClassifier()
    model.fit(X, y)

    importance = model.feature_importances_

    return {
        "feature_importance": dict(zip(X.columns, importance))
    }
