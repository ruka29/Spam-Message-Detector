from joblib import load
import pandas as pd

def detect_anomalies(new_messages):
    model = load('models/anomaly_model.pkl')
    vectorizer = load('models/vectorizer.pkl')
    X_new = vectorizer.transform(new_messages)
    predictions = model.predict(X_new)
    return predictions