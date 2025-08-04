import joblib
import pandas as pd

def predict(input_data):
    model = joblib.load("models/model.pkl")
    df = pd.DataFrame([input_data])
    return model.predict(df)[0]