import joblib
import pandas as pd

def predict(input_data: dict):
    model = joblib.load("models/model.pkl")

    # Rename keys to match training feature names
    renamed = {
        "sepal length (cm)": input_data["sepal_length_cm"],
        "sepal width (cm)": input_data["sepal_width_cm"],
        "petal length (cm)": input_data["petal_length_cm"],
        "petal width (cm)": input_data["petal_width_cm"],
    }

    df = pd.DataFrame([renamed])
    prediction = model.predict(df)[0]
    return prediction