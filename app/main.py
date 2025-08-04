from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI()

# Define input schema
class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def get_prediction(data: InputData):
    result = predict(data.dict())
    return {"prediction": result}