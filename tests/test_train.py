import os
import pytest
from src.train import train_model

def test_model_training_creates_file():
    model_path = "models/iris_classifier.pkl"
    if os.path.exists(model_path):
        os.remove(model_path)

    train_model()
    assert os.path.exists(model_path), "Model file was not created"