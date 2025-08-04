from src.predict import predict_species

def test_prediction_output():
    sample = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    result = predict_species(sample)
    assert result in ["setosa", "versicolor", "virginica"], "Unexpected prediction result"