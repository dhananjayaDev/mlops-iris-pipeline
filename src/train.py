import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# 🔹 Load parameters from params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

test_size = params["test_size"]
random_state = params["random_state"]
n_estimators = params["n_estimators"]

# 🔹 Load dataset
df = pd.read_csv("data/iris.csv")
X = df.drop(columns="species")
y = df["species"]

# 🔹 Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# 🔹 Start MLflow experiment
mlflow.set_experiment("iris_classification")

with mlflow.start_run():
    # 🔹 Train model
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)

    # 🔹 Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 🔹 Log parameters and metrics
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("test_size", test_size)
    mlflow.log_metric("accuracy", acc)

    # 🔹 Log model to MLflow
    mlflow.sklearn.log_model(clf, "model")

    # 🔹 Save model locally for DVC tracking
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/model.pkl")

    print(f"✅ Model trained with accuracy: {acc:.4f}")