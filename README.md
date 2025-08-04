# MLOps Project: Iris Classification Pipeline

A minimal, production-ready MLOps pipeline that trains, tracks, versions, and serves a machine learning model using modern tools.

---

## Features

- ✅ Model training with **scikit-learn**
- ✅ Experiment tracking via **MLflow**
- ✅ Data and model versioning with **DVC**
- ✅ REST API for predictions using **FastAPI**
- ✅ Containerized with **Docker**
- ✅ Configurable via `params.yaml`
- ✅ Remote storage for reproducibility

---

## Project Structure

```
mlops_project/
├── data/              # Raw dataset (iris.csv)
├── models/            # Trained model artifacts
├── src/               # Training and prediction logic
│   ├── train.py
│   ├── predict.py
├── app/               # FastAPI app
│   └── main.py
├── params.yaml        # Hyperparameters
├── dvc.yaml           # DVC pipeline definition
├── requirements.txt   # Python dependencies
├── Dockerfile         # Container setup
├── .dockerignore
├── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd mlops_project
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# OR
source venv/bin/activate      # Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
dvc repro
```

### 5. Serve Predictions via FastAPI

```bash
uvicorn app.main:app --reload
```

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

---

## Docker Usage

### Build the Image

```bash
docker build -t mlops-app .
```

### Run the Container

```bash
docker run -p 8000:8000 mlops-app
```

---

## DVC Remote Setup (Optional)

```bash
dvc remote add -d localremote ../dvc-storage
dvc push
```

---

## MLflow UI (Optional)

```bash
mlflow ui
```

Visit [http://localhost:5000](http://localhost:5000) to explore experiments.

---

## Example Prediction

```json
POST /predict
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

---

## License

MIT License. See `[https://github.com/dhananjayaDev/mlops-iris-pipeline?tab=MIT-1-ov-file](LICENSE)` file for details.

---

## Acknowledgments

Built with ❤️ using scikit-learn, MLflow, DVC, FastAPI, and Docker.



