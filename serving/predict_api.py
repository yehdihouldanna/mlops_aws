from fastapi import FastAPI
import pandas as pd
import mlflow
import os
import yaml

# Environment variables
MODEL_NAME = "Best_RandomForestClassifier"
MODEL_VERSION = "latest"
# Chargement des paramètres
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
params_path = os.path.join(BASE_DIR, "params.yaml")

params = yaml.safe_load(open(params_path))
mlflow_params = params['mlflow']

# Définition de l'adresse du serveur MLflow
mlflow.set_tracking_uri(mlflow_params["MLFLOW_TRACKING_URI"])


print("Chargement du modèle depuis MLflow Registry...")

try:
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    )
    print("Modèle chargé avec succès")
except Exception as e:
    print("Error loading model:", e)

app = FastAPI(title="API de prédiction ML")


@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"message": "API MLflow opérationnelle"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        prediction = model.predict(df)

        return {
            "prediction": int(prediction[0])
        }
    except Exception as e:
        return {"error": str(e)}
