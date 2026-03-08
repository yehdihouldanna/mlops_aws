# Déploiement d’un modèle MLflow avec Docker

## Objectif du laboratoire

Dans ce laboratoire, nous allons compléter notre pipeline MLOps en ajoutant une étape d’exploitation du modèle (inference / serving).

L’objectif est de :

1. Charger automatiquement le **dernier modèle enregistré dans le MLflow Registry**
2. Créer une **API de prédiction**
3. Conteneuriser cette API avec **Docker**
4. Déployer l’API sur une instance **EC2**
5. Communiquer avec le serveur **MLflow** via son **adresse IP**

Le modèle utilisé est :

```
Best_RandomForestClassifier
```

qui a été enregistré dans le **MLflow Model Registry** lors de l'étape d'entraînement.

---

# Structure du projet

Nous allons ajouter un dossier pour l’exploitation du modèle.

Structure finale :

```
mlops_aws
│
├── data
├── models
├── params.yaml # this is put in git ignore because it contains sensitive ID and Key, a reference of it is given instead
├── README.md
│
├── src
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│
├── serving
│   └── predict_api.py
│
├── requirements.txt
└── Dockerfile
```

---

# Étape 1 — Création du script d’inférence

Créer un dossier :

```
serving/
```

Puis créer un fichier :

```
predict_api.py
```

Ce script va :

- se connecter au **MLflow Tracking Server**
- charger le **dernier modèle enregistré**
- exposer une **API de prédiction**

---

# predict_api.py

```python
from fastapi import FastAPI
import pandas as pd
import mlflow
import os
import yaml

# Chargement des paramètres
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
params_path = os.path.join(BASE_DIR, "params.yaml")

params = yaml.safe_load(open(params_path))
mlflow_params = params['mlflow']

# Définition de l'adresse du serveur MLflow
mlflow.set_tracking_uri(mlflow_params["MLFLOW_TRACKING_URI"])

MODEL_NAME = "Best_RandomForestClassifier"
MODEL_VERSION = "latest"

print("Chargement du modèle depuis MLflow Registry...")

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}"
)

print("Modèle chargé avec succès")

app = FastAPI(title="API de prédiction ML")

@app.get("/")
def home():
    return {"message": "API MLflow opérationnelle"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return {
        "prediction": int(prediction[0])
    }
```

---

# Étape 2 — Configuration de la connexion au serveur MLflow

Même si nous utilisons **la même instance EC2**, nous allons communiquer avec MLflow via son **adresse IP**.

Cela permet de garder une architecture compatible si les services sont déployés sur **plusieurs serveurs EC2**.

Modifier le fichier :

```
params.yaml
```

Exemple :

```yaml
mlflow:

  MLFLOW_TRACKING_URI: http://54.210.120.25:5000
```

où :

```
54.210.120.25 = IP publique de l'instance EC2
5000 = port du serveur MLflow
```

---

# Étape 3 — Création du fichier requirements.txt

Créer le fichier :

```
requirements.txt
```

Contenu :

```
fastapi
uvicorn
pandas
scikit-learn
mlflow
boto3
s3fs
pyyaml
```

---

# Étape 4 — Création du Dockerfile

Nous allons maintenant conteneuriser notre API avec Docker.

Créer le fichier :

```
Dockerfile
```

---

# Dockerfile

```
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "serving.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---
# Etape 5 - Clonning du code du serveur : 

Actuellement nous clonons la totalité du repository pour la simplicité : 
mais un approche plus propre, on clonne uniquement les fichier nécessaire pour le serveur de prediciton : 

```bash
.
├── params.yaml # Ceci devrait etre recreé manuellement pour stocker les secret aws id et key, et aussi le tracking_uri
├── Dockerfile
├── requirements.txt
├── serving
│   └── predict_api.py
```

# Étape 5 — Construction de l’image Docker

Dans le dossier racine du projet :

Recrée votre fichier params.yaml (ceci est dans git ignore à cause des secret id et key)
du coup après le clonage du dossier de code, on doit recrée le fichier params.yaml

```
docker build -t mlflow-model-api .
```

---

# Étape 6 — Lancement du conteneur

```
docker run -p 8000:8000 mlflow-model-api


```

Mais pour notre container pour qu'il puisse communiquer avec s3 on doit le lancer avec les crédentiels;
Also allow the port 8000 in the sg for the ec2 instance,

```bash
docker run -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID="your_key" \
  -e AWS_SECRET_ACCESS_KEY="your_secret" \
  -e AWS_DEFAULT_REGION="your_region" \
  mlflow-model-api

```

L’API sera accessible à l’adresse :

```
http://EC2_PUBLIC_IP:8000
```

---

# Étape 7 — Test de l’API

Tester avec curl :

```
curl -X POST http://EC2_PUBLIC_IP:8000/predict \
-H "Content-Type: application/json" \
-d '{
"Pregnancies": 6,
"Glucose": 148,
"BloodPressure": 72,
"SkinThickness": 35,
"Insulin": 0,
"BMI": 33.6,
"DiabetesPedigreeFunction": 0.627,
"Age": 50
}'
```

Réponse attendue :

```
{
 "prediction": 1
}
```

---

# Architecture finale

Architecture du système :

```
                +----------------------+
                |   MLflow Tracking    |
                |       Server         |
                |    EC2 :5000         |
                +----------+-----------+
                           |
                           | accès au modèle
                           |
                   +-------v--------+
                   |   Docker API   |
                   |    FastAPI     |
                   |    port 8000   |
                   +-------+--------+
                           |
                           |
                     Requêtes utilisateur
```

Même si les deux services sont sur **le même EC2**, ils communiquent via **l’adresse IP**.

Cela permet facilement :

- de séparer les services sur plusieurs machines
- de scaler l’API indépendamment du serveur MLflow.

---

# Configuration du Security Group

Vérifier que l’instance EC2 autorise les ports :

```
5000   MLflow Server
8000   API de prédiction
```

---

# Vérifier que MLflow fonctionne

Commande de lancement du serveur MLflow :

```
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root s3://your-bucket \
--host 0.0.0.0 \
--port 5000
```

---

# Résultat attendu

À la fin de ce laboratoire, vous aurez :

- un modèle entraîné et enregistré dans MLflow
- un modèle récupéré automatiquement depuis le **Model Registry**
- une API de prédiction
- un service conteneurisé avec **Docker**
- un service d’inférence accessible via HTTP

Cela constitue une **architecture standard d’exploitation d’un modèle en production** dans un pipeline **MLOps**.





creation de dossier et de fichier sur EC2,
<!-- Dockerfile,
requirements -->
generatino de clé ssh,
ajout de deploy key sur git
pull le code sur ec2
