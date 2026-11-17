import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

def train_model():
    print("1. Cargando datos procesados de ClinVar...")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train = train_df.drop("CLASS", axis=1)
    y_train = train_df["CLASS"]
    X_test = test_df.drop("CLASS", axis=1)
    y_test = test_df["CLASS"]

    # Configurar MLflow usando base de datos local SQLite (necesaria para Model Registry)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ClinVar_Genetic_Classification")

    # Hiperparámetros del modelo
    params = {
        "n_estimators": 100,
        "max_depth": 12,
        "min_samples_split": 5,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1
    }

    print("2. Iniciando experimento en MLflow y entrenando Random Forest...")
    with mlflow.start_run(run_name="RandomForest_Balanced"):
        # 1. Registrar hiperparámetros en MLflow
        mlflow.log_params(params)

        # Entrenar modelo
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # Predicciones
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calcular métricas
        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4)
        }

        # 2. Registrar métricas en MLflow
        mlflow.log_metrics(metrics)

        # Guardar copia local del modelo y métricas para DVC y FastAPI
        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pkl"
        joblib.dump(model, model_path)

        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # 3. Registrar artefacto y guardar modelo en el Model Registry de MLflow
        mlflow.log_artifact("metrics.json")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="clinvar_model",
            registered_model_name="ClinVar-Classifier",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        print(f"Entrenamiento exitoso. Métricas registradas: {metrics}")

if __name__ == "__main__":
    train_model()
