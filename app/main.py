import os
import csv
from datetime import datetime
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(
    title="ClinVar Genetic Variant Classifier API",
    description="API REST MLOps para predecir si una variante genética tiene clasificación clínica conflictiva/patogénica en revisión.",
    version="1.0.0"
)

templates = Jinja2Templates(directory="app/templates")

# Cargar el modelo y los codificadores entrenados en las Fases 2 y 3
MODEL_PATH = "models/model.pkl"
ENCODERS_PATH = "models/encoders.pkl"

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODERS_PATH)

# Esquema de datos de entrada validado con Pydantic
class VariantInput(BaseModel):
    CHROM: str = "1"
    AF_ESP: float = 0.0012
    AF_EXAC: float = 0.0010
    AF_TGP: float = 0.0015
    CADD_PHRED: float = 24.5
    CADD_RAW: float = 3.42
    LoFtool: float = 0.085
    IMPACT: str = "MODERATE"

# Ruta principal: devuelve la página web interactiva
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

# Ruta de verificación de estado (útil para Docker y pruebas CI/CD)
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

# Ruta de predicción: recibe los datos, predice con el modelo y guarda logs para Monitoreo
@app.post("/predict")
def predict_variant(data: VariantInput):
    input_dict = data.model_dump()
    df = pd.DataFrame([input_dict])

    # Codificar CHROM e IMPACT usando los mismos LabelEncoders del entrenamiento
    for col in ["CHROM", "IMPACT"]:
        le = encoders[col]
        val = str(df[col].iloc[0])
        if val in le.classes_:
            df[col] = le.transform([val])[0]
        else:
            df[col] = 0

    features = [
        "CHROM", "AF_ESP", "AF_EXAC", "AF_TGP",
        "CADD_PHRED", "CADD_RAW", "LoFtool", "IMPACT"
    ]
    X = df[features]

    pred_class = int(model.predict(X)[0])
    prob_conflict = float(model.predict_proba(X)[0][1])

    # Guardar cada predicción en un CSV para la Fase 6 (Monitoreo)
    os.makedirs("monitoring", exist_ok=True)
    log_file = "monitoring/predictions_log.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(features + ["prediction", "probability", "timestamp"])
        writer.writerow(
            [input_dict[c] for c in features] +
            [pred_class, round(prob_conflict, 4), datetime.now().isoformat()]
        )

    label = (
        "Clasificación Conflictiva / Patogenicidad en Revisión (Clase 1)"
        if pred_class == 1
        else "Clasificación Consistente / Benigna (Clase 0)"
    )

    return {
        "prediction": pred_class,
        "label": label,
        "probability_conflict": round(prob_conflict * 100, 2)
    }

