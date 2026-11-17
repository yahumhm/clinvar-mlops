import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

def preprocess_data():
    print("1. Cargando dataset crudo de ClinVar...")
    raw_path = "data/raw/clinvar_raw.csv"
    df = pd.read_csv(raw_path, low_memory=False)

    # Seleccionamos 8 variables bioinformáticas clave + la columna objetivo (CLASS)
    # AF_ESP, AF_EXAC, AF_TGP: Frecuencias alélicas poblacionales
    # CADD_PHRED, CADD_RAW: Puntajes de patogenicidad/daño genético
    # LoFtool: Tolerancia del gen a pérdida de función
    # CHROM: Cromosoma | IMPACT: Impacto molecular (HIGH, MODERATE, LOW, MODIFIER)
    features = [
        "CHROM", "AF_ESP", "AF_EXAC", "AF_TGP", 
        "CADD_PHRED", "CADD_RAW", "LoFtool", "IMPACT"
    ]
    target = "CLASS"

    df = df[features + [target]].copy()

    print("2. Limpiando valores nulos y codificando variables...")
    # Rellenar valores numéricos faltantes con la mediana
    num_cols = ["AF_ESP", "AF_EXAC", "AF_TGP", "CADD_PHRED", "CADD_RAW", "LoFtool"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Convertir variables categóricas (CHROM e IMPACT) a texto limpio y codificarlas
    cat_cols = ["CHROM", "IMPACT"]
    encoders = {}
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna("MISSING")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Guardar los codificadores para usarlos después en la API de FastAPI
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoders, "models/encoders.pkl")

    print("3. Dividiendo en conjuntos de entrenamiento (80%) y prueba (20%)...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[target]
    )

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)
    print(f"Preprocesamiento completado: Train={train_df.shape}, Test={test_df.shape}")

if __name__ == "__main__":
    preprocess_data()