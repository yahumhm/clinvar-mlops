import os
import json
from datetime import datetime
import pandas as pd
from scipy.stats import ks_2samp

def detect_data_drift():
    print("1. Iniciando análisis de monitoreo y detección de Data Drift...")
    ref_path = "data/processed/train.csv"
    log_path = "monitoring/predictions_log.csv"

    if not os.path.exists(ref_path):
        print("Error: No se encontró el dataset de referencia (train.csv).")
        return

    ref_df = pd.read_csv(ref_path)

    # Si aún no hay suficientes logs de la API, simulamos un lote nuevo de variantes para evaluar
    if not os.path.exists(log_path) or len(pd.read_csv(log_path)) < 5:
        print("Pocos registros en predictions_log.csv. Generando muestra de producción simulada...")
        curr_df = ref_df.sample(n=100, random_state=99).copy()
        # Introducimos una ligera variación en CADD_PHRED para probar la detección
        curr_df["CADD_PHRED"] = curr_df["CADD_PHRED"] * 1.15
    else:
        curr_df = pd.read_csv(log_path)

    num_features = ["AF_ESP", "AF_EXAC", "AF_TGP", "CADD_PHRED", "CADD_RAW", "LoFtool"]
    drift_report = {
        "timestamp": datetime.now().isoformat(),
        "reference_samples": len(ref_df),
        "production_samples": len(curr_df),
        "features": {},
        "overall_drift_detected": False
    }

    drift_count = 0
    for col in num_features:
        if col in curr_df.columns:
            # Prueba estadística Kolmogorov-Smirnov (p-value < 0.05 indica cambio significativo)
            stat, p_value = ks_2samp(ref_df[col].dropna(), curr_df[col].dropna())
            is_drift = bool(p_value < 0.05)
            if is_drift:
                drift_count += 1

            drift_report["features"][col] = {
                "ref_mean": round(float(ref_df[col].mean()), 4),
                "prod_mean": round(float(curr_df[col].mean()), 4),
                "ks_statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 4),
                "drift_detected": is_drift
            }

    drift_report["overall_drift_detected"] = drift_count > 0
    drift_report["drifted_features_count"] = drift_count

    os.makedirs("monitoring", exist_ok=True)
    report_path = "monitoring/drift_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(drift_report, f, indent=4)

    print(f"Reporte de monitoreo guardado en {report_path}")
    print(f"Variables con Data Drift detectado: {drift_count}/{len(num_features)}")

if __name__ == "__main__":
    detect_data_drift()