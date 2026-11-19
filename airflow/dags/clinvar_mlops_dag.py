from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "mlops_student",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="clinvar_retraining_and_monitoring_pipeline",
    default_args=default_args,
    description="Pipeline automatizado de preprocesamiento, entrenamiento en MLflow y monitoreo de Data Drift",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "clinvar", "genomics"],
) as dag:

    task_preprocess = BashOperator(
        task_id="preprocess_clinvar_data",
        bash_command="cd /opt/airflow/project && python src/preprocess.py",
    )

    task_train = BashOperator(
        task_id="train_and_register_mlflow_model",
        bash_command="cd /opt/airflow/project && python src/train.py",
    )

    task_monitor = BashOperator(
        task_id="check_data_drift_monitoring",
        bash_command="cd /opt/airflow/project && python monitoring/drift_detector.py",
    )

    # Orden de ejecución del DAG: Preprocesamiento -> Entrenamiento -> Monitoreo
    task_preprocess >> task_train >> task_monitor