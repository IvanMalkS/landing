"""Airflow DAG for weekly Next Best Offer model retraining."""

from datetime import datetime, timedelta

from airflow import DAG

try:
    from airflow.providers.standard.operators.python import PythonOperator
except ImportError:
    from airflow.operators.python import PythonOperator


def retrain_nbo_model_task():
    import importlib

    nbo_mod = importlib.import_module("projects.04_next_best_offer.nbo")
    generate_synthetic_subscriber_data = nbo_mod.generate_synthetic_subscriber_data
    NBORecommender = nbo_mod.NBORecommender

    X, y = generate_synthetic_subscriber_data(n_samples=1000)
    recommender = NBORecommender()
    recommender.fit(X, y)
    print("NBO Model retrained successfully.")


default_args = {
    "owner": "ml_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "nbo_retrain_weekly",
    default_args=default_args,
    description="Weekly retraining pipeline for Next Best Offer CatBoost model",
    schedule="@weekly",
    start_date=datetime(2025, 1, 1),
    catchup=False,
)

retrain_task = PythonOperator(
    task_id="retrain_nbo_model",
    python_callable=retrain_nbo_model_task,
    dag=dag,
)
