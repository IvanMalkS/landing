import importlib

import numpy as np
from sklearn.metrics import accuracy_score

nbo_mod = importlib.import_module("projects.04_next_best_offer.nbo")
NBORecommender = nbo_mod.NBORecommender
calculate_psi = nbo_mod.calculate_psi
generate_synthetic_subscriber_data = nbo_mod.generate_synthetic_subscriber_data


def test_nbo_catboost_better_than_random_baseline():
    X_train, y_train = generate_synthetic_subscriber_data(n_samples=2000, seed=42)
    X_test, y_test = generate_synthetic_subscriber_data(n_samples=500, seed=123)

    model = NBORecommender()
    model.fit(X_train, y_train)

    top1_preds = [p[0] for p in model.predict_top_n(X_test, top_n=1)]
    model_acc = accuracy_score(y_test, top1_preds)

    # Random baseline accuracy (1 / 5 = 0.20)
    baseline_acc = 0.20
    assert model_acc > baseline_acc + 0.3, f"Model accuracy {model_acc} is not significantly better than baseline {baseline_acc}"


def test_airflow_dag_import_and_schedule():
    dag_mod = importlib.import_module("projects.04_next_best_offer.retrain_dag")
    dag = dag_mod.dag
    assert dag is not None
    assert dag.dag_id == "nbo_retrain_weekly"
    schedule_val = getattr(dag, "timetable", None) or getattr(dag, "schedule_interval", None)
    assert schedule_val is not None
    assert len(dag.tasks) >= 1


def test_psi_drift_detection():
    np.random.seed(42)
    reference = np.random.normal(loc=100.0, scale=10.0, size=1000)

    # Identical distribution -> low PSI
    same = np.random.normal(loc=100.0, scale=10.0, size=1000)
    psi_low = calculate_psi(reference, same)
    assert psi_low < 0.1, f"Low PSI test failed: {psi_low}"

    # Shifted distribution -> high PSI
    shifted = np.random.normal(loc=130.0, scale=10.0, size=1000)
    psi_high = calculate_psi(reference, shifted)
    assert psi_high > 0.25, f"High PSI test failed: {psi_high}"
