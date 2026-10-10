import importlib
import os
import tempfile

import numpy as np
import yaml
from fastapi.testclient import TestClient
from sklearn.metrics import f1_score, precision_score
from sklearn.model_selection import train_test_split

app_mod = importlib.import_module("projects.01_ticket_classifier.app")
app = app_mod.app

model_mod = importlib.import_module("projects.01_ticket_classifier.model")
ONNXInferenceSession = model_mod.ONNXInferenceSession
TicketClassifierModel = model_mod.TicketClassifierModel
generate_synthetic_tickets = model_mod.generate_synthetic_tickets


def test_ticket_classifier_performance_and_auto_routing():
    texts, labels = generate_synthetic_tickets(num_samples_per_topic=100)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = TicketClassifierModel(confidence_threshold=0.4)
    model.fit(X_train, y_train)

    # Macro F1 on holdout
    preds = model.pipeline.predict(X_test)
    macro_f1 = f1_score(y_test, preds, average="macro")
    assert macro_f1 >= 0.8, f"Macro F1 {macro_f1} is lower than 0.8"

    # Auto-routing share and precision
    results = model.predict_with_confidence(X_test)
    auto_routed = [r for r, y in zip(results, y_test) if r["auto_routed"]]
    auto_route_share = len(auto_routed) / len(results)
    assert auto_route_share >= 0.4, f"Auto routing share {auto_route_share} is lower than 0.4"

    auto_routed_true = [y for r, y in zip(results, y_test) if r["auto_routed"]]
    auto_routed_pred = [r["predicted_topic"] for r in auto_routed]
    precision = precision_score(auto_routed_true, auto_routed_pred, average="macro")
    assert precision >= 0.9, f"Auto routed precision {precision} is lower than 0.9"


def test_onnx_export_and_inference():
    texts, labels = generate_synthetic_tickets(num_samples_per_topic=20)
    model = TicketClassifierModel()
    model.fit(texts, labels)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")
        model.export_onnx(onnx_path)
        assert os.path.exists(onnx_path)

        onnx_sess = ONNXInferenceSession(onnx_path)
        sample_texts = texts[:5]
        sklearn_probs = model.pipeline.predict_proba(sample_texts)
        onnx_probs = onnx_sess.predict_proba(sample_texts)

        np.testing.assert_allclose(sklearn_probs, onnx_probs, rtol=1e-4, atol=1e-4)


def test_fastapi_service():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    response = client.post("/classify", json={"text": "Хочу узнать свой баланс на счету"})
    assert response.status_code == 200
    data = response.json()
    assert "topic" in data
    assert "confidence" in data
    assert "auto_routed" in data


def test_k8s_manifests():
    k8s_path = "projects/01_ticket_classifier/k8s.yaml"
    assert os.path.exists(k8s_path)
    with open(k8s_path, "r", encoding="utf-8") as f:
        docs = list(yaml.safe_load_all(f))

    assert len(docs) >= 2
    deployment = next(d for d in docs if d["kind"] == "Deployment")
    service = next(d for d in docs if d["kind"] == "Service")

    assert deployment["metadata"]["name"] == "ticket-classifier-deployment"
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert "livenessProbe" in container
    assert "readinessProbe" in container
    assert service["metadata"]["name"] == "ticket-classifier-service"
