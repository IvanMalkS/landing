"""Project 01: Support Ticket Classifier (40+ categories in telecom)"""

from typing import Any, Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

CATEGORIES = [f"topic_{i:02d}" for i in range(1, 41)]

TOPIC_KEYWORDS = {
    "topic_01": ["balans", "schet", "popolnenie", "dengi", "spisanie"],
    "topic_02": ["internet", "skorost", "trafik", "megabayt", "gigabayt"],
    "topic_03": ["rouming", "za granicey", "poezdka", "strana", "svyaz"],
    "topic_04": ["tarif", "smena tarifa", "usloviya", "abonentskaya plata", "plan"],
    "topic_05": ["esim", "sim karta", "zamena sim", "blokirovka", "puk"],
}

# Fill remaining topics with generic ASCII keyword templates
for idx in range(6, 41):
    top_name = f"topic_{idx:02d}"
    TOPIC_KEYWORDS[top_name] = [
        f"kategoriya_{idx}",
        f"vopros_{idx}",
        f"problema_{idx}",
        f"servis_{idx}",
    ]


def generate_synthetic_tickets(num_samples_per_topic: int = 100) -> Tuple[List[str], List[str]]:
    """Generates synthetic telecom customer tickets for 40 topics."""
    texts = []
    labels = []
    np.random.seed(42)

    templates = [
        "vopros po {kw}",
        "sboy {kw}",
        "pomogite s {kw}",
        "uznat pro {kw}",
        "ploho rabotaet {kw}",
    ]

    for topic, kws in TOPIC_KEYWORDS.items():
        for _ in range(num_samples_per_topic):
            kw = np.random.choice(kws)
            tmpl = np.random.choice(templates)
            text = tmpl.format(kw=kw)
            texts.append(text)
            labels.append(topic)

    return texts, labels


class TicketClassifierModel:
    def __init__(self, confidence_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 1))),
            ("clf", LogisticRegression(C=10.0, max_iter=200)),
        ])
        self.classes_: np.ndarray | None = None

    def fit(self, X: List[str], y: List[str]):
        self.pipeline.fit(X, y)
        self.classes_ = self.pipeline.classes_
        return self

    def predict_with_confidence(self, texts: List[str]) -> List[Dict[str, Any]]:
        probs = self.pipeline.predict_proba(texts)
        results = []
        for prob in probs:
            max_idx = np.argmax(prob)
            max_prob = float(prob[max_idx])
            topic = str(self.classes_[max_idx])
            auto_routed = max_prob >= self.confidence_threshold
            results.append({
                "topic": topic if auto_routed else "operator_escalation",
                "predicted_topic": topic,
                "confidence": max_prob,
                "auto_routed": auto_routed,
            })
        return results

    def export_onnx(self, filepath: str):
        initial_type = [("string_input", StringTensorType([None, 1]))]
        clf_step = self.pipeline.named_steps["clf"]
        options = {id(clf_step): {"zipmap": False}}
        onnx_model = convert_sklearn(self.pipeline, initial_types=initial_type, options=options)
        with open(filepath, "wb") as f:
            f.write(onnx_model.SerializeToString())


class ONNXInferenceSession:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        input_data = np.array(texts, dtype=object).reshape(-1, 1)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_data})
        probs = outputs[1]
        return probs
