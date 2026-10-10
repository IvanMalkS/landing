import importlib

import pytest
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

pipe_mod = importlib.import_module("projects.02_call_analysis.pipeline")
CallAnalysisPipeline = pipe_mod.CallAnalysisPipeline
FasterWhisperASR = pipe_mod.FasterWhisperASR
ObjectionClassifier = pipe_mod.ObjectionClassifier
SCRIPT_HINTS = pipe_mod.SCRIPT_HINTS
TextStubASR = pipe_mod.TextStubASR
generate_synthetic_call_data = pipe_mod.generate_synthetic_call_data


def test_objection_classifier_f1():
    texts, labels = generate_synthetic_call_data()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    clf = ObjectionClassifier()
    clf.fit(X_train, y_train)

    preds = [clf.predict(t) for t in X_test]
    f1 = f1_score(y_test, preds, average="macro")
    assert f1 >= 0.8, f"Objection classifier F1 score {f1} is lower than 0.8"


def test_script_hint_returns_for_all_objections():
    clf = ObjectionClassifier()
    texts, labels = generate_synthetic_call_data()
    clf.fit(texts, labels)

    for objection, hint in SCRIPT_HINTS.items():
        predicted = clf.predict(objection)
        assert predicted == objection
        assert hint is not None
        assert len(hint) > 10


def test_end_to_end_call_analysis_pipeline():
    texts, labels = generate_synthetic_call_data()
    clf = ObjectionClassifier()
    clf.fit(texts, labels)

    asr_stub = TextStubASR()
    pipeline = CallAnalysisPipeline(asr=asr_stub, classifier=clf)

    res = pipeline.process_call("Это очень дорого для нас")
    assert res["transcription"] == "Это очень дорого для нас"
    assert res["detected_objection"] == "дорого"
    assert res["script_hint"] == SCRIPT_HINTS["дорого"]


@pytest.mark.slow
def test_faster_whisper_slow():
    asr = FasterWhisperASR("tiny")
    res = asr.transcribe("тестовая аудио запись")
    assert isinstance(res, str)
