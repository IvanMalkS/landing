import importlib

import pytest
from sklearn.metrics import cohen_kappa_score

judge_mod = importlib.import_module("projects.05_llm_judge.judge")
LLMJudgeEvaluator = judge_mod.LLMJudgeEvaluator
calculate_cohens_kappa = judge_mod.calculate_cohens_kappa
generate_synthetic_dialogues = judge_mod.generate_synthetic_dialogues
run_ci_gate = judge_mod.run_ci_gate


def test_custom_cohens_kappa_matches_sklearn():
    y_true = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1]
    y_pred = [1, 2, 3, 4, 4, 1, 2, 2, 4, 5, 5, 4, 3, 1, 1]

    custom_kappa = calculate_cohens_kappa(y_true, y_pred)
    sklearn_kappa = float(cohen_kappa_score(y_true, y_pred))

    assert pytest.approx(custom_kappa, abs=1e-5) == sklearn_kappa


def test_evaluator_300_dialogues_and_metrics():
    dialogues, human_scores = generate_synthetic_dialogues(n_dialogues=300, seed=42)
    evaluator = LLMJudgeEvaluator()

    metrics = evaluator.evaluate_dataset(dialogues, human_scores)
    assert "accuracy" in metrics
    assert "cohen_kappa" in metrics
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == 5


def test_ci_gate_pass_and_block():
    good_metrics = {"accuracy": 0.85, "cohen_kappa": 0.70}
    bad_metrics = {"accuracy": 0.45, "cohen_kappa": 0.20}

    assert run_ci_gate(good_metrics, min_accuracy=0.60, min_kappa=0.40) == 0
    assert run_ci_gate(bad_metrics, min_accuracy=0.60, min_kappa=0.40) == 1
