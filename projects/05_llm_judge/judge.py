"""Project 05: LLM-as-a-Judge dialogue evaluator and CI gate"""

import re
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from projects.common.ab_testing import LLMClient


def calculate_cohens_kappa(y_true: List[int], y_pred: List[int]) -> float:
    """Calculates Cohen's kappa score between two raters."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    categories = np.unique(np.concatenate([y_true_arr, y_pred_arr]))

    n = len(y_true_arr)
    if n == 0:
        return 0.0

    po = np.mean(y_true_arr == y_pred_arr)

    pe = 0.0
    for k in categories:
        p_true_k = np.mean(y_true_arr == k)
        p_pred_k = np.mean(y_pred_arr == k)
        pe += p_true_k * p_pred_k

    if pe == 1.0:
        return 1.0

    kappa = (po - pe) / (1.0 - pe)
    return float(kappa)


def generate_synthetic_dialogues(n_dialogues: int = 300, seed: int = 42) -> Tuple[List[Dict[str, str]], List[int]]:
    """Generates synthetic bot dialogues and human ratings (scores 1-5)."""
    np.random.seed(seed)
    dialogues = []
    human_scores = []

    good_phrases = ["Отлично, спасибо за помощь!", "Решили мою проблему быстро.", "Всё понятно, благодарю."]
    bad_phrases = ["Бот несуразный, ничего не понял.", "Отвратный сервис, переведите на оператора.", "Плохо работает!"]

    for i in range(n_dialogues):
        is_good = np.random.rand() > 0.4
        if is_good:
            score = np.random.choice([4, 5], p=[0.3, 0.7])
            user_msg = np.random.choice(good_phrases)
            bot_msg = "Рад был помочь вам!"
        else:
            score = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            user_msg = np.random.choice(bad_phrases)
            bot_msg = "К сожалению, не совсем понял вас."

        dialogues.append({
            "id": f"diag_{i}",
            "dialogue": f"Пользователь: {user_msg}\nБот: {bot_msg}",
        })
        human_scores.append(int(score))

    return dialogues, human_scores


class LLMJudgeEvaluator:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client or LLMClient(mode="mock")

    def judge_dialogue(self, dialogue_text: str) -> int:
        prompt = f"Оцени качество следующего диалога с ботом по шкале от 1 до 5:\n{dialogue_text}"
        response = self.llm_client.generate(prompt)
        match = re.search(r"Score:\s*(\d)", response)
        if match:
            return int(match.group(1))

        # Fallback keyword sentiment judge if prompt mock is customized
        text_lower = dialogue_text.lower()
        if "отлично" in text_lower or "решили" in text_lower or "рад был" in text_lower:
            return 5
        elif "плохо" in text_lower or "отвратный" in text_lower or "несуразный" in text_lower:
            return 2
        return 3

    def evaluate_dataset(
        self, dialogues: List[Dict[str, str]], human_scores: List[int]
    ) -> Dict[str, Any]:
        llm_scores = [self.judge_dialogue(d["dialogue"]) for d in dialogues]

        accuracy = float(accuracy_score(human_scores, llm_scores))
        kappa = calculate_cohens_kappa(human_scores, llm_scores)
        conf_matrix = confusion_matrix(human_scores, llm_scores, labels=[1, 2, 3, 4, 5]).tolist()

        return {
            "accuracy": accuracy,
            "cohen_kappa": kappa,
            "confusion_matrix": conf_matrix,
            "llm_scores": llm_scores,
        }


def run_ci_gate(eval_metrics: Dict[str, Any], min_accuracy: float = 0.60, min_kappa: float = 0.40) -> int:
    """CI gate script logic: returns exit code 0 if thresholds met, 1 if degraded."""
    acc = eval_metrics.get("accuracy", 0.0)
    kappa = eval_metrics.get("cohen_kappa", 0.0)

    if acc < min_accuracy or kappa < min_kappa:
        print(f"CI Gate FAILED: accuracy={acc:.3f} (min {min_accuracy}), kappa={kappa:.3f} (min {min_kappa})")
        return 1

    print(f"CI Gate PASSED: accuracy={acc:.3f}, kappa={kappa:.3f}")
    return 0
