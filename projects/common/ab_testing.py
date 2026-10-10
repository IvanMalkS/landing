"""Common utilities for A/B testing, metrics, synthetic data generation, and LLM client."""

import os
from typing import Any, Dict

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower


def calculate_ab_uplift(control_mean: float, treatment_mean: float) -> Dict[str, float]:
    """Calculates absolute and relative uplift."""
    abs_uplift = treatment_mean - control_mean
    rel_uplift = abs_uplift / control_mean if control_mean != 0 else 0.0
    return {
        "absolute_uplift": float(abs_uplift),
        "relative_uplift": float(rel_uplift),
    }


def proportion_ztest(
    count_control: int,
    n_control: int,
    count_treatment: int,
    n_treatment: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Performs a two-sample Z-test for proportions and computes confidence interval for the difference."""
    p_c = count_control / n_control
    p_t = count_treatment / n_treatment
    p_pooled = (count_control + count_treatment) / (n_control + n_treatment)

    se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1 / n_control + 1 / n_treatment))
    z_stat = (p_t - p_c) / se_pooled if se_pooled > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    se_diff = np.sqrt((p_c * (1 - p_c) / n_control) + (p_t * (1 - p_t) / n_treatment))
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_t - p_c
    ci_lower = float(diff - z_crit * se_diff)
    ci_upper = float(diff + z_crit * se_diff)

    return {
        "z_stat": float(z_stat),
        "p_value": p_value,
        "is_significant": p_value < alpha,
        "diff": float(diff),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def calculate_sample_size_mde(
    baseline_p: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Calculates required sample size per group for proportion Z-test based on MDE."""
    p1 = baseline_p
    p2 = baseline_p * (1 + mde_relative)
    p2 = min(max(p2, 0.0), 1.0)

    # Effect size for proportions (Cohen's h)
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    analysis = NormalIndPower()
    sample_size = analysis.solve_power(effect_size=abs(h), alpha=alpha, power=power, ratio=1.0)
    return int(np.ceil(sample_size))


class LLMClient:
    """Mock/Real LLM Client."""

    def __init__(self, mode: str | None = None):
        self.mode = mode or os.getenv("LLM_MODE", "mock")

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        if self.mode == "mock":
            # Deterministic mock responses based on key prompt contents
            prompt_lower = prompt.lower()
            if "eval" in prompt_lower or "judge" in prompt_lower or "оцени" in prompt_lower:
                return "Score: 4\nReason: Good answer"
            if "тариф" in prompt_lower or "интернет" in prompt_lower or "база знаний" in prompt_lower:
                return "Согласно базе знаний [Doc_1], наш тариф включает 30 ГБ интернета и безлимитные звонки."
            return "Спасибо за ваш вопрос. Подробную информацию вы можете найти в разделе справки [Doc_1]."
        else:
            # Placeholder for real LLM API call if configured
            return f"Real LLM response for: {prompt[:30]}"
