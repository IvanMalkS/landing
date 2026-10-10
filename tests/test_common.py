import numpy as np
import pytest

from projects.common.ab_testing import (
    LLMClient,
    calculate_ab_uplift,
    calculate_sample_size_mde,
    proportion_ztest,
)


def test_ab_uplift_calculation():
    res = calculate_ab_uplift(0.10, 0.12)
    assert pytest.approx(res["absolute_uplift"], abs=1e-5) == 0.02
    assert pytest.approx(res["relative_uplift"], abs=1e-5) == 0.20


def test_proportion_ztest_and_mde():
    # Test MDE sample size calculation
    n_required = calculate_sample_size_mde(baseline_p=0.10, mde_relative=0.20, alpha=0.05, power=0.8)
    assert n_required > 1000

    # Synthetic test with true known effect
    np.random.seed(42)
    control = np.random.binomial(1, 0.10, size=3000)
    treatment = np.random.binomial(1, 0.13, size=3000)

    res = proportion_ztest(
        count_control=int(control.sum()),
        n_control=len(control),
        count_treatment=int(treatment.sum()),
        n_treatment=len(treatment),
        alpha=0.05,
    )
    assert res["is_significant"] is True
    assert res["ci_lower"] < res["diff"] < res["ci_upper"]


def test_aa_simulations():
    """Verify false positive rate on A/A test over 200 simulations <= 8% at alpha=0.05."""
    np.random.seed(42)
    n_simulations = 200
    n_samples = 1000
    baseline_p = 0.10
    alpha = 0.05

    false_positives = 0
    for _ in range(n_simulations):
        c_samples = np.random.binomial(1, baseline_p, size=n_samples)
        t_samples = np.random.binomial(1, baseline_p, size=n_samples)

        res = proportion_ztest(
            count_control=int(c_samples.sum()),
            n_control=n_samples,
            count_treatment=int(t_samples.sum()),
            n_treatment=n_samples,
            alpha=alpha,
        )
        if res["is_significant"]:
            false_positives += 1

    fpr = false_positives / n_simulations
    assert fpr <= 0.08, f"False positive rate {fpr} exceeds 8%"


def test_llm_client_mock():
    client = LLMClient(mode="mock")
    resp = client.generate("Какой тариф выбрать?")
    assert "Doc_1" in resp
