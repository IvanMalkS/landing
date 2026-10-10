"""Project 04: Next Best Offer Recommendation & Drift Monitoring"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

OFFERS = ["offer_01", "offer_02", "offer_03", "offer_04", "offer_05"]


def generate_synthetic_subscriber_data(
    n_samples: int = 2000, seed: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generates synthetic subscriber feature history and chosen target offer."""
    np.random.seed(seed)
    monthly_fee = np.random.uniform(200, 2000, size=n_samples)
    data_usage_gb = np.random.uniform(1, 100, size=n_samples)
    call_minutes = np.random.uniform(50, 1000, size=n_samples)
    tenure_months = np.random.randint(1, 120, size=n_samples)

    df = pd.DataFrame({
        "monthly_fee": monthly_fee,
        "data_usage_gb": data_usage_gb,
        "call_minutes": call_minutes,
        "tenure_months": tenure_months,
    })

    # Rule-based synthetic preferences for deterministic target generation
    target_idx = np.zeros(n_samples, dtype=int)
    target_idx[data_usage_gb > 50] = 0  # offer_01 high data
    target_idx[(data_usage_gb <= 50) & (call_minutes > 500)] = 1  # offer_02 high minutes
    target_idx[(data_usage_gb <= 50) & (call_minutes <= 500) & (monthly_fee > 1000)] = 2  # offer_03 high fee
    target_idx[(tenure_months > 60) & (monthly_fee <= 1000)] = 3  # offer_04 loyal
    # Random remaining to offer_05
    mask = (target_idx == 0) & (data_usage_gb <= 50)
    target_idx[mask] = 4

    # Add slight noise
    noise_mask = np.random.rand(n_samples) < 0.05
    target_idx[noise_mask] = np.random.randint(0, 5, size=noise_mask.sum())

    targets = pd.Series([OFFERS[idx] for idx in target_idx], name="target_offer")
    return df, targets


class NBORecommender:
    def __init__(self):
        self.model = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=6,
            verbose=0,
            random_seed=42,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict_top_n(self, X: pd.DataFrame, top_n: int = 2) -> List[List[str]]:
        probs = self.model.predict_proba(X)
        classes = self.model.classes_
        results = []
        for prob in probs:
            top_indices = np.argsort(prob)[::-1][:top_n]
            results.append([str(classes[idx]) for idx in top_indices])
        return results


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """Calculates Population Stability Index (PSI) between reference and actual feature distributions."""
    buckets = np.linspace(0, 100, num_buckets + 1)
    quantiles = np.percentile(expected, buckets)
    # Ensure unique bins
    quantiles[0] -= 1e-5
    quantiles[-1] += 1e-5
    for i in range(1, len(quantiles)):
        if quantiles[i] <= quantiles[i - 1]:
            quantiles[i] = quantiles[i - 1] + 1e-5

    expected_counts, _ = np.histogram(expected, bins=quantiles)
    actual_counts, _ = np.histogram(actual, bins=quantiles)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid zero division with eps smoothing
    eps = 1e-4
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)
