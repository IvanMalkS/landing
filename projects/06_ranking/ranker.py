"""Project 06: Search Listing Ranking (Daily Rentals) with LightGBM LambdaRank & IPS"""

from typing import Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score


def generate_synthetic_search_sessions(
    n_queries: int = 200, items_per_query: int = 20, seed: int = 42
) -> pd.DataFrame:
    """Generates synthetic search sessions with position bias."""
    np.random.seed(seed)
    records = []

    for q_id in range(n_queries):
        # Baseline rating (1-5 stars)
        ratings = np.random.uniform(1.0, 5.0, size=items_per_query)
        prices = np.random.uniform(1000, 10000, size=items_per_query)
        distance_center = np.random.uniform(0.5, 15.0, size=items_per_query)

        # Multi-factor hidden relevance utility
        true_relevance = (10000.0 - prices) / 1000.0 + (15.0 - distance_center) / 1.5 + ratings

        # Display positions
        positions = np.arange(1, items_per_query + 1)
        np.random.shuffle(positions)

        # Propensity model: P(Examined | pos) ~ 1 / sqrt(pos)
        propensities = 1.0 / np.sqrt(positions)

        # Relevance grade for click simulation
        rel_grade = pd.qcut(true_relevance, q=5, labels=[0, 1, 2, 3, 4]).astype(int)
        click_prob = (rel_grade / 4.0) * propensities
        clicked = (np.random.rand(items_per_query) < click_prob).astype(int)

        for i in range(items_per_query):
            records.append({
                "query_id": q_id,
                "item_id": i,
                "true_relevance": true_relevance[i],
                "rating": ratings[i],
                "price": prices[i],
                "distance_center": distance_center[i],
                "position": positions[i],
                "propensity": propensities[i],
                "ips_weight": 1.0 / propensities[i],
                "clicked": clicked[i],
            })

    return pd.DataFrame(records)


def group_wise_train_test_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits search sessions into train and test sets by query_id without data leakage."""
    np.random.seed(seed)
    unique_queries = df["query_id"].unique()
    n_test = int(len(unique_queries) * test_size)
    test_queries = set(np.random.choice(unique_queries, size=n_test, replace=False))

    train_df = df[~df["query_id"].isin(test_queries)].copy()
    test_df = df[df["query_id"].isin(test_queries)].copy()
    return train_df, test_df


class LambdaRanker:
    def __init__(self):
        self.model = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=100,
            learning_rate=0.05,
            num_leaves=15,
            random_state=42,
            verbosity=-1,
        )

    def fit(self, train_df: pd.DataFrame, use_ips_weights: bool = True):
        features = ["rating", "price", "distance_center"]
        X = train_df[features]
        y = train_df["clicked"]

        # Sort by query_id for LightGBM ranker
        sorted_indices = train_df["query_id"].argsort()
        X_sorted = X.iloc[sorted_indices]
        y_sorted = y.iloc[sorted_indices]
        query_groups = train_df.groupby("query_id", sort=False).size().values

        sample_weight = train_df["ips_weight"].iloc[sorted_indices] if use_ips_weights else None

        self.model.fit(
            X_sorted,
            y_sorted,
            group=query_groups,
            sample_weight=sample_weight,
        )
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        features = ["rating", "price", "distance_center"]
        return self.model.predict(test_df[features])


def evaluate_ndcg_at_k(df: pd.DataFrame, score_col: str, k: int = 10) -> float:
    """Calculates mean NDCG@k on true_relevance across query groups."""
    ndcg_list = []
    for q_id, group in df.groupby("query_id"):
        if len(group) < 2:
            continue
        y_true = np.array([group["true_relevance"].values])
        y_score = np.array([group[score_col].values])
        score = ndcg_score(y_true, y_score, k=k)
        ndcg_list.append(score)
    return float(np.mean(ndcg_list))
