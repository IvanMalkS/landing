"""Project 08: Similar Items Recommendation with Cold Start Handling"""

from typing import Any, Dict, List

import implicit
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CATEGORIES = ["studio", "1_room", "2_room", "luxury_villa", "cozy_cottage"]

CATEGORY_KEYWORDS = {
    "studio": "компактная студия в центре с современным евроремонтом",
    "1_room": "однокомнатная квартира с балконом и бытовой техникой",
    "2_room": "двухкомнатная просторная квартира для всей семьи",
    "luxury_villa": "роскошная вилла с бассейном сауной и панорамным видом",
    "cozy_cottage": "уютный коттедж за городом камин мангал природа",
}


def generate_synthetic_items(n_items: int = 100, seed: int = 42) -> List[Dict[str, Any]]:
    """Generates synthetic item catalog with descriptions and category clusters."""
    np.random.seed(seed)
    items = []
    for item_id in range(n_items):
        cat = CATEGORIES[item_id % len(CATEGORIES)]
        kw = CATEGORY_KEYWORDS[cat]
        desc = f"Объявление {item_id}: {kw}. Удобное расположение."
        items.append({
            "item_id": item_id,
            "category": cat,
            "description": desc,
        })
    return items


def generate_synthetic_interactions(
    items: List[Dict[str, Any]], n_users: int = 200, n_interactions: int = 4000, seed: int = 42
) -> sp.csr_matrix:
    """Generates sparse user-item interaction matrix with category cluster preference."""
    np.random.seed(seed)
    n_items = len(items)
    rows = []
    cols = []
    data = []

    # Assign category affinity to users
    for u in range(n_users):
        fav_cat = CATEGORIES[u % len(CATEGORIES)]
        cat_items = [it["item_id"] for it in items if it["category"] == fav_cat]
        other_items = [it["item_id"] for it in items if it["category"] != fav_cat]

        # 80% interactions within favorite category cluster
        n_u = np.random.randint(10, 30)
        for _ in range(n_u):
            if np.random.rand() < 0.8:
                item_id = np.random.choice(cat_items)
            else:
                item_id = np.random.choice(other_items)
            rows.append(u)
            cols.append(item_id)
            data.append(1.0)

    matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    return matrix


class ALSModel:
    """Collaborative filtering using implicit ALS."""

    def __init__(self, factors: int = 32, regularization: float = 0.01, iterations: int = 20):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=42,
        )

    def fit(self, user_item_matrix: sp.csr_matrix):
        # implicit expects user-item matrix
        self.model.fit(user_item_matrix)

    def get_similar_items(self, item_id: int, N: int = 10) -> List[int]:
        ids, _ = self.model.similar_items(item_id, N=N + 1)
        # Filter out self
        similar = [int(i) for i in ids if i != item_id]
        return similar[:N]


class ColdStartContentRecommender:
    """Content-based recommender using TF-IDF for new cold items without interaction history."""

    def __init__(self, vector_size: int = 128):
        self.vectorizer = TfidfVectorizer(max_features=vector_size)
        self.item_vectors: np.ndarray | None = None
        self.items: List[Dict[str, Any]] = []

    def fit(self, items: List[Dict[str, Any]]):
        self.items = items
        texts = [it["description"] for it in items]
        self.item_vectors = self.vectorizer.fit_transform(texts).toarray()

    def get_similar_items_for_cold_item(
        self, new_item_description: str, N: int = 10
    ) -> List[Dict[str, Any]]:
        if self.item_vectors is None:
            return []
        new_vec = self.vectorizer.transform([new_item_description]).toarray()
        sims = cosine_similarity(new_vec, self.item_vectors)[0]
        top_indices = np.argsort(sims)[::-1][:N]

        results = []
        for idx in top_indices:
            results.append({
                "item_id": self.items[idx]["item_id"],
                "category": self.items[idx]["category"],
                "description": self.items[idx]["description"],
                "similarity": float(sims[idx]),
            })
        return results
