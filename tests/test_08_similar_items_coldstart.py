import importlib

sim_mod = importlib.import_module("projects.08_similar_items_coldstart.recommender")
ALSModel = sim_mod.ALSModel
ColdStartContentRecommender = sim_mod.ColdStartContentRecommender
generate_synthetic_interactions = sim_mod.generate_synthetic_interactions
generate_synthetic_items = sim_mod.generate_synthetic_items


def test_als_similar_items_cluster_share():
    items = generate_synthetic_items(n_items=100, seed=42)
    matrix = generate_synthetic_interactions(items, n_users=200, n_interactions=3000, seed=42)

    als = ALSModel(factors=32, iterations=25)
    als.fit(matrix)

    # Check top 5 similar items for warm items across the dataset
    same_cluster_count = 0
    total_evaluated = 0

    for target_item in items[:30]:
        target_cat = target_item["category"]
        similar_ids = als.get_similar_items(target_item["item_id"], N=5)

        for sim_id in similar_ids:
            if items[sim_id]["category"] == target_cat:
                same_cluster_count += 1
            total_evaluated += 1

    same_cluster_share = same_cluster_count / total_evaluated
    assert same_cluster_share >= 0.70, f"Same cluster share {same_cluster_share:.3f} is less than 0.70"


def test_cold_start_item_returns_nonempty_relevant_recommendations():
    items = generate_synthetic_items(n_items=100, seed=42)
    content_rec = ColdStartContentRecommender()
    content_rec.fit(items)

    new_cold_item_desc = "Новое объявление: роскошная вилла с бассейном за городом"
    recs = content_rec.get_similar_items_for_cold_item(new_cold_item_desc, N=5)

    assert len(recs) == 5
    top_rec = recs[0]
    assert top_rec["category"] == "luxury_villa"
    assert top_rec["similarity"] > 0.3
