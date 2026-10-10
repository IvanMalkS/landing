import importlib

rank_mod = importlib.import_module("projects.06_ranking.ranker")
LambdaRanker = rank_mod.LambdaRanker
evaluate_ndcg_at_k = rank_mod.evaluate_ndcg_at_k
generate_synthetic_search_sessions = rank_mod.generate_synthetic_search_sessions
group_wise_train_test_split = rank_mod.group_wise_train_test_split


def test_no_query_leakage_in_split():
    df = generate_synthetic_search_sessions(n_queries=100, items_per_query=20, seed=42)
    train_df, test_df = group_wise_train_test_split(df, test_size=0.2, seed=42)

    train_queries = set(train_df["query_id"].unique())
    test_queries = set(test_df["query_id"].unique())

    # Ensure empty intersection
    assert train_queries.intersection(test_queries) == set()


def test_lambdarank_ndcg10_uplift_over_rating_baseline_and_ips():
    df = generate_synthetic_search_sessions(n_queries=150, items_per_query=20, seed=42)
    train_df, test_df = group_wise_train_test_split(df, test_size=0.2, seed=42)

    # Train model with IPS weights
    ranker = LambdaRanker()
    ranker.fit(train_df, use_ips_weights=True)

    test_df["model_score"] = ranker.predict(test_df)
    test_df["baseline_score"] = test_df["rating"]

    model_ndcg = evaluate_ndcg_at_k(test_df, score_col="model_score", k=10)
    baseline_ndcg = evaluate_ndcg_at_k(test_df, score_col="baseline_score", k=10)

    relative_uplift = (model_ndcg - baseline_ndcg) / baseline_ndcg
    assert relative_uplift >= 0.10, f"NDCG@10 relative uplift {relative_uplift:.3f} is less than 10%"


def test_ips_weights_are_applied():
    df = generate_synthetic_search_sessions(n_queries=50, items_per_query=10, seed=42)
    assert "ips_weight" in df.columns
    assert (df["ips_weight"] > 0).all()

    # Confirm ranker fits without error when using IPS sample weights
    ranker = LambdaRanker()
    ranker.fit(df, use_ips_weights=True)
    preds = ranker.predict(df)
    assert len(preds) == len(df)
