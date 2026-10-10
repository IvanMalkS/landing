import importlib

import numpy as np
import torch

rec_mod = importlib.import_module("projects.07_two_tower_seq.recommender")
FAISSIndex = rec_mod.FAISSIndex
GRU4RecLite = rec_mod.GRU4RecLite
TwoTowerModel = rec_mod.TwoTowerModel
generate_synthetic_rec_data = rec_mod.generate_synthetic_rec_data
train_sequential_model = rec_mod.train_sequential_model
train_two_tower_model = rec_mod.train_two_tower_model


def test_two_tower_recall50_better_than_popularity():
    data = generate_synthetic_rec_data(num_users=300, num_items=200, n_interactions=2000, seed=42)
    user_ids, item_ids = data["user_ids"], data["item_ids"]
    num_users, num_items = data["num_users"], data["num_items"]

    # Popularity baseline
    item_counts = np.bincount(item_ids, minlength=num_items)
    pop_top50 = np.argsort(item_counts)[::-1][:50]

    # Two-Tower training
    model = TwoTowerModel(num_users=num_users, num_items=num_items, embed_dim=32)
    train_two_tower_model(model, user_ids, item_ids, epochs=25)

    model.eval()
    with torch.no_grad():
        all_user_ids = torch.arange(num_users, dtype=torch.long)
        all_item_ids = torch.arange(num_items, dtype=torch.long)
        u_embs = model.user_tower(all_user_ids).numpy()
        i_embs = model.item_tower(all_item_ids).numpy()

    faiss_idx = FAISSIndex(i_embs)
    _, faiss_top50 = faiss_idx.search(u_embs, top_k=50)

    # Evaluate Recall@50 on test interactions
    tt_hits = 0
    pop_hits = 0
    test_len = 500
    for u, i in zip(user_ids[:test_len], item_ids[:test_len]):
        if i in faiss_top50[u]:
            tt_hits += 1
        if i in pop_top50:
            pop_hits += 1

    tt_recall = tt_hits / test_len
    pop_recall = pop_hits / test_len

    assert tt_recall > pop_recall, f"TwoTower Recall@50 {tt_recall:.3f} not better than Popularity {pop_recall:.3f}"


def test_sequential_model_hitrate10_better_than_non_sequential():
    data = generate_synthetic_rec_data(num_users=300, num_items=200, seed=42)
    sequences, targets = data["sequences"], data["targets"]
    num_items = data["num_items"]

    # Non-sequential baseline: most frequent item in targets
    most_freq_target = np.bincount(targets).argmax()

    # Train GRU4Rec
    model = GRU4RecLite(num_items=num_items, embed_dim=32, hidden_dim=32)
    train_sequential_model(model, sequences, targets, epochs=30)

    model.eval()
    with torch.no_grad():
        seq_tensor = torch.tensor(sequences, dtype=torch.long)
        logits = model(seq_tensor)
        top10_preds = torch.topk(logits, k=10, dim=1).indices.numpy()

    hits = 0
    for target, pred_top10 in zip(targets, top10_preds):
        if target in pred_top10:
            hits += 1

    seq_hitrate10 = hits / len(targets)
    non_seq_hitrate10 = np.mean(targets == most_freq_target)

    assert seq_hitrate10 > non_seq_hitrate10 + 0.3, f"Sequential HitRate@10 {seq_hitrate10:.3f} not superior to baseline {non_seq_hitrate10:.3f}"


def test_faiss_exact_brute_force_equivalence_small_index():
    np.random.seed(42)
    n_items = 50
    embed_dim = 16
    item_embs = np.random.randn(n_items, embed_dim).astype(np.float32)

    faiss_idx = FAISSIndex(item_embs)
    query_vecs = np.random.randn(5, embed_dim).astype(np.float32)

    # FAISS search
    _, faiss_indices = faiss_idx.search(query_vecs, top_k=5)

    # Brute force search
    norm_item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)
    norm_query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
    sims = np.dot(norm_query_vecs, norm_item_embs.T)
    brute_force_indices = np.argsort(-sims, axis=1)[:, :5]

    np.testing.assert_array_equal(faiss_indices, brute_force_indices)
