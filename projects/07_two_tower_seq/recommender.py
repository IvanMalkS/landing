"""Project 07: Two-Tower Candidate Retrieval & Sequential Recommendation"""

from typing import Any, Dict, Tuple

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UserTower(nn.Module):
    def __init__(self, num_users: int, embed_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_users, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(user_ids)
        x = F.relu(self.fc(x))
        return F.normalize(x, p=2, dim=1)


class ItemTower(nn.Module):
    def __init__(self, num_items: int, embed_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_items, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(item_ids)
        x = F.relu(self.fc(x))
        return F.normalize(x, p=2, dim=1)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embed_dim: int = 32):
        super().__init__()
        self.user_tower = UserTower(num_users, embed_dim)
        self.item_tower = ItemTower(num_items, embed_dim)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_tower(user_ids)
        i_emb = self.item_tower(item_ids)
        # Cosine similarity matrix (in-batch negatives)
        return torch.matmul(u_emb, i_emb.T)


def train_two_tower_model(
    model: TwoTowerModel,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    epochs: int = 15,
    lr: float = 0.01,
) -> TwoTowerModel:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    user_tensor = torch.tensor(user_ids, dtype=torch.long)
    item_tensor = torch.tensor(item_ids, dtype=torch.long)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(user_tensor, item_tensor)
        labels = torch.arange(len(user_tensor), dtype=torch.long)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    return model


class FAISSIndex:
    def __init__(self, item_embeddings: np.ndarray):
        self.embed_dim = item_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embed_dim)
        # Normalize vectors for cosine similarity via inner product
        faiss.normalize_L2(item_embeddings)
        self.index.add(item_embeddings.astype(np.float32))

    def search(self, query_embeddings: np.ndarray, top_k: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        query_vecs = query_embeddings.copy().astype(np.float32)
        faiss.normalize_L2(query_vecs)
        distances, indices = self.index.search(query_vecs, top_k)
        return distances, indices


class GRU4RecLite(nn.Module):
    """GRU-based sequential recommendation model for guest session interactions."""

    def __init__(self, num_items: int, embed_dim: int = 32, hidden_dim: int = 32):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_items)

    def forward(self, sequence_tensor: torch.Tensor) -> torch.Tensor:
        # sequence_tensor: [batch_size, seq_len]
        embeds = self.item_embedding(sequence_tensor)
        out, h_n = self.gru(embeds)
        # Take last time step hidden state
        last_hidden = h_n[-1]
        logits = self.fc(last_hidden)
        return logits


def train_sequential_model(
    model: GRU4RecLite,
    sequences: np.ndarray,
    targets: np.ndarray,
    epochs: int = 20,
    lr: float = 0.01,
) -> GRU4RecLite:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    seq_tensor = torch.tensor(sequences, dtype=torch.long)
    target_tensor = torch.tensor(targets, dtype=torch.long)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(seq_tensor)
        loss = criterion(logits, target_tensor)
        loss.backward()
        optimizer.step()

    return model


def generate_synthetic_rec_data(
    num_users: int = 500, num_items: int = 200, n_interactions: int = 3000, seed: int = 42
) -> Dict[str, Any]:
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Preferential user-item affinity matrix
    user_pref = np.random.randint(0, 5, size=(num_users, num_items))
    users = []
    items = []
    for _ in range(n_interactions):
        u = np.random.randint(0, num_users)
        i = np.random.choice(num_items, p=user_pref[u] / user_pref[u].sum())
        users.append(u)
        items.append(i)

    # Sequential dataset with deterministic sequential dependency: i -> (i+1) % num_items
    seq_list = []
    target_list = []
    for _ in range(1000):
        start_item = np.random.randint(1, num_items - 5)
        seq = [start_item, start_item + 1, start_item + 2]
        target = start_item + 3
        seq_list.append(seq)
        target_list.append(target)

    return {
        "user_ids": np.array(users),
        "item_ids": np.array(items),
        "sequences": np.array(seq_list),
        "targets": np.array(target_list),
        "num_users": num_users,
        "num_items": num_items,
    }
