import importlib

import pytest

rag_mod = importlib.import_module("projects.03_rag_bot.rag")
RAGBot = rag_mod.RAGBot
RAGVectorStore = rag_mod.RAGVectorStore
SYNTHETIC_KNOWLEDGE_BASE = rag_mod.SYNTHETIC_KNOWLEDGE_BASE
generate_synthetic_questions = rag_mod.generate_synthetic_questions


def test_recall_at_5_retrieval():
    store = RAGVectorStore()
    store.index_documents(SYNTHETIC_KNOWLEDGE_BASE)

    questions = generate_synthetic_questions()
    kb_questions = [q for q in questions if q["in_kb"]]

    hits_count = 0
    for q in kb_questions:
        hits = store.search(q["query"], top_k=5)
        retrieved_ids = [h["doc_id"] for h in hits]
        if q["expected_doc"] in retrieved_ids:
            hits_count += 1

    recall = hits_count / len(kb_questions)
    assert recall >= 0.85, f"Recall@5 {recall} is lower than 0.85"


def test_out_of_kb_escalation_and_source_attribution():
    store = RAGVectorStore()
    store.index_documents(SYNTHETIC_KNOWLEDGE_BASE)
    bot = RAGBot(vector_store=store, relevance_threshold=0.25)

    # In-KB query
    res_in = bot.query("Сколько стоит интернет в базовом тарифе?")
    assert res_in["escalated"] is False
    assert "doc_01" in res_in["sources"]
    assert "[doc_01]" in res_in["answer"]

    # Out-of-KB query
    res_out = bot.query("Как приготовить борщ?")
    assert res_out["escalated"] is True
    assert res_out["sources"] == []
    assert "оператора" in res_out["answer"]


def test_deflection_rate_calculation():
    store = RAGVectorStore()
    store.index_documents(SYNTHETIC_KNOWLEDGE_BASE)
    bot = RAGBot(vector_store=store, relevance_threshold=0.25)

    bot.query("Сколько стоит интернет в базовом тарифе?")  # deflection
    bot.query("Как приготовить борщ?")  # escalation

    rate = bot.get_deflection_rate()
    assert pytest.approx(rate, abs=1e-5) == 0.5
