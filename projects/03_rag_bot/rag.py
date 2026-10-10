"""Project 03: RAG Support Bot with Qdrant and LLMClient"""

import uuid
from typing import Any, Dict, List

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sklearn.feature_extraction.text import TfidfVectorizer

from projects.common.ab_testing import LLMClient

SYNTHETIC_KNOWLEDGE_BASE = [
    {
        "id": "doc_01",
        "title": "Тарифы и интернет",
        "text": "Наш базовый тариф включает 30 ГБ мобильного интернета и безлимитные мессенджеры за 500 рублей в месяц.",
    },
    {
        "id": "doc_02",
        "title": "Роуминг за рубежом",
        "text": "В международном роуминге доступна опция За границей как дома за 350 рублей в день.",
    },
    {
        "id": "doc_03",
        "title": "Замена SIM и eSIM",
        "text": "Оформить eSIM или заменить пластиковую SIM-карту можно бесплатно в любом офисе продаж или через личный кабинет.",
    },
    {
        "id": "doc_04",
        "title": "Оплата и баланс",
        "text": "Пополнить баланс без комиссии можно на сайте, в приложении банком или настроить автоплатеж.",
    },
    {
        "id": "doc_05",
        "title": "Домашний интернет и ТВ",
        "text": "Подключение домашнего интернета на скорости до 500 Мбит/с и ТВ до 200 каналов стоит от 600 рублей.",
    },
]


def generate_synthetic_questions() -> List[Dict[str, Any]]:
    """Generates synthetic test queries, expected doc IDs, and out-of-kb indicators."""
    return [
        {"query": "Сколько стоит гигабайт интернета в базовом тарифе?", "expected_doc": "doc_01", "in_kb": True},
        {"query": "Как пополнить счет без комиссии?", "expected_doc": "doc_04", "in_kb": True},
        {"query": "Сколько стоит роуминг за границей?", "expected_doc": "doc_02", "in_kb": True},
        {"query": "Как перейти на eSIM?", "expected_doc": "doc_03", "in_kb": True},
        {"query": "Какая скорость у домашнего интернета?", "expected_doc": "doc_05", "in_kb": True},
        {"query": "Где купить квантовый компьютинг и межгалактический корабль?", "expected_doc": None, "in_kb": False},
        {"query": "Рецепт борща с чесноком", "expected_doc": None, "in_kb": False},
    ]


class RAGVectorStore:
    def __init__(self, vector_size: int = 256):
        self.vector_size = vector_size
        self.client = QdrantClient(location=":memory:")
        self.collection_name = "kb_documents"
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=self.vector_size,
        )

    def _pad_vector(self, vec: np.ndarray) -> List[float]:
        if len(vec) < self.vector_size:
            padded = np.pad(vec, (0, self.vector_size - len(vec)))
        else:
            padded = vec[: self.vector_size]
        norm = np.linalg.norm(padded)
        if norm > 0:
            padded = padded / norm
        return padded.tolist()

    def index_documents(self, docs: List[Dict[str, str]]):
        texts = [f"{d['title']} {d['text']}" for d in docs]
        matrix = self.vectorizer.fit_transform(texts).toarray()

        points = []
        for i, doc in enumerate(docs):
            vec = self._pad_vector(matrix[i])
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={"doc_id": doc["id"], "title": doc["title"], "text": doc["text"]},
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        vec_raw = self.vectorizer.transform([query]).toarray()[0]
        if np.linalg.norm(vec_raw) == 0:
            return []
        vec = self._pad_vector(vec_raw)
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=vec,
            limit=top_k,
        )
        hits = []
        for r in res.points:
            hits.append({
                "doc_id": r.payload["doc_id"],
                "title": r.payload["title"],
                "text": r.payload["text"],
                "score": float(r.score),
            })
        return hits


class RAGBot:
    def __init__(
        self,
        vector_store: RAGVectorStore,
        llm_client: LLMClient | None = None,
        relevance_threshold: float = 0.25,
    ):
        self.vector_store = vector_store
        self.llm_client = llm_client or LLMClient(mode="mock")
        self.relevance_threshold = relevance_threshold
        self.total_queries = 0
        self.deflected_queries = 0
        self.escalated_queries = 0

    def query(self, user_query: str) -> Dict[str, Any]:
        self.total_queries += 1
        hits = self.vector_store.search(user_query, top_k=5)

        if not hits or hits[0]["score"] < self.relevance_threshold:
            self.escalated_queries += 1
            return {
                "answer": "К сожалению, в базе знаний нет ответа на ваш вопрос. Переводим разговор на оператора.",
                "escalated": True,
                "sources": [],
                "top_score": hits[0]["score"] if hits else 0.0,
            }

        top_doc = hits[0]
        self.deflected_queries += 1
        prompt = f"Контекст: {top_doc['text']}\nВопрос: {user_query}"
        answer = self.llm_client.generate(prompt)

        source_ref = f"[{top_doc['doc_id']}]"
        if source_ref not in answer:
            answer = f"{answer} Источник: {source_ref}"

        return {
            "answer": answer,
            "escalated": False,
            "sources": [top_doc["doc_id"]],
            "top_score": top_doc["score"],
        }

    def get_deflection_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.deflected_queries / self.total_queries
