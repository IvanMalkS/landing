import importlib

from fastapi import FastAPI
from pydantic import BaseModel

rag_mod = importlib.import_module("projects.03_rag_bot.rag")
RAGBot = rag_mod.RAGBot
RAGVectorStore = rag_mod.RAGVectorStore
SYNTHETIC_KNOWLEDGE_BASE = rag_mod.SYNTHETIC_KNOWLEDGE_BASE

app = FastAPI(title="RAG Bot Service")

store = RAGVectorStore()
store.index_documents(SYNTHETIC_KNOWLEDGE_BASE)
bot = RAGBot(vector_store=store, relevance_threshold=0.25)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    escalated: bool
    sources: list[str]
    top_score: float


@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest):
    return bot.query(req.query)
