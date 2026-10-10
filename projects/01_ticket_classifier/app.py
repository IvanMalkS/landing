import importlib

from fastapi import FastAPI
from pydantic import BaseModel

model_mod = importlib.import_module("projects.01_ticket_classifier.model")
TicketClassifierModel = model_mod.TicketClassifierModel
generate_synthetic_tickets = model_mod.generate_synthetic_tickets

app = FastAPI(title="Ticket Classifier Service")

# Global trained model instance
texts, labels = generate_synthetic_tickets(50)
model = TicketClassifierModel(confidence_threshold=0.4)
model.fit(texts, labels)


class TicketRequest(BaseModel):
    text: str


class TicketResponse(BaseModel):
    topic: str
    predicted_topic: str
    confidence: float
    auto_routed: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=TicketResponse)
def classify_ticket(req: TicketRequest):
    res = model.predict_with_confidence([req.text])[0]
    return res
