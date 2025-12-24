from pydantic import BaseModel
from typing import List, Dict, Any

class NLURequest(BaseModel):
    text: str
    language: str = "fr"  # Par défaut français pour le Sprint 1

class Entity(BaseModel):
    label: str
    text: str

class NLUResponse(BaseModel):
    intent: str
    confidence: float
    entities: List[Entity]
    sentiment: str = "neutral" # Placeholder pour Sprint 4

class RAGRequest(BaseModel):
    query: str

class RAGResponse(BaseModel):
    answer: str
    sources: List[str]