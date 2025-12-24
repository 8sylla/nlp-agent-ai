import pytest
from httpx import AsyncClient
from app.main import app
from app.core.nlu_engine import nlu_engine

# 1. Test Unitaire du Moteur NLU
def test_nlu_intent_detection():
    # Cas simple : Suivi de commande
    text = "Où est ma commande CMD-12345 ?"
    result = nlu_engine.analyze(text)
    
    assert result["intent"] == "TRACK_ORDER"
    # Vérifie qu'on extrait bien l'entité
    entities_text = [e["text"] for e in result["entities"]]
    assert "CMD-12345" in entities_text

# 2. Test d'Intégration API (Healthcheck)
@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "support-agent-ai"}

# 3. Test d'Intégration API (Endpoint Analyse)
@pytest.mark.asyncio
async def test_analyze_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        payload = {"text": "Je veux rendre mon colis"}
        response = await ac.post("/v1/analyze", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "intent" in data
    assert data["intent"] == "RETURN"
    