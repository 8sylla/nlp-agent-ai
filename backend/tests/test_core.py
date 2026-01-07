import pytest
from httpx import AsyncClient
from app.main import app
from app.core.nlu_engine import nlu_engine
from app.core.mock_services import order_service
from app.core.sentiment_engine import sentiment_engine

# --- TEST 1 : LE CERVEAU NLU (Spacy) ---
def test_nlu_intent_detection():
    # On teste avec une phrase du nouveau dataset
    text = "Je veux retourner ce produit car il est cassé"
    result = nlu_engine.analyze(text)
    
    assert result["intent"] == "RETURN"
    assert result["confidence"] > 0.5

# --- TEST 2 : LE SERVICE TRANSACTIONNEL (Mock ERP) ---
def test_order_service_success():
    # On teste une commande qui existe vraiment dans mock_services.py
    status = order_service.get_order_status("CMD-123")
    
    assert status is not None
    assert "iPhone 15" in status
    assert "Casablanca" in status

def test_order_service_not_found():
    # On teste une commande inexistante
    status = order_service.get_order_status("CMD-99999")
    assert status is None

# --- TEST 3 : L'INTELLIGENCE ÉMOTIONNELLE (Sentiment) ---
def test_sentiment_anger():
    text = "C'est une arnaque, je suis furieux !"
    result = sentiment_engine.analyze(text)
    
    # On vérifie que le score est bas (1 ou 2 étoiles)
    assert result['stars'] <= 2
    assert result['is_negative'] is True

def test_sentiment_happy():
    text = "Merci beaucoup, c'est super !"
    result = sentiment_engine.analyze(text)
    
    assert result['stars'] >= 4
    assert result['is_negative'] is False

# --- TEST 4 : API HEALTHCHECK (Infra) ---
@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "support-agent-ai"}

