import pytest
from httpx import AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test endpoint racine"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_health_endpoint():
    """Test health check"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/nlp/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_classify_endpoint():
    """Test classification endpoint"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/nlp/classify",
            json={"text": "Où est ma commande ?", "use_cache": False}
        )

    assert response.status_code == 200
    data = response.json()

    assert "intent" in data
    assert "confidence" in data
    assert "all_intents" in data
    assert "processing_time_ms" in data
    assert data["intent"] == "track_order"


@pytest.mark.asyncio
async def test_classify_invalid_input():
    """Test avec input invalide"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Texte vide
        response = await client.post(
            "/api/nlp/classify",
            json={"text": ""}
        )

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_classify_batch_endpoint():
    """Test batch classification"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/nlp/classify/batch",
            json={
                "texts": [
                    "Où est ma commande ?",
                    "Bonjour",
                    "Je veux retourner un article"
                ]
            }
        )

    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    assert len(data["results"]) == 3
    assert "total_processing_time_ms" in data


@pytest.mark.asyncio
async def test_classify_batch_too_many():
    """Test batch avec trop de textes"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/api/nlp/classify/batch",
            json={"texts": ["text"] * 101}  # Max 100
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_stats_endpoint():
    """Test endpoint de stats"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/api/nlp/stats")

    assert response.status_code == 200
    data = response.json()

    assert "total_predictions" in data
    assert "cache_hit_rate" in data

