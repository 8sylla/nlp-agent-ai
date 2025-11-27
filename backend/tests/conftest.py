import pytest
import asyncio
from pathlib import Path
import sys

# Ajouter le backend au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.nlu_service import NLUService


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def nlu_service():
    """Fixture pour le service NLU"""
    service = NLUService()
    await service.initialize_redis()
    yield service

    # Cleanup
    if service.redis_client:
        await service.redis_client.close()