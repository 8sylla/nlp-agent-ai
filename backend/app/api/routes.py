from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .api.routes import router as nlu_router
from .services.nlu_service import NLUService
from .core.config import settings
from .core.logging_config import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events"""
    # Startup
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Initialiser NLU service
    nlu_service = NLUService()
    await nlu_service.initialize_redis()

    logger.info("✅ Application started successfully")

    yield

    # Shutdown
    logger.info("🛑 Shutting down application")

    # Cleanup (fermer connexions, etc.)
    if nlu_service.redis_client:
        await nlu_service.redis_client.close()


# Créer app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Service NLU pour classification d'intention avec cache Redis",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En prod: spécifier domaines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure routes
app.include_router(nlu_router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }

