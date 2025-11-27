# ============================================
# Makefile - Sprint 1 NLU Foundation
# ============================================

.PHONY: help setup install clean dataset train serve test docker-up docker-down mlflow logs

# Variables
PYTHON := python
PIP := pip
VENV := venv
BACKEND := backend
DATA_DIR := $(BACKEND)/data
MODELS_DIR := $(BACKEND)/models

# Couleurs pour output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# ============================================
# Aide
# ============================================

help: ## Afficher cette aide
	@echo "$(BLUE)╔══════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║    Sprint 1 - NLU Foundation Setup      ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)Commandes disponibles:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Workflow complet:$(NC)"
	@echo "  1. $(YELLOW)make setup$(NC)      - Setup initial"
	@echo "  2. $(YELLOW)make docker-up$(NC)  - Démarrer services"
	@echo "  3. $(YELLOW)make dataset$(NC)    - Créer dataset"
	@echo "  4. $(YELLOW)make train$(NC)      - Entraîner modèle"
	@echo "  5. $(YELLOW)make serve$(NC)      - Lancer API"
	@echo "  6. $(YELLOW)make test$(NC)       - Tests"

# ============================================
# Setup & Installation
# ============================================

setup: ## Setup complet du projet
	@echo "$(GREEN)🚀 Setup du projet...$(NC)"
	@make install
	@make docker-up
	@echo "$(GREEN)✅ Setup terminé!$(NC)"

install: ## Installer dépendances Python
	@echo "$(GREEN)📦 Installation des dépendances...$(NC)"
	cd $(BACKEND) && \
		$(PYTHON) -m venv $(VENV) && \
		. $(VENV)/bin/activate && \
		$(PIP) install --upgrade pip && \
		$(PIP) install -r requirements.txt && \
		$(PYTHON) -m spacy download fr_core_news_lg
	@echo "$(GREEN)✅ Dépendances installées$(NC)"

install-dev: ## Installer dépendances de dev
	@echo "$(GREEN)📦 Installation dépendances dev...$(NC)"
	cd $(BACKEND) && \
		. $(VENV)/bin/activate && \
		$(PIP) install black isort flake8 mypy pytest pytest-cov pytest-asyncio
	@echo "$(GREEN)✅ Dépendances dev installées$(NC)"

# ============================================
# Docker Services
# ============================================

docker-up: ## Démarrer services Docker (PostgreSQL, Redis, MLflow)
	@echo "$(GREEN)🐳 Démarrage des services Docker...$(NC)"
	docker-compose up -d
	@echo "$(YELLOW)⏳ Attente démarrage services (10s)...$(NC)"
	@sleep 10
	@make docker-status
	@echo "$(GREEN)✅ Services démarrés$(NC)"

docker-down: ## Arrêter services Docker
	@echo "$(YELLOW)🛑 Arrêt des services Docker...$(NC)"
	docker-compose down
	@echo "$(GREEN)✅ Services arrêtés$(NC)"

docker-restart: ## Redémarrer services Docker
	@make docker-down
	@make docker-up

docker-status: ## Vérifier status services Docker
	@echo "$(BLUE)📊 Status des services:$(NC)"
	@docker-compose ps

docker-logs: ## Voir logs Docker
	docker-compose logs -f

docker-clean: ## Nettoyer containers et volumes Docker
	@echo "$(RED)⚠️  Nettoyage complet Docker (volumes supprimés)$(NC)"
	@read -p "Continuer? (y/N) " confirm && [ "$$confirm" = "y" ] || exit 1
	docker-compose down -v
	@echo "$(GREEN)✅ Nettoyage terminé$(NC)"

# ============================================
# Dataset
# ============================================

dataset: ## Créer le dataset avec augmentation
	@echo "$(GREEN)📊 Création du dataset...$(NC)"
	cd $(BACKEND) && \
		. $(VENV)/bin/activate && \
		$(PYTHON) data/create_dataset.py
	@echo "$(GREEN)✅ Dataset créé: $(DATA_DIR)/processed/$(NC)"
	@make dataset-stats

dataset-stats: ## Afficher stats du dataset
	@echo "$(BLUE)📈 Statistiques du dataset:$(NC)"
	@cd $(BACKEND) && cat data/processed/metadata.json | $(PYTHON) -m json.tool

dataset-view: ## Voir quelques exemples du dataset
	@echo "$(BLUE)👀 Exemples du dataset:$(NC)"
	@cd $(BACKEND) && head -n 5 data/processed/train.json

dataset-clean: ## Supprimer le dataset
	@echo "$(YELLOW)🗑️  Suppression du dataset...$(NC)"
	rm -rf $(DATA_DIR)/processed
	@echo "$(GREEN)✅ Dataset supprimé$(NC)"

# ============================================
# Training
# ============================================

train: ## Entraîner le modèle
	@echo "$(GREEN)🎓 Entraînement du modèle...$(NC)"
	@if [ ! -d "$(DATA_DIR)/processed" ]; then \
		echo "$(RED)❌ Dataset non trouvé. Lancez 'make dataset' d'abord$(NC)"; \
		exit 1; \
	fi
	cd $(BACKEND) && \
		. $(VENV)/bin/activate && \
		$(PYTHON) -m app.services.training_service
	@echo "$(GREEN)✅ Entraînement terminé$(NC)"
	@make train-stats

train-stats: ## Afficher stats du modèle entraîné
	@echo "$(BLUE)📊 Modèle entraîné:$(NC)"
	@if [ -f "$(MODELS_DIR)/best_model/config.json" ]; then \
		cat $(MODELS_DIR)/best_model/config.json | $(PYTHON) -m json.tool; \
	else \
		echo "$(RED)❌ Aucun modèle trouvé$(NC)"; \
	fi

train-clean: ## Supprimer modèles entraînés
	@echo "$(YELLOW)🗑️  Suppression des modèles...$(NC)"
	rm -rf $(MODELS_DIR)/best_model
	rm -rf $(MODELS_DIR)/confusion_matrix.png
	@echo "$(GREEN)✅ Modèles supprimés$(NC)"

# ============================================
# Serving
# ============================================

serve: ## Lancer l'API FastAPI
	@echo "$(GREEN)🚀 Démarrage de l'API...$(NC)"
	@if [ ! -d "$(MODELS_DIR)/best_model" ]; then \
		echo "$(RED)❌ Modèle non trouvé. Lancez 'make train' d'abord$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)📡 API disponible sur: http://localhost:8000$(NC)"
	@echo "$(BLUE)📚 Docs: http://localhost:8000/docs$(NC)"
	cd $(BACKEND) && \
		. $(VENV)/bin/activate && \
		$(PYTHON) run.py

serve-prod: ## Lancer en mode production (Gunicorn)
	@echo "$(GREEN)🚀 Démarrage en mode production...$(NC)"
	cd $(BACKEND) && \
		. $(VENV)/bin/activate && \
		gunicorn app.main:app \
			--workers 4 \
			--worker-class uvicorn.workers.UvicornWorker \
			--bind 0.0.0.0:8000 \
			--log-level info

# ============================================
# Testing
# ============================================

test: ## Lancer tous les tests
	@echo "$(GREEN)🧪 Lancement des tests...$(NC)"
	cd $(BACKEND) && \
		. $(VENV)/bin/activate && \
		pytest tests/ -v

test-unit: ## Tests unitaires seulement
	@echo "$(GREEN)🧪 Tests unitaires...$(NC)"
	cd $(BACKEND) && \
		. $(VENV)/bin/activate && \
		pytest tests/test_intent_classifier.py tests/test_nlu_service.py -v

test-api: ## Tests API seulement
	@echo "$(GREEN)🧪 Tests API...$(NC)"
	cd $(BACKEND) && \
		. $(VENV)/bin/activate && \
		pytest tests/test_api.py -v

test-integration: ## Tests d'intégration
	@echo "$(GREEN)🧪 Tests d'intégration...$(NC)"
	cd $(BACKEND) && \
		. $(VENV)/bin/activate && \
		pytest tests/test_integration.py -v

test-cov: ## Tests avec coverage
	@echo "$(GREEN)🧪 Tests avec coverage...$(NC)"
	cd $(BACKEND) && \
		. $(VENV)/bin/activate && \
		pytest tests/ --cov=app --cov-report=html --cov-report=term
	@echo "$(BLUE)📊 Rapport: $(BACKEND)/htmlcov/index.html$(NC)"

test-watch: ## Tests en mode watch (reruns on changes)
	@echo "$(GREEN)🧪 Tests en mode watch...$(NC)"
	cd $(BACKEND) && \
		. $(VENV)/bin/activate && \
		pytest-watch tests/ -v

# ============================================
# MLflow
# ============================================

mlflow: ## Ouvrir MLflow UI
	@echo "$(BLUE)📊 MLflow UI: http://localhost:5000$(NC)"
	@open http://localhost:5000 || xdg-open http://localhost:5000 || start http://localhost:5000

mlflow-clean: ## Nettoyer expériences MLflow
	@echo "$(YELLOW)🗑️  Nettoyage MLflow...$(NC)"
	docker-compose exec mlflow rm -rf /mlflow/*
	@echo "$(GREEN)✅ MLflow nettoyé$(NC)"

# ============================================
# Code Quality
# ============================================

format: ## Formater le code (Black + isort)
	@echo "$(GREEN)✨ Formatage du code...$(NC)"
	cd $(BACKEND) && \
		. $(VENV)/bin/activate && \
		black app/ tests/ && \
		isort app/ tests/
	@echo "$(GREEN)✅ Code formaté$(NC)"

lint: ## Linter (flake8 + mypy)
	@echo "$(GREEN)🔍 Linting...$(NC)"
	cd $(BACKEND) && \
		. $(VENV)/bin/activate && \
		flake8 app/ tests/ --max-line-length=120 && \
		mypy app/ --ignore-missing-imports
	@echo "$(GREEN)✅ Linting OK$(NC)"

check: format lint test ## Format + Lint + Test

# ============================================
# API Testing
# ============================================

api-test: ## Tester l'API manuellement
	@echo "$(GREEN)🔧 Tests API manuels...$(NC)"
	@echo ""
	@echo "$(BLUE)1. Health check:$(NC)"
	@curl -s http://localhost:8000/health | $(PYTHON) -m json.tool
	@echo ""
	@echo "$(BLUE)2. Classification:$(NC)"
	@curl -s -X POST http://localhost:8000/api/nlp/classify \
		-H "Content-Type: application/json" \
		-d '{"text": "Où est ma commande ?"}' | $(PYTHON) -m json.tool
	@echo ""
	@echo "$(BLUE)3. Stats:$(NC)"
	@curl -s http://localhost:8000/api/nlp/stats | $(PYTHON) -m json.tool

api-benchmark: ## Benchmark de l'API
	@echo "$(GREEN)⚡ Benchmark de l'API...$(NC)"
	cd $(BACKEND) && \
		. $(VENV)/bin/activate && \
		$(PYTHON) -c "import asyncio; \
			from httpx import AsyncClient; \
			import time; \
			async def bench(): \
				async with AsyncClient(base_url='http://localhost:8000') as client: \
					start = time.time(); \
					tasks = [client.post('/api/nlp/classify', json={'text': 'test'}) for _ in range(100)]; \
					await asyncio.gather(*tasks); \
					print(f'100 requêtes en {time.time()-start:.2f}s'); \
			asyncio.run(bench())"

# ============================================
# Logs & Monitoring
# ============================================

logs: ## Voir logs backend
	@tail -f $(BACKEND)/logs/*.log 2>/dev/null || echo "Pas de logs"

logs-redis: ## Logs Redis
	docker-compose logs -f redis

logs-postgres: ## Logs PostgreSQL
	docker-compose logs -f postgres

logs-mlflow: ## Logs MLflow
	docker-compose logs -f mlflow

# ============================================
# Utilities
# ============================================

notebook: ## Lancer Jupyter notebook
	@echo "$(GREEN)📓 Lancement Jupyter...$(NC)"
	cd notebooks && \
		. ../$(BACKEND)/$(VENV)/bin/activate && \
		jupyter notebook

shell: ## Shell Python avec env activé
	@echo "$(GREEN)🐚 Python shell...$(NC)"
	cd $(BACKEND) && \
		. $(VENV)/bin/activate && \
		$(PYTHON)

redis-cli: ## Accéder à Redis CLI
	@echo "$(GREEN)🔴 Redis CLI...$(NC)"
	docker exec -it chatbot_redis redis-cli

psql: ## Accéder à PostgreSQL
	@echo "$(GREEN)🐘 PostgreSQL CLI...$(NC)"
	docker exec -it chatbot_postgres psql -U postgres -d chatbot_db

# ============================================
# Clean & Reset
# ============================================

clean: ## Nettoyer fichiers temporaires
	@echo "$(YELLOW)🧹 Nettoyage...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf $(BACKEND)/htmlcov
	@echo "$(GREEN)✅ Nettoyage terminé$(NC)"

reset: clean dataset-clean train-clean docker-clean ## Reset complet du projet
	@echo "$(RED)⚠️  Reset complet effectué$(NC)"

# ============================================
# Status & Info
# ============================================

status: ## Vérifier status complet
	@echo "$(BLUE)╔══════════════════════════════════════════╗$(NC)"
	@echo "$(BLUE)║          Status du Projet Sprint 1      ║$(NC)"
	@echo "$(BLUE)╚══════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(GREEN)📦 Services Docker:$(NC)"
	@make docker-status
	@echo ""
	@echo "$(GREEN)📊 Dataset:$(NC)"
	@if [ -d "$(DATA_DIR)/processed" ]; then \
		echo "  ✅ Dataset présent"; \
		cat $(DATA_DIR)/processed/metadata.json | $(PYTHON) -c "import sys,json; d=json.load(sys.stdin); print(f'  📈 Train: {d[\"train_size\"]}, Val: {d[\"val_size\"]}, Test: {d[\"test_size\"]}')"; \
	else \
		echo "  ❌ Dataset absent (lancez 'make dataset')"; \
	fi
	@echo ""
	@echo "$(GREEN)🎓 Modèle:$(NC)"
	@if [ -d "$(MODELS_DIR)/best_model" ]; then \
		echo "  ✅ Modèle entraîné présent"; \
	else \
		echo "  ❌ Modèle absent (lancez 'make train')"; \
	fi
	@echo ""
	@echo "$(GREEN)🔗 URLs utiles:$(NC)"
	@echo "  📡 API: http://localhost:8000"
	@echo "  📚 Docs: http://localhost:8000/docs"
	@echo "  📊 MLflow: http://localhost:5000"

version: ## Afficher versions
	@echo "$(BLUE)📌 Versions:$(NC)"
	@$(PYTHON) --version
	@docker --version
	@docker-compose --version

# ============================================
# Quick Start
# ============================================

quickstart: setup dataset train ## Setup + Dataset + Train (tout en un)
	@echo ""
	@echo "$(GREEN)╔══════════════════════════════════════════╗$(NC)"
	@echo "$(GREEN)║   ✅ Quickstart terminé avec succès!     ║$(NC)"
	@echo "$(GREEN)╚══════════════════════════════════════════╝$(NC)"
	@echo ""
	@echo "$(YELLOW)Prochaines étapes:$(NC)"
	@echo "  1. $(BLUE)make serve$(NC)     - Lancer l'API"
	@echo "  2. $(BLUE)make api-test$(NC)  - Tester l'API"
	@echo "  3. $(BLUE)make test$(NC)      - Lancer les tests"
	@echo "  4. $(BLUE)make mlflow$(NC)    - Voir les métriques"

# Default target
.DEFAULT_GOAL := help