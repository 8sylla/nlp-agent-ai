.PHONY: help install start stop restart logs test-backend ingest clean

# --- COMMANDES PRINCIPALES ---

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Construit les images Docker
	docker-compose build

start: ## Démarre tout le projet en arrière-plan
	docker-compose up -d
	@echo "🚀 Backend: http://localhost:8000/docs"
	@echo "🎨 Frontend: http://localhost:3000"
	@echo "📊 Admin: http://localhost:3000/admin"

stop: ## Arrête les conteneurs
	docker-compose down

restart: stop start ## Redémarre tout

logs: ## Affiche les logs du backend en temps réel
	docker-compose logs -f api

# --- OUTILS & DATA ---

ingest: ## Recharge les données (FAQ FR + AR)
	@echo "📥 Ingestion des données..."
	docker-compose exec api python ingest_data.py
	docker-compose exec api python ingest_arabic.py
	@echo "✅ Terminé."

test-backend: ## Lance les tests unitaires Python (dans Docker)
	@echo "🧪 Lancement des tests Backend..."
	docker-compose exec api pytest -v

format: ## Formate le code (Black/Ruff) - Optionnel
	docker-compose exec api pip install black
	docker-compose exec api black .

clean: ## Nettoie les fichiers temporaires et conteneurs
	docker-compose down -v
	find . -type d -name "__pycache__" -exec rm -rf {} +