.PHONY: help install start stop restart logs ingest-all train-nlu db-update test clean fclean

# --- COMMANDES PRINCIPALES ---

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Construit les images Docker (Force le rebuild)
	docker-compose build --no-cache

start: ## Démarre tout le projet en arrière-plan
	docker-compose up -d
	@echo "==================================================="
	@echo "  APPLICATION EN LIGNE"
	@echo "  Backend API:    http://localhost:8000/docs"
	@echo "  Frontend Chat:  http://localhost:3000"
	@echo "  Dashboard:      http://localhost:3000/admin"
	@echo "🕸️  Neo4j Browser:  http://localhost:7474"
	@echo "==================================================="

stop: ## Arrête les conteneurs
	docker-compose down

restart: stop start ## Redémarre tout proprement

logs: ## Affiche les logs du backend en temps réel
	docker-compose logs -f api

# --- DATA & IA (TRAINING & INGESTION) ---

train-nlu: ## Entraîne le modèle NLU Spacy et recharge l'API
	@echo "🧠 Démarrage de l'entraînement NLU (Classification & Entités)..."
	docker-compose exec api python train_nlu.py
	@echo "🔄 Redémarrage de l'API pour charger le nouveau modèle..."
	docker-compose restart api
	@echo "✅ Entraînement terminé et modèle chargé !"

ingest-all: ## Lance TOUTES les ingestions (Graph + Vector + Arab)
	@echo " Démarrage de l'ingestion complète..."
	@make ingest-graph
	@make ingest-vector
	@echo "✅ Ingestion globale terminée !"

ingest-graph: ## Construit le Graphe Neo4j (GraphRAG)
	@echo " Ingestion GraphRAG (Neo4j + Gemini/Groq)..."
	docker-compose exec api python ingest_graph.py

ingest-vector: ## Construit les Vecteurs Postgres (VectorRAG)
	@echo "📚 Ingestion VectorRAG (FAQ FR & AR)..."
	docker-compose exec api python ingest_data.py
	docker-compose exec api python ingest_arabic.py

# --- MAINTENANCE & TESTS ---

db-update: ## Ajoute la colonne Feedback si absente
	@echo " Mise à jour du schéma DB..."
	docker-compose exec api python -c "from app.core.database import get_db_connection; c=get_db_connection(); cur=c.cursor(); cur.execute('ALTER TABLE conversations ADD COLUMN IF NOT EXISTS feedback INT;'); c.commit(); print('Done.')"

test: ## Lance les tests unitaires (test_core.py)
	@echo "🧪 Lancement des tests..."
	docker-compose exec api pytest tests/test_core.py -v

format: ## Formate le code Python (Black)
	docker-compose exec api pip install black
	docker-compose exec api black .

clean: ## Nettoie les conteneurs (Sans supprimer les Data)
	docker-compose down --remove-orphans

fclean: ## ⚠️ DANGER: Supprime tout (Conteneurs + Volumes Data)
	@echo "⚠️  ATTENTION: Suppression de toutes les données (Neo4j, PG)..."
	docker-compose down -v