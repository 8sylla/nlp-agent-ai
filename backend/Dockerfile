FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# --- OPTIMISATION CACHE ---
# 1. On copie et installe les grosses librairies (ML) d'abord
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

# 2. On installe le modèle Spacy (Lourd aussi)
RUN pip install https://github.com/explosion/spacy-models/releases/download/fr_core_news_lg-3.7.0/fr_core_news_lg-3.7.0.tar.gz

# 3. Ensuite seulement, on copie les requirements légers (Web)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# --------------------------

COPY . .

EXPOSE 8000
