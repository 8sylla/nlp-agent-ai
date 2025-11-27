# 🚀 Guide de Démarrage Rapide - Sprint 1

## ⚡ TL;DR - Démarrage en 5 minutes

```bash
# 1. Clone et setup
git clone <votre-repo>
cd nlp-chatbot-project

# 2. Installation complète
make quickstart

# 3. Lancer l'API
make serve

# 4. Tester
make api-test
```

**C'est tout !** 🎉

---

## 📁 Copier les Fichiers

Copiez tous les fichiers fournis dans cette structure :

```
nlp-chatbot-project/
├── Makefile                          # ← Artifact "sprint1_makefile"
├── docker-compose.yml                # ← Artifact "sprint1_setup"
├── README.md                         # ← Artifact "sprint1_readme"
│
├── backend/
│   ├── requirements.txt              # ← Artifact "sprint1_setup"
│   ├── .env                          # ← Copier de .env.example
│   ├── run.py                        # ← Artifact "us13_inference_api"
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # ← Artifact "us13_inference_api"
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py             # ← Artifact "sprint1_setup"
│   │   │   ├── intents.py            # ← Artifact "us11_data_collection"
│   │   │   └── logging_config.py     # ← Artifact "sprint1_setup"
│   │   │
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── intent_classifier.py  # ← Artifact "us12_model_training"
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── nlu_service.py        # ← Artifact "us13_inference_api"
│   │   │   └── training_service.py   # ← Artifact "us12_model_training"
│   │   │
│   │   └── api/
│   │       ├── __init__.py
│   │       └── routes.py             # ← Artifact "us13_inference_api"
│   │
│   ├── data/
│   │   └── create_dataset.py         # ← Artifact "us11_data_collection"
│   │
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py               # ← Artifact "sprint1_tests"
│       ├── test_intent_classifier.py # ← Artifact "sprint1_tests"
│       ├── test_nlu_service.py       # ← Artifact "sprint1_tests"
│       ├── test_api.py               # ← Artifact "sprint1_tests"
│       ├── test_dataset.py           # ← Artifact "sprint1_tests"
│       └── test_integration.py       # ← Artifact "sprint1_tests"
│
└── notebooks/
    └── sprint1_demo.ipynb            # ← Artifact "sprint1_notebook"
```

---

## 🛠️ Workflow Détaillé

### Étape 1 : Installation

```bash
# Créer environnement virtuel et installer dépendances
make install

# Ou manuellement :
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download fr_core_news_lg
```

### Étape 2 : Services Docker

```bash
# Démarrer PostgreSQL, Redis, MLflow
make docker-up

# Vérifier status
make docker-status

# Voir logs
make docker-logs
```

### Étape 3 : Dataset

```bash
# Créer le dataset (seed + augmentation)
make dataset

# Voir statistiques
make dataset-stats

# Visualiser exemples
make dataset-view
```

**Résultat attendu** :
```
✅ Loaded 50 seed examples
🔄 Starting data augmentation...
✅ Generated 550+ augmented examples
📊 Total dataset size: 600+

📊 Dataset split:
  Train: 420 (70%)
  Val:   90 (15%)
  Test:  90 (15%)
```

### Étape 4 : Entraînement

```bash
# Entraîner le modèle (5 epochs)
make train

# Voir résultats
make train-stats

# Ouvrir MLflow UI
make mlflow
```

**Durée** : 15-20 min (CPU) ou 3-5 min (GPU)

**Métriques attendues** :
```
✅ Val Accuracy: >85%
✅ Val F1 Score: >0.85
✅ Test Accuracy: >85%
```

### Étape 5 : Lancer l'API

```bash
# Démarrer FastAPI en mode dev
make serve

# L'API sera sur http://localhost:8000
# Docs sur http://localhost:8000/docs
```

### Étape 6 : Tester

```bash
# Tests manuels de l'API
make api-test

# Tests automatisés
make test

# Tests avec coverage
make test-cov
```

---

## 🧪 Exemples d'Utilisation

### cURL

```bash
# Classification simple
curl -X POST http://localhost:8000/api/nlp/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Où est ma commande ?"}'

# Réponse :
{
  "intent": "track_order",
  "confidence": 0.94,
  "all_intents": {
    "track_order": 0.94,
    "return_request": 0.03,
    "product_inquiry": 0.01,
    "payment_issue": 0.01,
    "greeting": 0.01
  },
  "from_cache": false,
  "processing_time_ms": 45.2
}
```

### Python

```python
import httpx
import asyncio

async def classify(text):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/nlp/classify",
            json={"text": text}
        )
        return response.json()

# Utilisation
result = asyncio.run(classify("Je veux retourner un article"))
print(f"Intent: {result['intent']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### JavaScript / Node.js

```javascript
const axios = require('axios');

async function classify(text) {
  const response = await axios.post('http://localhost:8000/api/nlp/classify', {
    text: text
  });
  return response.data;
}

// Utilisation
classify("Bonjour").then(result => {
  console.log(`Intent: ${result.intent}`);
  console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
});
```

---

## 📊 Vérifier les Résultats

### 1. Dataset

```bash
# Voir statistiques
cat backend/data/processed/metadata.json | python -m json.tool

# Visualisations
open backend/data/processed/train_analysis.png
```

### 2. Modèle

```bash
# Configuration du modèle
cat backend/models/best_model/config.json | python -m json.tool

# Confusion matrix
open backend/models/confusion_matrix.png
```

### 3. MLflow

Ouvrir http://localhost:5000

Vous verrez :
- 📊 Métriques d'entraînement (accuracy, loss, F1)
- 📈 Graphiques d'évolution
- 🔧 Hyperparamètres
- 💾 Modèles sauvegardés

### 4. API

Ouvrir http://localhost:8000/docs

Documentation interactive Swagger :
- 🧪 Tester tous les endpoints
- 📖 Voir schémas de requête/réponse
- 🔍 Explorer les modèles Pydantic

---

## 🎯 Critères de Succès

### US-1.1 - Dataset ✅

- [x] 100+ exemples par intent
- [x] Format JSON standardisé
- [x] Split 70/15/15
- [x] Data augmentation appliquée
- [x] Documentation + visualisations

**Commande de vérification** :
```bash
make dataset-stats
```

### US-1.2 - Modèle ✅

- [x] Accuracy > 85% sur validation
- [x] Support FR et EN
- [x] Temps inférence < 100ms
- [x] Modèle versionné MLflow
- [x] Confusion matrix

**Commande de vérification** :
```bash
make train-stats
make mlflow
```

### US-1.3 - API ✅

- [x] Endpoint POST /api/nlp/classify
- [x] Validation Pydantic
- [x] Cache Redis
- [x] Logging structuré
- [x] Tests unitaires
- [x] Documentation OpenAPI

**Commande de vérification** :
```bash
make api-test
make test
```

---

## 🐛 Problèmes Fréquents

### 1. Module not found

```bash
# Vérifier que venv est activé
which python  # Doit pointer vers venv/bin/python

# Réactiver
source backend/venv/bin/activate
```

### 2. Redis connection refused

```bash
# Vérifier Docker
docker-compose ps

# Redémarrer Redis
docker-compose restart redis
```

### 3. Modèle non trouvé

```bash
# Vérifier que l'entraînement est terminé
ls -la backend/models/best_model/

# Si absent, réentraîner
make train
```

### 4. Port 8000 déjà utilisé

```bash
# Trouver le processus
lsof -i :8000

# Tuer le processus
kill -9 <PID>

# Ou changer le port dans .env
PORT=8001
```

### 5. Dataset trop petit

```python
# Éditer backend/data/create_dataset.py
# Ligne ~90, augmenter le facteur
creator.augment_data(augmentation_factor=15)  # Au lieu de 10
```

---

## 📈 Performances Attendues

### Modèle

| Métrique | Target | Résultat Attendu |
|----------|--------|------------------|
| Accuracy | >85% | ~90% |
| F1 Score | >85% | ~89% |
| Inference | <100ms | ~50ms |

### API

| Métrique | Valeur |
|----------|--------|
| Latence sans cache | ~50ms |
| Latence avec cache | ~5-10ms |
| Throughput | ~20 req/s (CPU) |
| Cache hit rate | ~70% après warmup |

### Dataset

| Split | Taille | Ratio |
|-------|--------|-------|
| Train | ~420 | 70% |
| Val | ~90 | 15% |
| Test | ~90 | 15% |
| **Total** | **~600** | **100%** |

---

## 🔗 URLs Utiles

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | API FastAPI |
| Docs | http://localhost:8000/docs | Documentation Swagger |
| ReDoc | http://localhost:8000/redoc | Documentation alternative |
| MLflow | http://localhost:5000 | Tracking ML |
| Health | http://localhost:8000/health | Health check |
| Stats | http://localhost:8000/api/nlp/stats | Statistiques cache |

---

## 🎓 Prochaines Étapes

### Sprint 2 Preview

1. **NER (Named Entity Recognition)**
   - Extraction ORDER_ID, DATE, MONEY
   - Fine-tuning Spacy

2. **Sentiment Analysis**
   - Classification positif/neutre/négatif
   - Détection d'urgence

3. **Dialog Management**
   - Gestion contexte conversationnel
   - Multi-turn conversations

---

## 📚 Resources

### Documentation
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/docs/transformers)
- [MLflow](https://mlflow.org/)

### Modèles
- [CamemBERT](https://huggingface.co/camembert-base)
- [Spacy French](https://spacy.io/models/fr)

---

## 💡 Commandes Make Utiles

```bash
make help           # Voir toutes les commandes
make status         # Vérifier status complet
make quickstart     # Setup + Dataset + Train (tout en un)
make clean          # Nettoyer fichiers temporaires
make reset          # Reset complet
```

---

## ✅ Checklist Finale

Avant de passer au Sprint 2, vérifiez :

- [ ] Dataset créé (600+ exemples)
- [ ] Modèle entraîné (accuracy >85%)
- [ ] API fonctionne (tests passent)
- [ ] Cache Redis opérationnel
- [ ] MLflow accessible
- [ ] Tests passent (>80% coverage)
- [ ] Documentation à jour

**Commande de vérification complète** :
```bash
make status
make test
```

---

## 🎉 Félicitations !

Vous avez complété le **Sprint 1 - NLU Foundation** !

**Compétences acquises** :
- ✅ Création et augmentation de datasets
- ✅ Fine-tuning de modèles Transformers
- ✅ API REST avec FastAPI
- ✅ Cache multi-niveaux (Redis + Memory)
- ✅ MLOps avec MLflow
- ✅ Tests automatisés

**Story Points complétés** : 42 SP

**Temps estimé** : 2-3 heures (si tout se passe bien)

---

## 🆘 Besoin d'Aide ?

- 📖 Consultez le [README complet](README.md)
- 🐛 Vérifiez [Troubleshooting](#-problèmes-fréquents)
- 💬 Créez une issue sur GitHub
- 📧 Contactez l'équipe

---

**Bon développement ! 🚀**