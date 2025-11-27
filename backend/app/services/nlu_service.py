import torch
import json
from pathlib import Path
from typing import Dict, List, Optional
import hashlib
import asyncio
from functools import lru_cache

import redis.asyncio as redis
from transformers import CamembertTokenizer

from ..models.intent_classifier import IntentClassifier
from ..core.config import settings
from ..core.logging_config import logger


class NLUService:
    """Service d'inférence NLU avec cache multi-niveaux"""

    _instance = None

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info("🔧 Initializing NLU Service...")

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # Charger modèle
        self.model_path = Path(settings.MODEL_CACHE_DIR) / "best_model"
        self._load_model()

        # Redis client (sera initialisé de manière asynchrone)
        self.redis_client: Optional[redis.Redis] = None

        # Cache en mémoire (LRU)
        self.memory_cache = {}
        self.max_cache_size = 1000

        # Statistiques
        self.stats = {
            'total_predictions': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        self._initialized = True
        logger.info("✅ NLU Service initialized")

    def _load_model(self):
        """Charger le modèle entraîné"""

        # Charger config
        with open(self.model_path / "config.json") as f:
            config = json.load(f)

        self.num_labels = config['num_labels']
        self.label2id = config['label2id']
        self.id2label = config['id2label']

        # Charger tokenizer
        self.tokenizer = CamembertTokenizer.from_pretrained(str(self.model_path))

        # Charger modèle
        self.model = IntentClassifier(num_labels=self.num_labels)
        self.model.load_state_dict(
            torch.load(self.model_path / "model.pt", map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"✅ Model loaded from {self.model_path}")
        logger.info(f"📊 Number of intents: {self.num_labels}")

    async def initialize_redis(self):
        """Initialiser connexion Redis (async)"""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("✅ Redis connected")
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}. Using memory cache only.")
            self.redis_client = None

    async def predict(self, text: str, use_cache: bool = True) -> Dict:
        """
        Prédire l'intention d'un texte

        Args:
            text: Texte à classifier
            use_cache: Utiliser le cache ou non

        Returns:
            Dict contenant intent, confidence, et all_intents
        """

        self.stats['total_predictions'] += 1

        # Vérifier cache si activé
        if use_cache:
            cached_result = await self._get_from_cache(text)
            if cached_result:
                self.stats['cache_hits'] += 1
                cached_result['from_cache'] = True
                return cached_result

        self.stats['cache_misses'] += 1

        # Preprocessing
        text_clean = self._preprocess_text(text)

        # Tokenization
        encoding = self.tokenizer.encode_plus(
            text_clean,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)[0]

        # Résultats
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
        predicted_intent = self.id2label[predicted_idx]

        # Tous les intents avec leurs scores
        all_intents = {
            self.id2label[i]: float(probabilities[i])
            for i in range(self.num_labels)
        }

        result = {
            'intent': predicted_intent,
            'confidence': confidence,
            'all_intents': all_intents,
            'from_cache': False
        }

        # Mettre en cache
        if use_cache:
            await self._put_in_cache(text, result)

        return result

    async def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Prédire pour un batch de textes (plus efficace)"""

        # Preprocessing
        texts_clean = [self._preprocess_text(t) for t in texts]

        # Tokenization batch
        encodings = self.tokenizer.batch_encode_plus(
            texts_clean,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        # Inference batch
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)

        # Convertir résultats
        results = []
        for i in range(len(texts)):
            probs = probabilities[i]
            predicted_idx = torch.argmax(probs).item()
            confidence = probs[predicted_idx].item()
            predicted_intent = self.id2label[predicted_idx]

            all_intents = {
                self.id2label[j]: float(probs[j])
                for j in range(self.num_labels)
            }

            results.append({
                'intent': predicted_intent,
                'confidence': confidence,
                'all_intents': all_intents,
                'from_cache': False
            })

        return results

    def _preprocess_text(self, text: str) -> str:
        """Prétraitement basique du texte"""
        # Normalisation simple
        text = text.strip()
        # Ajouter plus de preprocessing si nécessaire
        return text

    async def _get_from_cache(self, text: str) -> Optional[Dict]:
        """Récupérer du cache (memory → redis)"""

        cache_key = self._generate_cache_key(text)

        # L1: Memory cache
        if cache_key in self.memory_cache:
            logger.debug(f"Cache hit (memory): {text[:50]}")
            return self.memory_cache[cache_key]

        # L2: Redis cache
        if self.redis_client:
            try:
                cached_value = await self.redis_client.get(cache_key)
                if cached_value:
                    logger.debug(f"Cache hit (redis): {text[:50]}")
                    result = json.loads(cached_value)
                    # Populate memory cache
                    self._add_to_memory_cache(cache_key, result)
                    return result
            except Exception as e:
                logger.error(f"Redis get error: {e}")

        return None

    async def _put_in_cache(self, text: str, result: Dict):
        """Mettre en cache"""

        cache_key = self._generate_cache_key(text)

        # Memory cache
        self._add_to_memory_cache(cache_key, result)

        # Redis cache (TTL: 1 heure)
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    3600,
                    json.dumps(result)
                )
            except Exception as e:
                logger.error(f"Redis set error: {e}")

    def _generate_cache_key(self, text: str) -> str:
        """Générer clé de cache déterministe"""
        return f"intent:{hashlib.md5(text.encode()).hexdigest()}"

    def _add_to_memory_cache(self, key: str, value: Dict):
        """Ajouter au cache mémoire avec LRU"""
        if len(self.memory_cache) >= self.max_cache_size:
            # Supprimer le plus ancien
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]

        self.memory_cache[key] = value

    def get_stats(self) -> Dict:
        """Obtenir statistiques du service"""
        cache_hit_rate = (
            self.stats['cache_hits'] / self.stats['total_predictions']
            if self.stats['total_predictions'] > 0 else 0
        )

        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'memory_cache_size': len(self.memory_cache)
        }

