import pytest


@pytest.mark.asyncio
async def test_nlu_service_initialization(nlu_service):
    """Test initialisation du service"""
    assert nlu_service is not None
    assert nlu_service.model is not None
    assert nlu_service.tokenizer is not None
    assert nlu_service.num_labels == 5


@pytest.mark.asyncio
async def test_predict_track_order(nlu_service):
    """Test prédiction intent track_order"""
    text = "Où est ma commande ?"
    result = await nlu_service.predict(text, use_cache=False)

    assert result['intent'] == 'track_order'
    assert result['confidence'] > 0.5
    assert 'all_intents' in result
    assert len(result['all_intents']) == 5


@pytest.mark.asyncio
async def test_predict_return_request(nlu_service):
    """Test prédiction intent return_request"""
    text = "Je veux retourner un article"
    result = await nlu_service.predict(text, use_cache=False)

    assert result['intent'] == 'return_request'
    assert result['confidence'] > 0.5


@pytest.mark.asyncio
async def test_predict_greeting(nlu_service):
    """Test prédiction intent greeting"""
    text = "Bonjour"
    result = await nlu_service.predict(text, use_cache=False)

    assert result['intent'] == 'greeting'
    assert result['confidence'] > 0.7


@pytest.mark.asyncio
async def test_predict_product_inquiry(nlu_service):
    """Test prédiction intent product_inquiry"""
    text = "Ce produit est disponible en bleu ?"
    result = await nlu_service.predict(text, use_cache=False)

    assert result['intent'] == 'product_inquiry'
    assert result['confidence'] > 0.5


@pytest.mark.asyncio
async def test_predict_payment_issue(nlu_service):
    """Test prédiction intent payment_issue"""
    text = "Mon paiement n'est pas passé"
    result = await nlu_service.predict(text, use_cache=False)

    assert result['intent'] == 'payment_issue'
    assert result['confidence'] > 0.5


@pytest.mark.asyncio
async def test_cache_functionality(nlu_service):
    """Test que le cache fonctionne"""
    text = "Test de cache"

    # Premier appel (sans cache)
    result1 = await nlu_service.predict(text, use_cache=True)
    assert result1['from_cache'] is False

    # Deuxième appel (avec cache)
    result2 = await nlu_service.predict(text, use_cache=True)
    assert result2['from_cache'] is True

    # Résultats identiques
    assert result1['intent'] == result2['intent']
    assert result1['confidence'] == result2['confidence']


@pytest.mark.asyncio
async def test_predict_batch(nlu_service):
    """Test prédiction batch"""
    texts = [
        "Où est ma commande ?",
        "Je veux retourner un article",
        "Bonjour"
    ]

    results = await nlu_service.predict_batch(texts)

    assert len(results) == 3
    assert results[0]['intent'] == 'track_order'
    assert results[1]['intent'] == 'return_request'
    assert results[2]['intent'] == 'greeting'


@pytest.mark.asyncio
async def test_empty_text(nlu_service):
    """Test avec texte vide (edge case)"""
    text = ""

    # Devrait quand même retourner un résultat
    result = await nlu_service.predict(text, use_cache=False)
    assert 'intent' in result
    assert 'confidence' in result


@pytest.mark.asyncio
async def test_long_text(nlu_service):
    """Test avec texte très long (edge case)"""
    text = "Bonjour " * 200  # Dépasse max_length

    result = await nlu_service.predict(text, use_cache=False)
    assert 'intent' in result
    # Le modèle devrait gérer la troncation


@pytest.mark.asyncio
async def test_special_characters(nlu_service):
    """Test avec caractères spéciaux"""
    text = "Où est ma commande ?!?! 😊"

    result = await nlu_service.predict(text, use_cache=False)
    assert result['intent'] == 'track_order'


@pytest.mark.asyncio
async def test_stats(nlu_service):
    """Test récupération des stats"""
    stats = nlu_service.get_stats()

    assert 'total_predictions' in stats
    assert 'cache_hits' in stats
    assert 'cache_misses' in stats
    assert 'cache_hit_rate' in stats
    assert 'memory_cache_size' in stats

    assert stats['total_predictions'] >= 0
    assert 0 <= stats['cache_hit_rate'] <= 1

