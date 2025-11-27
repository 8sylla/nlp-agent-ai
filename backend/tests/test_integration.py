import pytest


@pytest.mark.asyncio
async def test_full_pipeline(nlu_service):
    """Test du pipeline complet: texte → intent"""

    test_cases = [
        {
            "text": "Bonjour, je veux suivre ma commande",
            "expected_intent": "track_order",
            "min_confidence": 0.6
        },
        {
            "text": "Je voudrais retourner cet article",
            "expected_intent": "return_request",
            "min_confidence": 0.6
        },
        {
            "text": "Salut !",
            "expected_intent": "greeting",
            "min_confidence": 0.7
        },
        {
            "text": "Mon paiement a échoué",
            "expected_intent": "payment_issue",
            "min_confidence": 0.5
        },
        {
            "text": "Ce produit existe en noir ?",
            "expected_intent": "product_inquiry",
            "min_confidence": 0.5
        }
    ]

    for case in test_cases:
        result = await nlu_service.predict(case["text"], use_cache=False)

        assert result['intent'] == case['expected_intent'], \
            f"Failed for '{case['text']}': got {result['intent']}, expected {case['expected_intent']}"

        assert result['confidence'] >= case['min_confidence'], \
            f"Low confidence for '{case['text']}': {result['confidence']}"


@pytest.mark.asyncio
async def test_cache_performance(nlu_service):
    """Test que le cache améliore les performances"""
    import time

    text = "Performance test"

    # Sans cache
    start = time.time()
    await nlu_service.predict(text, use_cache=False)
    time_no_cache = time.time() - start

    # Avec cache (premier appel)
    start = time.time()
    await nlu_service.predict(text, use_cache=True)
    time_first_call = time.time() - start

    # Avec cache (deuxième appel - devrait être en cache)
    start = time.time()
    result = await nlu_service.predict(text, use_cache=True)
    time_cached = time.time() - start

    # Le deuxième appel avec cache devrait être significativement plus rapide
    assert result['from_cache'] is True
    assert time_cached < time_no_cache / 2, "Cache should improve performance significantly"


# ============================================
# Script pour lancer tous les tests
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])