import pytest
import torch
from app.models.intent_classifier import IntentClassifier


def test_model_initialization():
    """Test création du modèle"""
    model = IntentClassifier(num_labels=5)

    assert model is not None
    assert isinstance(model, torch.nn.Module)
    assert model.classifier.out_features == 5


def test_model_forward_pass():
    """Test forward pass"""
    model = IntentClassifier(num_labels=5)
    model.eval()

    # Dummy input
    batch_size = 2
    seq_length = 128

    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))

    # Forward
    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    # Vérifications
    assert logits.shape == (batch_size, 5)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()


def test_model_output_probabilities():
    """Test que les proba sommées = 1"""
    model = IntentClassifier(num_labels=5)
    model.eval()

    input_ids = torch.randint(0, 1000, (1, 128))
    attention_mask = torch.ones((1, 128))

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)

    # Somme des probabilités = 1
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-6)

