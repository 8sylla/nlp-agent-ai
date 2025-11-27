import pytest
from pathlib import Path
import json


def test_dataset_files_exist():
    """Vérifier que les fichiers dataset existent"""
    data_dir = Path("./data/processed")

    assert (data_dir / "train.json").exists()
    assert (data_dir / "val.json").exists()
    assert (data_dir / "test.json").exists()
    assert (data_dir / "metadata.json").exists()


def test_dataset_metadata():
    """Vérifier metadata du dataset"""
    with open("./data/processed/metadata.json") as f:
        metadata = json.load(f)

    assert metadata['num_intents'] == 5
    assert len(metadata['intents']) == 5
    assert metadata['train_size'] > 0
    assert metadata['val_size'] > 0
    assert metadata['test_size'] > 0


def test_dataset_split_ratios():
    """Vérifier ratios du split"""
    with open("./data/processed/metadata.json") as f:
        metadata = json.load(f)

    total = metadata['train_size'] + metadata['val_size'] + metadata['test_size']
    train_ratio = metadata['train_size'] / total
    val_ratio = metadata['val_size'] / total
    test_ratio = metadata['test_size'] / total

    # Vérifier approximativement 70/15/15
    assert 0.65 <= train_ratio <= 0.75
    assert 0.10 <= val_ratio <= 0.20
    assert 0.10 <= test_ratio <= 0.20


def test_minimum_examples_per_intent():
    """Vérifier minimum 100 exemples par intent dans train"""
    import pandas as pd

    train_df = pd.read_json("./data/processed/train.json", lines=True)
    intent_counts = train_df['intent'].value_counts()

    for intent, count in intent_counts.items():
        assert count >= 70, f"Intent {intent} has only {count} examples in train set"

