import json
import random
from typing import List, Dict, Tuple
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Importer les intents
import sys
sys.path.append('..')
from app.core.intents import INTENT_EXAMPLES, INTENT_DESCRIPTIONS

class DatasetCreator:
    def __init__(self, output_dir: str = "./processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = []

    def load_seed_data(self):
        """Charger les exemples de base"""
        for intent, examples in INTENT_EXAMPLES.items():
            for text in examples:
                self.data.append({
                    "text": text,
                    "intent": intent
                })

        print(f"✅ Loaded {len(self.data)} seed examples")
        return self

    def augment_data(self, augmentation_factor: int = 10):
        """Augmentation des données avec nlpaug"""
        import nlpaug.augmenter.word as naw

        print("🔄 Starting data augmentation...")

        # Augmenteurs
        synonym_aug = naw.SynonymAug(aug_src='wordnet', lang='fra')

        # Augmentation contextuelle avec un modèle plus léger
        try:
            contextual_aug = naw.ContextualWordEmbsAug(
                model_path='camembert-base',
                action="substitute",
                aug_min=1,
                aug_max=3
            )
        except:
            print("⚠️ Contextual augmentation not available, using synonym only")
            contextual_aug = None

        original_data = self.data.copy()
        augmented_count = 0

        for item in original_data:
            text = item['text']
            intent = item['intent']

            # Générer variations
            for _ in range(augmentation_factor):
                try:
                    # Méthode 1: Synonymes
                    aug_text = synonym_aug.augment(text)
                    if isinstance(aug_text, list):
                        aug_text = aug_text[0]

                    if aug_text and aug_text != text:
                        self.data.append({
                            "text": aug_text,
                            "intent": intent,
                            "augmented": True,
                            "method": "synonym"
                        })
                        augmented_count += 1

                    # Méthode 2: Contextuel (si disponible)
                    if contextual_aug and random.random() > 0.5:
                        aug_text = contextual_aug.augment(text)
                        if isinstance(aug_text, list):
                            aug_text = aug_text[0]

                        if aug_text and aug_text != text:
                            self.data.append({
                                "text": aug_text,
                                "intent": intent,
                                "augmented": True,
                                "method": "contextual"
                            })
                            augmented_count += 1

                except Exception as e:
                    continue

        print(f"✅ Generated {augmented_count} augmented examples")
        print(f"📊 Total dataset size: {len(self.data)}")
        return self

    def add_manual_examples(self, examples: List[Dict]):
        """Ajouter des exemples manuels"""
        self.data.extend(examples)
        print(f"✅ Added {len(examples)} manual examples")
        return self

    def validate_dataset(self) -> bool:
        """Valider la qualité du dataset"""
        print("\n🔍 Validating dataset...")

        # Comptage par intent
        intent_counts = Counter(item['intent'] for item in self.data)

        # Vérifier minimum 100 exemples par intent
        min_examples = 100
        all_valid = True

        for intent, count in intent_counts.items():
            status = "✅" if count >= min_examples else "❌"
            print(f"{status} {intent}: {count} examples (min: {min_examples})")
            if count < min_examples:
                all_valid = False

        # Vérifier duplicates
        texts = [item['text'] for item in self.data]
        duplicates = len(texts) - len(set(texts))
        print(f"\n📊 Duplicates: {duplicates}")

        # Longueur moyenne
        avg_length = sum(len(item['text'].split()) for item in self.data) / len(self.data)
        print(f"📏 Average text length: {avg_length:.1f} words")

        return all_valid

    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split en train/val/test avec stratification"""

        # Convertir en DataFrame
        df = pd.DataFrame(self.data)

        # Stratified split
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - train_ratio),
            stratify=df['intent'],
            random_state=42
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=test_ratio / (val_ratio + test_ratio),
            stratify=temp_df['intent'],
            random_state=42
        )

        print(f"\n📊 Dataset split:")
        print(f"  Train: {len(train_df)} ({train_ratio*100:.0f}%)")
        print(f"  Val:   {len(val_df)} ({val_ratio*100:.0f}%)")
        print(f"  Test:  {len(test_df)} ({test_ratio*100:.0f}%)")

        return train_df, val_df, test_df

    def save_dataset(self, train_df, val_df, test_df):
        """Sauvegarder les datasets"""

        # JSON format
        train_df.to_json(self.output_dir / "train.json", orient='records', lines=True)
        val_df.to_json(self.output_dir / "val.json", orient='records', lines=True)
        test_df.to_json(self.output_dir / "test.json", orient='records', lines=True)

        # CSV format (pour inspection)
        train_df.to_csv(self.output_dir / "train.csv", index=False)
        val_df.to_csv(self.output_dir / "val.csv", index=False)
        test_df.to_csv(self.output_dir / "test.csv", index=False)

        # Metadata
        metadata = {
            "num_intents": len(train_df['intent'].unique()),
            "intents": list(train_df['intent'].unique()),
            "intent_descriptions": INTENT_DESCRIPTIONS,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "total_size": len(train_df) + len(val_df) + len(test_df)
        }

        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ Datasets saved to {self.output_dir}")

    def visualize_dataset(self, df: pd.DataFrame, split_name: str = "train"):
        """Créer des visualisations du dataset"""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Dataset Analysis - {split_name.upper()}', fontsize=16)

        # 1. Distribution des intents
        intent_counts = df['intent'].value_counts()
        axes[0, 0].bar(intent_counts.index, intent_counts.values)
        axes[0, 0].set_title('Intent Distribution')
        axes[0, 0].set_xlabel('Intent')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Distribution des longueurs de texte
        df['text_length'] = df['text'].apply(lambda x: len(x.split()))
        axes[0, 1].hist(df['text_length'], bins=30, edgecolor='black')
        axes[0, 1].set_title('Text Length Distribution')
        axes[0, 1].set_xlabel('Number of Words')
        axes[0, 1].set_ylabel('Frequency')

        # 3. Box plot longueurs par intent
        df.boxplot(column='text_length', by='intent', ax=axes[1, 0])
        axes[1, 0].set_title('Text Length by Intent')
        axes[1, 0].set_xlabel('Intent')
        axes[1, 0].set_ylabel('Number of Words')

        # 4. Statistiques
        stats_text = f"""
        Total Examples: {len(df)}
        Unique Intents: {df['intent'].nunique()}
        Avg Text Length: {df['text_length'].mean():.1f} words
        Min Text Length: {df['text_length'].min()} words
        Max Text Length: {df['text_length'].max()} words
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{split_name}_analysis.png', dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved: {split_name}_analysis.png")

# ============================================
# Script principal
# ============================================

def main():
    """Pipeline complet de création du dataset"""

    print("="*60)
    print("🚀 DATASET CREATION PIPELINE")
    print("="*60)

    # Initialiser
    creator = DatasetCreator(output_dir="../data/processed")

    # 1. Charger seed data
    creator.load_seed_data()

    # 2. Augmentation (génère ~10x plus d'exemples)
    creator.augment_data(augmentation_factor=10)

    # 3. Valider
    if not creator.validate_dataset():
        print("\n⚠️ Dataset validation failed! Adding more examples...")
        # Ici vous pouvez ajouter plus d'exemples manuels

    # 4. Split
    train_df, val_df, test_df = creator.split_dataset()

    # 5. Visualiser
    creator.visualize_dataset(train_df, "train")
    creator.visualize_dataset(val_df, "val")
    creator.visualize_dataset(test_df, "test")

    # 6. Sauvegarder
    creator.save_dataset(train_df, val_df, test_df)

    print("\n" + "="*60)
    print("✅ DATASET CREATION COMPLETE!")
    print("="*60)

    # Afficher quelques exemples
    print("\n📋 Sample examples:")
    for intent in train_df['intent'].unique()[:3]:
        samples = train_df[train_df['intent'] == intent].head(2)
        print(f"\n{intent}:")
        for _, row in samples.iterrows():
            print(f"  - {row['text']}")


if __name__ == "__main__":
    main()
