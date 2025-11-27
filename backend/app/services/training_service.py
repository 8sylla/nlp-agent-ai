import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CamembertTokenizer, get_linear_schedule_with_warmup
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.intent_classifier import IntentClassifier
from ..core.config import settings

class IntentDataset(Dataset):
    """Dataset PyTorch pour les intents"""

    def __init__(
            self,
            texts: List[str],
            labels: List[int],
            tokenizer: CamembertTokenizer,
            max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenization
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class IntentClassifierTrainer:
    """Entraîneur pour le modèle de classification d'intention"""

    def __init__(
            self,
            data_dir: str = "./data/processed",
            model_cache_dir: str = "./models",
            experiment_name: str = "intent-classification"
    ):
        self.data_dir = Path(data_dir)
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")

        # MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)

        # Charger metadata
        with open(self.data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        self.num_labels = self.metadata['num_intents']
        self.label2id = {label: i for i, label in enumerate(self.metadata['intents'])}
        self.id2label = {i: label for label, i in self.label2id.items()}

        print(f"📊 Loaded metadata: {self.num_labels} intents")

        # Tokenizer
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Charger et préparer les dataloaders"""

        print("📥 Loading datasets...")

        # Charger JSONs
        import pandas as pd
        train_df = pd.read_json(self.data_dir / "train.json", lines=True)
        val_df = pd.read_json(self.data_dir / "val.json", lines=True)
        test_df = pd.read_json(self.data_dir / "test.json", lines=True)

        # Convertir labels en IDs
        train_labels = [self.label2id[label] for label in train_df['intent']]
        val_labels = [self.label2id[label] for label in val_df['intent']]
        test_labels = [self.label2id[label] for label in test_df['intent']]

        # Créer datasets
        train_dataset = IntentDataset(
            train_df['text'].tolist(),
            train_labels,
            self.tokenizer
        )

        val_dataset = IntentDataset(
            val_df['text'].tolist(),
            val_labels,
            self.tokenizer
        )

        test_dataset = IntentDataset(
            test_df['text'].tolist(),
            test_labels,
            self.tokenizer
        )

        # Créer dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0  # Important pour Windows
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )

        print(f"✅ Data loaded:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Test batches: {len(test_loader)}")

        return train_loader, val_loader, test_loader

    def train(
            self,
            num_epochs: int = 5,
            learning_rate: float = 2e-5,
            warmup_steps: int = 500,
            weight_decay: float = 0.01
    ):
        """Pipeline d'entraînement complet"""

        print("\n" + "="*60)
        print("🚀 TRAINING PIPELINE")
        print("="*60)

        # Start MLflow run
        with mlflow.start_run() as run:

            # Log hyperparameters
            params = {
                'model_name': 'camembert-base',
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'batch_size': 16,
                'warmup_steps': warmup_steps,
                'weight_decay': weight_decay,
                'num_labels': self.num_labels
            }
            mlflow.log_params(params)

            # Charger données
            train_loader, val_loader, test_loader = self.load_data()

            # Initialiser modèle
            model = IntentClassifier(num_labels=self.num_labels)
            model.to(self.device)

            # Loss et optimizer
            criterion = nn.CrossEntropyLoss()

            # Optimizer avec weight decay
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in model.named_parameters()
                               if not any(nd in n for nd in no_decay)],
                    'weight_decay': weight_decay
                },
                {
                    'params': [p for n, p in model.named_parameters()
                               if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0
                }
            ]

            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

            # Learning rate scheduler
            total_steps = len(train_loader) * num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

            # Training loop
            best_val_f1 = 0.0

            for epoch in range(num_epochs):
                print(f"\n📚 Epoch {epoch+1}/{num_epochs}")

                # Train
                train_loss = self._train_epoch(
                    model, train_loader, criterion, optimizer, scheduler
                )

                # Validation
                val_loss, val_metrics = self._evaluate(
                    model, val_loader, criterion
                )

                # Log metrics
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1'],
                    'val_precision': val_metrics['precision'],
                    'val_recall': val_metrics['recall']
                }, step=epoch)

                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  Val F1: {val_metrics['f1']:.4f}")

                # Save best model
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    self._save_model(model, "best_model")
                    print(f"  💾 Saved best model (F1: {best_val_f1:.4f})")

            # Test sur best model
            print("\n🧪 Testing best model...")
            best_model = IntentClassifier(num_labels=self.num_labels)
            best_model.load_state_dict(
                torch.load(self.model_cache_dir / "best_model" / "model.pt")
            )
            best_model.to(self.device)

            test_loss, test_metrics = self._evaluate(
                best_model, test_loader, criterion
            )

            # Log test metrics
            mlflow.log_metrics({
                'test_loss': test_loss,
                'test_accuracy': test_metrics['accuracy'],
                'test_f1': test_metrics['f1'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall']
            })

            print(f"\n📊 Test Results:")
            print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  F1 Score: {test_metrics['f1']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall: {test_metrics['recall']:.4f}")

            # Confusion matrix
            self._plot_confusion_matrix(
                test_metrics['y_true'],
                test_metrics['y_pred']
            )

            # Log model
            mlflow.pytorch.log_model(best_model, "model")

            print("\n✅ Training complete!")
            print(f"📁 Model saved to: {self.model_cache_dir / 'best_model'}")
            print(f"🔗 MLflow run: {run.info.run_id}")

    def _train_epoch(
            self,
            model: nn.Module,
            dataloader: DataLoader,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LambdaLR
    ) -> float:
        """Entraîner une epoch"""

        model.train()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc="Training")

        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        return total_loss / len(dataloader)

    def _evaluate(
            self,
            model: nn.Module,
            dataloader: DataLoader,
            criterion: nn.Module
    ) -> Tuple[float, Dict]:
        """Évaluer le modèle"""

        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculer métriques
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_true': all_labels,
            'y_pred': all_preds
        }

        return total_loss / len(dataloader), metrics

    def _save_model(self, model: nn.Module, name: str):
        """Sauvegarder le modèle"""
        save_dir = self.model_cache_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder poids
        torch.save(model.state_dict(), save_dir / "model.pt")

        # Sauvegarder config
        config = {
            'num_labels': self.num_labels,
            'label2id': self.label2id,
            'id2label': self.id2label,
            'model_name': 'camembert-base'
        }

        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Sauvegarder tokenizer
        self.tokenizer.save_pretrained(save_dir)

    def _plot_confusion_matrix(self, y_true: List[int], y_pred: List[int]):
        """Créer et sauvegarder la confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[self.id2label[i] for i in range(self.num_labels)],
            yticklabels=[self.id2label[i] for i in range(self.num_labels)]
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Sauvegarder
        save_path = self.model_cache_dir / "confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved: {save_path}")

        # Log to MLflow
        mlflow.log_artifact(str(save_path))

# ============================================
# Script d'entraînement
# ============================================

def main():
    """Lancer l'entraînement"""

    trainer = IntentClassifierTrainer(
        data_dir="./data/processed",
        model_cache_dir="./models"
    )

    trainer.train(
        num_epochs=5,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01
    )


if __name__ == "__main__":
    main()