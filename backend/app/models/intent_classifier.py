import torch
import torch.nn as nn
from transformers import CamembertModel, CamembertTokenizer
from typing import Dict, List, Tuple
import numpy as np

class IntentClassifier(nn.Module):
    """
    Classifieur d'intention basé sur CamemBERT
    """

    def __init__(
            self,
            num_labels: int,
            model_name: str = "camembert-base",
            dropout: float = 0.1,
            hidden_size: int = 768
    ):
        super(IntentClassifier, self).__init__()

        # CamemBERT encoder
        self.camembert = CamembertModel.from_pretrained(model_name)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Init weights
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        if self.classifier.bias is not None:
            self.classifier.bias.data.zero_()

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)

        Returns:
            logits: (batch_size, num_labels)
        """
        # CamemBERT encoding
        outputs = self.camembert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Utiliser [CLS] token (premier token)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

