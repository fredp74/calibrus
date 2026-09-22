"""
calibrus/model.py — Calibrus hybrid model: text encoder (DistilBERT) + numeric
encoder (MLP) fused together, classification head (3 classes) + regression head (score).

Raw output (before calibration): class logits + continuous score.
Calibration (temperature) is applied separately at inference time (see calibrate.py).
"""

import torch
import torch.nn as nn
from transformers import AutoModel

NUM_CLASSES = 3  # short=0, hold=1, long=2


class NumericEncoder(nn.Module):
    """Simple MLP to encode numeric features (feat_*) into a dense vector."""

    def __init__(self, n_features: int, hidden_dim: int = 128, out_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        return self.net(x_num)


class CalibrusModel(nn.Module):
    """
    Hybrid model: text (frozen or fine-tuned DistilBERT) + numeric (MLP) -> fusion
    -> classification head (route) + regression head (continuous score).
    """

    def __init__(self, n_features: int, text_model_name: str = "distilbert-base-uncased",
                 freeze_text_encoder: bool = False, num_hidden_dim: int = 128,
                 num_out_dim: int = 64, fusion_hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        self.numeric_encoder = NumericEncoder(
            n_features=n_features, hidden_dim=num_hidden_dim,
            out_dim=num_out_dim, dropout=dropout,
        )

        fusion_dim = text_dim + num_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classification_head = nn.Linear(fusion_hidden_dim, NUM_CLASSES)
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 1),
            nn.Tanh(),  # score bounded to [-1, 1]
        )

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    def forward(self, x_num: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor):
        text_repr = self.encode_text(input_ids, attention_mask)
        num_repr = self.numeric_encoder(x_num)
        fused = self.fusion(torch.cat([text_repr, num_repr], dim=-1))

        class_logits = self.classification_head(fused)
        score = self.regression_head(fused).squeeze(-1)

        return {"class_logits": class_logits, "score": score}

    @classmethod
    def from_dataset_meta(cls, meta: dict, **kwargs) -> "CalibrusModel":
        """Builds the model directly from the metadata saved by dataset.py."""
        n_features = len(meta["feat_cols"])
        text_model_name = meta.get("text_model", "distilbert-base-uncased")
        return cls(n_features=n_features, text_model_name=text_model_name, **kwargs)
