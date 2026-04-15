"""
model_definitions.py — PyTorch nn.Module class definitions.

These classes mirror exactly what was trained in the notebook.
No file I/O, no Streamlit, no OpenCV — pure PyTorch.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


class MobileNetExtractor(nn.Module):
    """
    MobileNetV2 spatial feature extractor.

    The classifier head is kept for checkpoint compatibility but is not used
    during inference — only extract_features() is called by FeatureExtractor.

    Input  : (B, 3, 224, 224) normalised tensor
    Output : (B, 1280) feature vector
    """

    def __init__(self) -> None:
        super().__init__()
        base = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.features   = base.features
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 2),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return (1, 1280) feature vector without gradient tracking."""
        with torch.no_grad():
            x = self.features(x)
            x = self.pool(x)
            return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class TemporalAttention(nn.Module):
    """
    Soft attention over BiLSTM hidden states.

    Learns to weight frames by importance so the classifier can focus on
    the most discriminative moments in a sequence.
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_out: (B, T, hidden_size)
        Returns:
            context : (B, hidden_size) — attended context vector
            weights : (B, T)           — per-frame attention weights
        """
        scores  = self.attn(lstm_out).squeeze(-1)      # (B, T)
        weights = torch.softmax(scores, dim=1)           # (B, T)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, weights


class BiLSTMAttentionClassifier(nn.Module):
    """
    Bidirectional LSTM + temporal attention for behaviour classification.

    Input : (B, T=45, 1280) feature sequences from MobileNetExtractor
    Output: logits (B, num_classes) + attention weights (B, T)
    """

    def __init__(self, num_classes: int = 2, hidden: int = 256,
                 layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=1280,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.attention  = TemporalAttention(hidden * 2)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _    = self.bilstm(x)
        ctx, attn = self.attention(out)
        return self.classifier(self.dropout(ctx)), attn
