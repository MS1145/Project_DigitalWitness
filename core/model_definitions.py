"""
model_definitions.py - PyTorch nn.Module class definitions.

These classes mirror exactly what was trained in the notebook.
No file I/O, no Streamlit, no OpenCV - pure PyTorch.
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
    during inference - only extract_features() is called by FeatureExtractor.

    Input: (B, 3, 224, 224) normalised tensor
    Output: (B, 1280) feature vector

    Architecture source:
        Sandler, M. et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks.
        Proceedings of the IEEE/CVF CVPR. https://arxiv.org/abs/1801.04381

    Pretrained weights:
        ImageNet-1K weights loaded via torchvision (MobileNet_V2_Weights.DEFAULT).
        The last 3 feature blocks were fine-tuned on our retail shoplifting dataset.
        torchvision: https://pytorch.org/vision/stable/models.html
    """

    def __init__(self) -> None:
        super().__init__()
        # load the pretrained backbone - we only fine-tune the last few blocks
        # so most of the ImageNet knowledge is preserved
        base = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.features   = base.features
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        # this head is never called during inference but the checkpoint was
        # saved with it so we keep it here to avoid a state_dict mismatch
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
    the most suspicious moments in a sequence rather than treating all
    frames equally.

    Based on the additive attention mechanism from:
        Bahdanau, D., Cho, K., & Bengio, Y. (2015).
        Neural Machine Translation by Jointly Learning to Align and Translate.
        ICLR 2015. https://arxiv.org/abs/1409.0473

    We adapted it for sequence classification (not translation) by collapsing
    the attended context into a single vector for the final linear layer.
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
            context: (B, hidden_size) - attended context vector
            weights: (B, T) - per-frame attention weights (useful for XAI)
        """
        scores  = self.attn(lstm_out).squeeze(-1)      # (B, T)
        weights = torch.softmax(scores, dim=1)         # (B, T)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, weights


class BiLSTMAttentionClassifier(nn.Module):
    """
    Bidirectional LSTM + temporal attention for behaviour classification.

    Input : (B, T=45, 1280) feature sequences from MobileNetExtractor
    Output: logits (B, num_classes) + attention weights (B, T)

    BiLSTM architecture based on:
        Schuster, M. & Paliwal, K.K. (1997). Bidirectional recurrent neural networks.
        IEEE Transactions on Signal Processing. https://doi.org/10.1109/78.650093

    We use bidirectional processing so the model can look at context both
    before and after each frame - helpful for catching behaviours that only
    make sense in hindsight (e.g. concealment followed by exit).

    Training config: hidden=256, layers=2, dropout=0.3, seq_len=45, stride=15
    Framework: PyTorch (Paszke et al., 2019) https://arxiv.org/abs/1912.01703
    """

    def __init__(self, num_classes: int = 2, hidden: int = 256,
                 layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=1280,  # matches MobileNetV2 output dim
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.attention  = TemporalAttention(hidden * 2)  # *2 because bidirectional
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _    = self.bilstm(x)
        ctx, attn = self.attention(out)
        return self.classifier(self.dropout(ctx)), attn
