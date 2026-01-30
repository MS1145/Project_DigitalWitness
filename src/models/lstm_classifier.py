"""
LSTM-based temporal intent classifier for Digital Witness.

Uses LSTM networks to model temporal patterns in feature sequences
for classifying shopping intent: normal, suspicious, shoplifting.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path

from ..config import (
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    LSTM_SEQUENCE_LENGTH,
    LSTM_DROPOUT,
    INTENT_CLASSES,
    MODELS_DIR
)


@dataclass
class IntentPrediction:
    """Prediction result from LSTM classifier."""
    intent_class: str             # "normal", "pickup", "concealment", "bypass"
    class_id: int
    confidence: float             # Prediction confidence
    class_probabilities: Dict[str, float]  # Per-class probabilities
    temporal_attention: Optional[np.ndarray] = None  # Which frames contributed most
    sequence_id: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "intent_class": self.intent_class,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "class_probabilities": self.class_probabilities,
            "sequence_id": self.sequence_id,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


class LSTMIntentClassifier:
    """
    LSTM-based classifier for temporal intent prediction.

    Processes sequences of frame features to classify shopping behavior
    over time. Uses bidirectional LSTM with attention mechanism.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = LSTM_HIDDEN_DIM,
        num_layers: int = LSTM_NUM_LAYERS,
        num_classes: int = len(INTENT_CLASSES),
        dropout: float = LSTM_DROPOUT,
        bidirectional: bool = True,
        device: str = "auto"
    ):
        """
        Initialize LSTM classifier.

        Args:
            input_dim: Input feature dimension (from CNN)
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
            device: Device for inference
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.device_str = device

        self.classes = INTENT_CLASSES
        self.model = None
        self.device = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization of PyTorch model."""
        if self._initialized:
            return

        try:
            import torch
            import torch.nn as nn

            # Determine device
            if self.device_str == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_str)

            # Build LSTM model
            self.model = LSTMModel(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_classes=self.num_classes,
                dropout=self.dropout,
                bidirectional=self.bidirectional
            ).to(self.device)

            self.model.eval()
            self._initialized = True

        except ImportError:
            raise ImportError(
                "PyTorch not installed. Run: pip install torch"
            )

    def predict_sequence(
        self,
        features: np.ndarray,
        sequence_id: str = "",
        start_time: float = 0.0,
        end_time: float = 0.0
    ) -> IntentPrediction:
        """
        Predict intent from a temporal feature sequence.

        Args:
            features: Feature sequence of shape (seq_len, feature_dim)
            sequence_id: Identifier for this sequence
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            IntentPrediction with class and confidence
        """
        self.initialize()

        import torch

        # Ensure correct shape: (batch=1, seq_len, features)
        if features.ndim == 2:
            features = features[np.newaxis, :, :]

        # Convert to tensor
        input_tensor = torch.FloatTensor(features).to(self.device)

        # Run inference
        with torch.no_grad():
            logits, attention = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)

        # Get prediction
        probs_np = probs.cpu().numpy()[0]
        attention_np = attention.cpu().numpy()[0] if attention is not None else None

        predicted_class = int(np.argmax(probs_np))
        confidence = float(probs_np[predicted_class])

        # Build probability dict
        class_probs = {
            cls: float(probs_np[i])
            for i, cls in enumerate(self.classes)
        }

        return IntentPrediction(
            intent_class=self.classes[predicted_class],
            class_id=predicted_class,
            confidence=confidence,
            class_probabilities=class_probs,
            temporal_attention=attention_np,
            sequence_id=sequence_id,
            start_time=start_time,
            end_time=end_time
        )

    def predict_batch(
        self,
        features_list: List[np.ndarray],
        sequence_info: Optional[List[Dict]] = None
    ) -> List[IntentPrediction]:
        """
        Predict intent for multiple sequences.

        Args:
            features_list: List of feature sequences
            sequence_info: Optional list of dicts with sequence_id, start_time, end_time

        Returns:
            List of IntentPrediction objects
        """
        predictions = []

        for i, features in enumerate(features_list):
            info = sequence_info[i] if sequence_info else {}
            pred = self.predict_sequence(
                features,
                sequence_id=info.get("sequence_id", f"seq_{i}"),
                start_time=info.get("start_time", 0.0),
                end_time=info.get("end_time", 0.0)
            )
            predictions.append(pred)

        return predictions

    def train(
        self,
        sequences: List[np.ndarray],
        labels: List[int],
        val_sequences: Optional[List[np.ndarray]] = None,
        val_labels: Optional[List[int]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        early_stopping: int = 10
    ) -> Dict[str, Any]:
        """
        Train the LSTM classifier.

        Args:
            sequences: List of training sequences
            labels: Training labels
            val_sequences: Validation sequences
            val_labels: Validation labels
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            early_stopping: Patience for early stopping

        Returns:
            Training history and metrics
        """
        self.initialize()

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Set to training mode
        self.model.train()

        # Prepare data
        X_train = self._pad_sequences(sequences)
        y_train = np.array(labels)

        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Prepare validation data
        val_loader = None
        if val_sequences and val_labels:
            X_val = self._pad_sequences(val_sequences)
            y_val = np.array(val_labels)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            train_acc = train_correct / train_total

            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            val_loss = 0.0
            val_acc = 0.0

            if val_loader:
                self.model.eval()
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs, _ = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()

                avg_val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total

                history["val_loss"].append(avg_val_loss)
                history["val_acc"].append(val_acc)

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            # Progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}: "
                      f"Train Loss={avg_train_loss:.4f}, Acc={train_acc:.4f}, "
                      f"Val Loss={avg_val_loss:.4f}, Acc={val_acc:.4f}")

        self.model.eval()

        return {
            "history": history,
            "final_train_acc": history["train_acc"][-1],
            "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None
        }

    def _pad_sequences(
        self,
        sequences: List[np.ndarray],
        max_len: Optional[int] = None
    ) -> np.ndarray:
        """Pad sequences to same length."""
        if not sequences:
            return np.array([])

        if max_len is None:
            max_len = max(seq.shape[0] for seq in sequences)

        feature_dim = sequences[0].shape[1]
        padded = np.zeros((len(sequences), max_len, feature_dim))

        for i, seq in enumerate(sequences):
            length = min(seq.shape[0], max_len)
            padded[i, :length, :] = seq[:length]

        return padded

    def save_model(self, path: Optional[Path] = None):
        """Save model weights."""
        self.initialize()

        import torch

        path = path or (MODELS_DIR / "lstm_classifier.pt")
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_classes": self.num_classes,
                "dropout": self.dropout,
                "bidirectional": self.bidirectional,
                "classes": self.classes
            }
        }, path)

    def load_model(self, path: Optional[Path] = None):
        """Load model weights."""
        import torch

        path = path or (MODELS_DIR / "lstm_classifier.pt")

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path, map_location="cpu")

        # Update config
        config = checkpoint["config"]
        self.input_dim = config["input_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.num_layers = config["num_layers"]
        self.num_classes = config["num_classes"]
        self.dropout = config["dropout"]
        self.bidirectional = config["bidirectional"]
        self.classes = config["classes"]

        # Initialize and load weights
        self.initialize()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def close(self):
        """Release resources."""
        self.model = None
        self._initialized = False


class LSTMModel:
    """PyTorch LSTM model (defined separately to avoid torch import at module level)."""

    def __new__(cls, *args, **kwargs):
        import torch
        import torch.nn as nn

        class _LSTMModel(nn.Module):
            def __init__(
                self,
                input_dim: int,
                hidden_dim: int,
                num_layers: int,
                num_classes: int,
                dropout: float,
                bidirectional: bool
            ):
                super().__init__()

                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                self.bidirectional = bidirectional

                # LSTM layer
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional
                )

                # Attention layer
                attention_dim = hidden_dim * 2 if bidirectional else hidden_dim
                self.attention = nn.Sequential(
                    nn.Linear(attention_dim, attention_dim // 2),
                    nn.Tanh(),
                    nn.Linear(attention_dim // 2, 1)
                )

                # Output layer
                self.fc = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(attention_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes)
                )

            def forward(self, x):
                # LSTM forward
                lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*directions)

                # Attention
                attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
                attention_weights = torch.softmax(attention_weights, dim=1)

                # Weighted sum
                context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*directions)

                # Output
                output = self.fc(context)

                return output, attention_weights.squeeze(-1)

        return _LSTMModel(*args, **kwargs)
