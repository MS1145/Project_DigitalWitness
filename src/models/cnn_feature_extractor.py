"""
CNN-based spatial feature extraction for Digital Witness.

Supports multiple lightweight backbones optimized for speed:
- MobileNetV3-Small: Best speed/accuracy balance (recommended)
- EfficientNetV2-S: Highest accuracy, slower
- SqueezeNet: Fastest, compact, lower accuracy
- ResNet18: Baseline reference

Features:
- Batch processing for GPU efficiency
- Pretrained ImageNet weights (transfer learning)
- Configurable input size
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

from ..config import (
    CNN_BACKBONE,
    CNN_BACKBONE_CONFIGS,
    CNN_FEATURE_DIM,
    CNN_PRETRAINED,
    CNN_INPUT_SIZE
)


@dataclass
class FrameFeatures:
    """Extracted features for a single frame."""
    frame_number: int
    timestamp: float
    features: np.ndarray


@dataclass
class SequenceFeatures:
    """Features for a sequence of frames (LSTM input)."""
    sequence_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    features: np.ndarray
    frame_count: int


class CNNFeatureExtractor:
    """
    CNN feature extractor with multiple backbone support.

    Supported backbones:
    - mobilenet_v3_small: 2.5M params, ~5ms/frame (RECOMMENDED)
    - efficientnet_v2_s: 21M params, ~25ms/frame (highest accuracy)
    - squeezenet: 1.2M params, ~3ms/frame (fastest)
    - resnet18: 11.7M params, ~20ms/frame (baseline)
    """

    def __init__(
        self,
        backbone: str = CNN_BACKBONE,
        pretrained: bool = CNN_PRETRAINED,
        device: str = "auto"
    ):
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.device_str = device

        # Get config for selected backbone
        if backbone not in CNN_BACKBONE_CONFIGS:
            raise ValueError(f"Unknown backbone: {backbone}. "
                           f"Options: {list(CNN_BACKBONE_CONFIGS.keys())}")

        config = CNN_BACKBONE_CONFIGS[backbone]
        self.feature_dim = config["feature_dim"]
        self.input_size = config["input_size"]

        self.model = None
        self.transform = None
        self.device = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization of PyTorch model."""
        if self._initialized:
            return

        import torch
        import torch.nn as nn
        from torchvision import models, transforms

        # Determine device
        if self.device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_str)

        # Load backbone
        self.model = self._load_backbone()
        self.model = self.model.to(self.device)
        self.model.eval()

        # Image preprocessing (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self._initialized = True

    def _load_backbone(self):
        """Load and prepare backbone model."""
        import torch.nn as nn
        from torchvision import models

        if self.backbone_name == "mobilenet_v3_small":
            # MobileNetV3-Small: Fast, good accuracy
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if self.pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
            model = nn.Sequential(
                base.features,
                base.avgpool,
                nn.Flatten()
            )

        elif self.backbone_name == "efficientnet_v2_s":
            # EfficientNetV2-S: Best accuracy
            weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if self.pretrained else None
            base = models.efficientnet_v2_s(weights=weights)
            model = nn.Sequential(
                base.features,
                base.avgpool,
                nn.Flatten()
            )

        elif self.backbone_name == "squeezenet":
            # SqueezeNet: Fastest, most compact
            weights = models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if self.pretrained else None
            base = models.squeezenet1_1(weights=weights)
            model = nn.Sequential(
                base.features,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )

        elif self.backbone_name == "resnet18":
            # ResNet18: Baseline
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if self.pretrained else None
            base = models.resnet18(weights=weights)
            model = nn.Sequential(*list(base.children())[:-1], nn.Flatten())

        else:
            raise ValueError(f"Unknown backbone: {self.backbone_name}")

        return model

    def extract_features(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0
    ) -> FrameFeatures:
        """
        Extract features from a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)
            frame_number: Frame number
            timestamp: Timestamp in seconds

        Returns:
            FrameFeatures with extracted feature vector
        """
        self.initialize()
        import torch
        import cv2

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Transform and extract
        input_tensor = self.transform(rgb_frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(input_tensor)

        return FrameFeatures(
            frame_number=frame_number,
            timestamp=timestamp,
            features=features.squeeze().cpu().numpy()
        )

    def extract_features_batch(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a batch of frames (GPU optimized).

        Args:
            frames: List of BGR images

        Returns:
            Feature array of shape (batch_size, feature_dim)
        """
        self.initialize()
        import torch
        import cv2

        # Preprocess all frames
        tensors = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(rgb))

        # Stack and process
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            features = self.model(batch)

        return features.cpu().numpy()

    def extract_from_roi(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        frame_number: int = 0,
        timestamp: float = 0.0,
        context_ratio: float = 0.1
    ) -> FrameFeatures:
        """
        Extract features from a region of interest (person bbox).

        Args:
            frame: Full frame
            bbox: Bounding box (x1, y1, x2, y2)
            frame_number: Frame number
            timestamp: Timestamp
            context_ratio: Ratio to expand bbox for context

        Returns:
            FrameFeatures for the ROI
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Expand bbox for context
        width, height = x2 - x1, y2 - y1
        expand_w = int(width * context_ratio)
        expand_h = int(height * context_ratio)

        x1 = max(0, x1 - expand_w)
        y1 = max(0, y1 - expand_h)
        x2 = min(w, x2 + expand_w)
        y2 = min(h, y2 + expand_h)

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return FrameFeatures(
                frame_number=frame_number,
                timestamp=timestamp,
                features=np.zeros(self.feature_dim)
            )

        return self.extract_features(roi, frame_number, timestamp)

    def get_feature_dim(self) -> int:
        """Get output feature dimension."""
        return self.feature_dim

    def close(self):
        """Release resources."""
        self.model = None
        self.transform = None
        self._initialized = False

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
