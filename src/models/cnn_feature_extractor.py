"""
CNN-based spatial feature extraction for Digital Witness.

Uses a pretrained CNN backbone (ResNet18 by default) to extract
spatial features from video frames. These features capture visual
appearance and spatial relationships that complement pose estimation.

Why ResNet18?
-------------
- Pretrained on ImageNet: Already knows how to recognize objects, textures, shapes
- 512-dimensional output: Rich but compact representation
- Fast inference: ~5ms per frame on GPU, enabling near real-time processing
- Transfer learning: Retail/surveillance domain shares visual primitives with ImageNet

Feature Extraction Process:
---------------------------
1. Resize frame to 224x224 (ImageNet standard)
2. Normalize using ImageNet mean/std
3. Pass through ResNet18 (without final classification layer)
4. Output: 512-dimensional feature vector capturing visual semantics

These features become input to the LSTM for temporal modeling.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from pathlib import Path

from ..config import (
    CNN_BACKBONE,      # "resnet18" by default
    CNN_FEATURE_DIM,   # 512 dimensions
    CNN_PRETRAINED,    # True - use ImageNet weights
    CNN_INPUT_SIZE     # (224, 224) pixels
)


@dataclass
class FrameFeatures:
    """Extracted features for a single frame."""
    frame_number: int
    timestamp: float
    features: np.ndarray          # Feature vector
    roi_features: Optional[np.ndarray] = None  # Features from ROI if provided


@dataclass
class SequenceFeatures:
    """Features for a sequence of frames (for LSTM input)."""
    sequence_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    features: np.ndarray          # Shape: (seq_len, feature_dim)
    frame_count: int


class CNNFeatureExtractor:
    """
    CNN-based feature extractor using pretrained backbone.

    Extracts spatial features from video frames that capture:
    - Visual appearance
    - Object presence
    - Spatial layout
    - Scene context

    These features complement pose-based features for behavior analysis.
    """

    def __init__(
        self,
        backbone: str = CNN_BACKBONE,
        pretrained: bool = CNN_PRETRAINED,
        feature_dim: int = CNN_FEATURE_DIM,
        input_size: Tuple[int, int] = CNN_INPUT_SIZE,
        device: str = "auto"
    ):
        """
        Initialize CNN feature extractor.

        Args:
            backbone: CNN architecture ("resnet18", "resnet34", "mobilenet_v2")
            pretrained: Use ImageNet pretrained weights
            feature_dim: Output feature dimension
            input_size: Input image size (height, width)
            device: Device for inference ("auto", "cpu", "cuda")
        """
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.feature_dim = feature_dim
        self.input_size = input_size
        self.device_str = device

        self.model = None
        self.transform = None
        self.device = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization of PyTorch model."""
        if self._initialized:
            return

        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            # Determine device
            if self.device_str == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_str)

            # Load backbone
            if self.backbone_name == "resnet18":
                base_model = models.resnet18(pretrained=self.pretrained)
                base_features = 512
            elif self.backbone_name == "resnet34":
                base_model = models.resnet34(pretrained=self.pretrained)
                base_features = 512
            elif self.backbone_name == "mobilenet_v2":
                base_model = models.mobilenet_v2(pretrained=self.pretrained)
                base_features = 1280
            else:
                raise ValueError(f"Unsupported backbone: {self.backbone_name}")

            # Remove classification head
            if "resnet" in self.backbone_name:
                self.model = nn.Sequential(*list(base_model.children())[:-1])
            else:
                self.model = nn.Sequential(*list(base_model.children())[:-1])

            # Add feature projection if needed
            if base_features != self.feature_dim:
                self.model = nn.Sequential(
                    self.model,
                    nn.Flatten(),
                    nn.Linear(base_features, self.feature_dim)
                )
            else:
                self.model = nn.Sequential(
                    self.model,
                    nn.Flatten()
                )

            self.model = self.model.to(self.device)
            self.model.eval()

            # Image preprocessing
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

        except ImportError:
            raise ImportError(
                "PyTorch not installed. Run: pip install torch torchvision"
            )

    def extract_features(
        self,
        frame: np.ndarray,
        frame_number: int = 0,
        timestamp: float = 0.0,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> FrameFeatures:
        """
        Extract features from a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3)
            frame_number: Frame number
            timestamp: Timestamp in seconds
            roi: Optional region of interest (x1, y1, x2, y2)

        Returns:
            FrameFeatures with extracted feature vector
        """
        self.initialize()

        import torch

        # Convert BGR to RGB
        rgb_frame = frame[:, :, ::-1].copy()

        # Extract full frame features
        features = self._extract_from_image(rgb_frame)

        # Extract ROI features if provided
        roi_features = None
        if roi is not None:
            x1, y1, x2, y2 = roi
            roi_image = rgb_frame[y1:y2, x1:x2]
            if roi_image.size > 0:
                roi_features = self._extract_from_image(roi_image)

        return FrameFeatures(
            frame_number=frame_number,
            timestamp=timestamp,
            features=features,
            roi_features=roi_features
        )

    def _extract_from_image(self, image: np.ndarray) -> np.ndarray:
        """Extract features from an image."""
        import torch

        # Preprocess
        input_tensor = self.transform(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model(input_batch)

        return features.cpu().numpy().flatten()

    def extract_from_detection(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        frame_number: int = 0,
        timestamp: float = 0.0,
        context_ratio: float = 0.2
    ) -> FrameFeatures:
        """
        Extract features from a detected object region.

        Args:
            frame: Full frame as numpy array
            bbox: Detection bounding box (x1, y1, x2, y2)
            frame_number: Frame number
            timestamp: Timestamp
            context_ratio: Ratio to expand bbox for context

        Returns:
            FrameFeatures for the detection
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Expand bbox for context
        width = x2 - x1
        height = y2 - y1
        expand_w = int(width * context_ratio)
        expand_h = int(height * context_ratio)

        x1 = max(0, x1 - expand_w)
        y1 = max(0, y1 - expand_h)
        x2 = min(w, x2 + expand_w)
        y2 = min(h, y2 + expand_h)

        # Crop region
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            # Return zero features for empty ROI
            return FrameFeatures(
                frame_number=frame_number,
                timestamp=timestamp,
                features=np.zeros(self.feature_dim),
                roi_features=None
            )

        # Extract features
        rgb_roi = roi[:, :, ::-1].copy()
        features = self._extract_from_image(rgb_roi)

        return FrameFeatures(
            frame_number=frame_number,
            timestamp=timestamp,
            features=features,
            roi_features=None
        )

    def extract_sequence_features(
        self,
        frames: List[np.ndarray],
        start_frame: int = 0,
        fps: float = 30.0,
        sequence_id: str = ""
    ) -> SequenceFeatures:
        """
        Extract features from a sequence of frames for LSTM input.

        Args:
            frames: List of frames as numpy arrays
            start_frame: Starting frame number
            fps: Frames per second
            sequence_id: Identifier for this sequence

        Returns:
            SequenceFeatures with shape (seq_len, feature_dim)
        """
        self.initialize()

        import torch

        features_list = []

        for i, frame in enumerate(frames):
            frame_features = self.extract_features(
                frame,
                frame_number=start_frame + i,
                timestamp=(start_frame + i) / fps
            )
            features_list.append(frame_features.features)

        # Stack features
        features_array = np.vstack(features_list)

        return SequenceFeatures(
            sequence_id=sequence_id,
            start_frame=start_frame,
            end_frame=start_frame + len(frames) - 1,
            start_time=start_frame / fps,
            end_time=(start_frame + len(frames) - 1) / fps,
            features=features_array,
            frame_count=len(frames)
        )

    def extract_sliding_window_features(
        self,
        frames: List[np.ndarray],
        window_size: int = 30,
        stride: int = 15,
        fps: float = 30.0
    ) -> List[SequenceFeatures]:
        """
        Extract features using sliding windows.

        Args:
            frames: List of all frames
            window_size: Frames per window
            stride: Frames to skip between windows
            fps: Frames per second

        Returns:
            List of SequenceFeatures, one per window
        """
        sequences = []
        num_frames = len(frames)

        window_idx = 0
        for start in range(0, num_frames - window_size + 1, stride):
            window_frames = frames[start:start + window_size]

            seq_features = self.extract_sequence_features(
                window_frames,
                start_frame=start,
                fps=fps,
                sequence_id=f"window_{window_idx}"
            )
            sequences.append(seq_features)
            window_idx += 1

        return sequences

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
