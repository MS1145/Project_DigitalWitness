"""
feature_extractor.py - Loads MobileNetExtractor and converts BGR frames to
                       1280-dim numpy feature vectors.

OpenCV is used for colour space conversion (BGR -> RGB).
    OpenCV: Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal.
    https://opencv.org

torchvision transforms handle resizing and normalisation before the model sees
the frame. The normalisation values below are the standard ImageNet stats used
when the backbone was originally pretrained.
"""
from __future__ import annotations
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms

from core.model_definitions import MobileNetExtractor


class FeatureExtractor:
    """
    Wraps MobileNetExtractor with:
      - checkpoint loading
      - device management
      - per-frame image preprocessing
      - numpy interface (callers never see torch tensors)
    """

    # Standard ImageNet normalisation - these exact values must match what
    # was used during training otherwise features will be on a different scale.
    # Source: torchvision docs / PyTorch model zoo convention
    # https://pytorch.org/vision/stable/models.html
    _TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # MobileNetV2 expects 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        self._device = device
        self._model  = MobileNetExtractor().to(device)
        if checkpoint_path.exists():
            # weights_only=False because our checkpoint includes optimizer state
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            sd   = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self._model.load_state_dict(sd)
        # if the checkpoint is missing we still run - features will be from
        # ImageNet-pretrained weights only, which still work reasonably well
        self._model.eval()

    def extract(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Convert a BGR OpenCV frame to a (1280,) float32 numpy feature vector.

        OpenCV reads frames as BGR so we flip to RGB before passing to the
        model which was trained on RGB images.
        """
        import cv2
        rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        x    = self._TRANSFORM(rgb).unsqueeze(0).to(self._device)
        feat = self._model.extract_features(x).cpu().numpy().flatten()
        return feat

    @property
    def device(self) -> torch.device:
        return self._device
