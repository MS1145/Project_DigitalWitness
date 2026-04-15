"""
feature_extractor.py — Loads MobileNetExtractor and converts BGR frames to
                        1280-dim numpy feature vectors.
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

    _TRANSFORM = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        self._device = device
        self._model  = MobileNetExtractor().to(device)
        if checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            sd   = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self._model.load_state_dict(sd)
        self._model.eval()

    def extract(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Convert a BGR OpenCV frame to a (1280,) float32 numpy feature vector.
        """
        import cv2
        rgb  = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        x    = self._TRANSFORM(rgb).unsqueeze(0).to(self._device)
        feat = self._model.extract_features(x).cpu().numpy().flatten()
        return feat

    @property
    def device(self) -> torch.device:
        return self._device
