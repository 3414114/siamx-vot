"""Adapters for plugging arbitrary planar SOT trackers into the spherical pipeline."""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

import sot.SiamX.lib.models.models as models
from sot.tools.tracker import get_tracker


class LocalTrackerAdapter:
    """Minimal protocol that planar trackers should follow."""

    def initialize(self, image: np.ndarray, bbox: np.ndarray) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def track(self, image: np.ndarray) -> Tuple[float, float, float, float]:  # pragma: no cover - interface
        raise NotImplementedError


class SiamXAdapter(LocalTrackerAdapter):
    """Wrap SiamX so that it can run inside the tessellated SOT pipeline."""

    def __init__(self, arch: str = "SiamX", resume: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = models.__dict__[arch](align=False).to(self.device).eval()
        self.net = self._load_ckpt(resume)

        # Prime template branch to avoid extra latency on the first frame.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 127, 127, device=self.device)
            self.net.template(dummy)

        info = {
            "arch": arch,
            "align": False,
        }
        self.tracker = get_tracker("base", info)
        self.state: Optional[Dict] = None

    def _load_ckpt(self, path: Optional[str]):
        if path is None or not os.path.exists(path):
            raise FileNotFoundError("Checkpoint path is required for SiamX initialization.")

        print("Loading model:", path)
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt, strict=False)
        return self.net

    def initialize(self, image: np.ndarray, bbox: np.ndarray) -> None:
        cx, cy, w, h = bbox.astype(np.float32)
        self.state = self.tracker.init(image, np.array([cx, cy], np.float32), np.array([w, h], np.float32), self.net)

    def track(self, image: np.ndarray) -> Tuple[float, float, float, float]:
        if self.state is None:
            raise RuntimeError("Call initialize() before track().")

        self.state = self.tracker.track(self.state, image)
        cx, cy = self.state["target_pos"]
        w, h = self.state["target_sz"]
        return float(cx), float(cy), float(w), float(h)
