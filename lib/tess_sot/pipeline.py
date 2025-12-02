from __future__ import annotations

"""Tessellated SOT pipeline mirroring the spherical flow diagram."""

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from torchvision.ops import masks_to_boxes

from framework_torch import OmniFrameworkTorch
from .adapters import LocalTrackerAdapter


@dataclass
class TessSOTOutput:
    """Container for spherical tracking outputs."""

    mask: np.ndarray
    bbox: Optional[np.ndarray]
    bfov: Optional[Dict[str, float]]


class TessSOTPipeline:
    """Run a planar tracker on locally sampled search regions and lift predictions back."""

    def __init__(
        self,
        tracker: LocalTrackerAdapter,
        device: Optional[str] = None,
        framework: Optional[OmniFrameworkTorch] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.framework = framework or OmniFrameworkTorch(device=self.device)
        self.tracker = tracker

        self.last_mask: Optional[torch.Tensor] = None

    def _ensure_framework(self, height: int, width: int) -> None:
        if self.framework is None:
            self.framework = OmniFrameworkTorch(ori_width=width, ori_height=height, device=self.device)

    @staticmethod
    def _mask_to_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
        mask_tensor = torch.from_numpy(mask.astype(np.uint8))
        if mask_tensor.sum() <= 0:
            return None
        boxes = masks_to_boxes(mask_tensor.unsqueeze(0)).numpy()[0]
        x1, y1, x2, y2 = boxes
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    def _mask_from_bbox(self, bbox: np.ndarray, shape: torch.Size) -> torch.Tensor:
        _, h, w = shape
        cx, cy, bw, bh = bbox
        x1, x2 = int(cx - bw / 2), int(cx + bw / 2)
        y1, y2 = int(cy - bh / 2), int(cy + bh / 2)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        mask = torch.zeros((h, w), dtype=torch.uint8, device=self.device)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
        return mask

    def _extract_search_region(self, img: np.ndarray):
        rgb_tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float().to(self.device) / 255.0
        rgb_sr, _ = self.framework.obtain_search_region(rgb_tensor, self.last_mask)

        if rgb_sr.dim() == 4:
            rgb_sr = rgb_sr[0]

        sr_np = (rgb_sr.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        sr_np = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
        return sr_np, rgb_sr

    def initialize(self, img: np.ndarray, bfov: Dict[str, float]):
        H, W = img.shape[:2]
        self._ensure_framework(H, W)

        clon = float(bfov["clon"])
        clat = float(bfov["clat"])
        theta = float(bfov["fov_h"])
        phi = float(bfov["fov_v"])
        rot = float(bfov.get("rotation", 0.0))

        rgb_tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float().to(self.device) / 255.0
        rgb_sr, msk_sr = self.framework.obtain_search_region_from_bfov(rgb_tensor, clon, clat, theta, phi, rot)

        if rgb_sr.dim() == 4:
            rgb_sr = rgb_sr[0]
        sr_np = (rgb_sr.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        sr_np = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)

        ys, xs = torch.where(msk_sr > 0)
        if len(xs) == 0:
            raise RuntimeError("BFoV search region empty, check BFoV format!")

        x1, x2 = xs.min().item(), xs.max().item()
        y1, y2 = ys.min().item(), ys.max().item()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1

        self.tracker.initialize(sr_np, np.array([cx, cy, w, h], dtype=np.float32))
        self.last_mask = self.framework.bfov_to_mask(clon, clat, theta, phi, rot).to(self.device)

    def track(self, img: np.ndarray) -> TessSOTOutput:
        H, W = img.shape[:2]
        self._ensure_framework(H, W)

        if self.last_mask is None:
            raise RuntimeError("Call initialize() with an initial BFoV before tracking.")

        sr_np, rgb_sr = self._extract_search_region(img)
        cx, cy, w, h = self.tracker.track(sr_np)

        mask_sr = self._mask_from_bbox(np.array([cx, cy, w, h], dtype=np.float32), rgb_sr.shape)
        prd_mask_np, valid = self.framework.reproject_search_region(mask_sr)
        if (not valid) or prd_mask_np.sum() == 0:
            print("⚠ reprojection fail -> fallback to last mask")
            prd_mask_np = self.last_mask.cpu().numpy()
        else:
            self.last_mask = torch.from_numpy(prd_mask_np.astype(np.uint8)).to(self.device)

        bbox = self._mask_to_bbox(prd_mask_np)
        bfov_obj = self.framework.mask2Bfov(self.last_mask, need_rotation=False)
        bfov_dict = None
        if bfov_obj is not None:
            bfov_dict = {
                "clon": float(bfov_obj.clon),
                "clat": float(bfov_obj.clat),
                "fov_h": float(bfov_obj.fov_h),
                "fov_v": float(bfov_obj.fov_v),
                "rotation": float(bfov_obj.rotation),
            }

        return TessSOTOutput(mask=prd_mask_np.astype(np.uint8), bbox=bbox, bfov=bfov_dict)
