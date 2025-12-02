import argparse
import json
import os
import sys
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # 360VOT/lib
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

import sot.SiamX.lib.models.models as models
from framework_torch import OmniFrameworkTorch
from sot.tools.tracker import get_tracker


class SiamX360Framework:
    """Adapter that runs SiamX inside the 360VOT spherical pipeline.

    The class mirrors the four-stage flow described in the paper:
    1. Convert the BFoV annotation to a distortion-free local patch.
    2. Run SiamX on the patch exactly as in the planar setting.
    3. Interpret the predicted local bounding box as a binary mask.
    4. Reproject the mask back to the panorama for the next frame.
    """

    def __init__(self, arch: str = "SiamX", resume: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fw: Optional[OmniFrameworkTorch] = None

        self.net = models.__dict__[arch](align=False).to(self.device).eval()
        self.net = self._load_ckpt(resume)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 127, 127, device=self.device)
            self.net.template(dummy)

        info = edict()
        info.arch = arch
        info.align = False
        self.sot = get_tracker("base", info)

        self.state: Optional[Dict] = None
        self.last_mask: Optional[torch.Tensor] = None

    def _load_ckpt(self, path: Optional[str]):
        if path is None or not os.path.exists(path):
            raise FileNotFoundError("Checkpoint path is required for SiamX initialization.")

        print("Loading model:", path)
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt, strict=False)
        return self.net

    def _ensure_fw(self, height: int, width: int) -> None:
        if self.fw is None:
            self.fw = OmniFrameworkTorch(
                ori_width=width,
                ori_height=height,
                save_inter=False,
                device=self.device,
            )

    def _extract_sr(self, img: np.ndarray):
        rgb_tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float().to(self.device) / 255.0
        rgb_sr, mask_sr = self.fw.obtain_search_region(rgb_tensor, self.last_mask)

        if rgb_sr.dim() == 4:
            rgb_sr = rgb_sr[0]

        sr_np = (rgb_sr.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        sr_np = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)
        return sr_np, rgb_sr, mask_sr

    def init_with_bfov(self, img: np.ndarray, bfov: Dict[str, float]):
        H, W = img.shape[:2]
        self._ensure_fw(H, W)

        clon = float(bfov["clon"])
        clat = float(bfov["clat"])
        theta = float(bfov["fov_h"])
        phi = float(bfov["fov_v"])
        rot = float(bfov.get("rotation", 0.0))

        rgb_tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float().to(self.device) / 255.0

        rgb_sr, msk_sr = self.fw.obtain_search_region_from_bfov(
            rgb_tensor, clon, clat, theta, phi, rot
        )

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

        self.state = self.sot.init(
            sr_np,
            np.array([cx, cy], np.float32),
            np.array([w, h], np.float32),
            self.net,
        )

        # 保存 pano 空间掩膜，用于后续 track
        self.last_mask = self.fw.bfov_to_mask(clon, clat, theta, phi, rot).to(self.device)

    def track(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        self._ensure_fw(H, W)

        if self.last_mask is None:
            raise RuntimeError("Call init_with_bfov() first!")

        sr_np, rgb_sr, _ = self._extract_sr(img)
        self.state = self.sot.track(self.state, sr_np)

        cx, cy = self.state["target_pos"]
        w, h = self.state["target_sz"]
        if w <= 1 or h <= 1:
            print("⚠ BBox degenerate -> fallback last")
            return self.last_mask.cpu().numpy()

        Hs, Ws = rgb_sr.shape[1:]
        mask_sr = torch.zeros((Hs, Ws), dtype=torch.uint8, device=self.device)
        x1, x2 = int(cx - w / 2), int(cx + w / 2)
        y1, y2 = int(cy - h / 2), int(cy + h / 2)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(Ws, x2), min(Hs, y2)

        mask_sr[y1:y2, x1:x2] = 1
        prd_mask_np, valid = self.fw.reproject_search_region(mask_sr)
        if (not valid) or prd_mask_np.sum() == 0:
            print("⚠ reprojection fail -> fallback")
            return self.last_mask.cpu().numpy()

        prd_mask = torch.from_numpy(prd_mask_np.astype(np.uint8)).to(self.device)
        self.last_mask = prd_mask

        return prd_mask_np.astype(np.uint8)


def run_360_sot(img_dir: str, json_path: str, resume_path: str, output_dir: Optional[str] = None):
    with open(json_path, "r") as f:
        label = json.load(f)

    tracker = SiamX360Framework(resume=resume_path)

    save_dir = output_dir or os.path.join(THIS_DIR, "output")
    os.makedirs(save_dir, exist_ok=True)

    keys = sorted(label.keys())
    first = keys[0]

    img = cv2.imread(os.path.join(img_dir, first))
    if img is None:
        raise FileNotFoundError(f"Cannot read first frame: {first}")
    bfov = label[first]["bfov"]
    tracker.init_with_bfov(img, bfov)

    for k in keys:
        frame = cv2.imread(os.path.join(img_dir, k))
        if frame is None:
            print(f"⚠ Skip missing frame: {k}")
            continue
        mask = tracker.track(frame)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            print(f"⚠ Empty prediction on frame: {k}")
            continue
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        save_path = os.path.join(save_dir, k)
        cv2.imwrite(save_path, frame)
        print("[Save]", save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SiamX inside the 360VOT pipeline")
    parser.add_argument("--img_dir", required=False, default=os.path.join(THIS_DIR, "0010/image"), help="Path to frame folder")
    parser.add_argument("--json", required=False, default=os.path.join(THIS_DIR, "0010/label.json"), help="Path to label json")
    parser.add_argument(
        "--resume", required=False, default=os.path.join(THIS_DIR, "SiamX.pth"), help="Checkpoint for SiamX"
    )
    parser.add_argument("--output", required=False, default=os.path.join(THIS_DIR, "output"), help="Save directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_360_sot(args.img_dir, args.json, args.resume, args.output)
