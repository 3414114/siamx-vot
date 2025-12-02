import argparse
import json
import os
import sys
from typing import Dict, Optional

import cv2
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # 360VOT/lib
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from tess_sot import SiamXAdapter, TessSOTPipeline


class SiamX360Framework:
    """Adapter that runs SiamX inside the 360VOT spherical pipeline.

    The helper exposes the four-stage flow from the diagram: convert the
    BFoV to a search region, run the local tracker, interpret predictions as
    a mask, and project the result back to the panorama.
    """

    def __init__(self, arch: str = "SiamX", resume: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tracker = SiamXAdapter(arch=arch, resume=resume, device=self.device)
        self.pipeline = TessSOTPipeline(tracker=tracker, device=self.device)

    def init_with_bfov(self, img: np.ndarray, bfov: Dict[str, float]):
        self.pipeline.initialize(img, bfov)

    def track(self, img: np.ndarray) -> np.ndarray:
        result = self.pipeline.track(img)
        return result.mask


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
