import _init_paths
import os
import json
import cv2
import numpy as np
import torch

from tracker import get_tracker
import models.models as models
from utils import cxy_wh_2_rect, load_pretrain


def track_first_n_frames(
    json_path,
    image_dir,
    arch="SiamX",
    resume=None,
    tracker_type="base",
    output_dir="./track_output",
    num_frames=10
):
    # ========= 1. 读取 JSON ============
    with open(json_path, 'r') as f:
        box_info = json.load(f)

    # 取前 num_frames 个 key
    keys = sorted(box_info.keys())[:num_frames]

    # 第一帧 bbox
    first_key = keys[0]
    print(f"Using bbox from: {first_key}")

    bbox = box_info[first_key]["bbox"]
    cx, cy, w, h = bbox["cx"], bbox["cy"], bbox["w"], bbox["h"]

    target_pos = np.array([cx, cy])
    target_sz  = np.array([w, h])

    # ========= 2. 加载模型 ============
    net = models.__dict__[arch](align=False)
    net = load_pretrain(net, resume)
    net.eval()
    net.cuda()

    print("Model loaded.")

    # ========= 3. Warmup（必须主线程）===========
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 127, 127).cuda()
        net.template(dummy)

    print("Warmup OK")

    # ========= 4. 创建 tracker ============
    from easydict import EasyDict as edict
    info = edict()
    info.arch = arch
    info.align = False
    tracker = get_tracker(tracker_type, info)

    # ========= 5. 读取第一张图用于初始化 ============
    first_img_path = os.path.join(image_dir, first_key)
    im1 = cv2.imread(first_img_path)
    if im1 is None:
        raise FileNotFoundError("Image not found: " + first_img_path)

    print("Initializing tracker...")
    state = tracker.init(im1, target_pos, target_sz, net)

    # ========= 6. 循环预测后续帧 ============
    os.makedirs(output_dir, exist_ok=True)

    for key in keys:
        img_path = os.path.join(image_dir, key)
        img = cv2.imread(img_path)
        if img is None:
            print("Skip missing img:", img_path)
            continue

        # 第一帧用真实 bbox（画框），后续使用 tracker 的输出
        if key == first_key:
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
        else:
            state = tracker.track(state, img)
            pred_ltrb = cxy_wh_2_rect(state["target_pos"], state["target_sz"])
            x1, y1, w_pred, h_pred = pred_ltrb
            x2, y2 = x1 + w_pred, y1 + h_pred

        # 画框
        draw = img.copy()
        cv2.rectangle(draw, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 3)

        save_path = os.path.join(output_dir, key)
        cv2.imwrite(save_path, draw)

        print("Saved:", save_path)


if __name__ == "__main__":
    ROOT_DIR = os.path.abspath(os.path.join(__file__, "../../"))
    json_path   = os.path.join(ROOT_DIR, "test/0010/label.json")
    image_dir   = os.path.join(ROOT_DIR, "test/0010/image")

    track_first_n_frames(
        json_path=json_path,
        image_dir=image_dir,
        resume=resume,
        output_dir="./track_output",
        num_frames=10
    )
