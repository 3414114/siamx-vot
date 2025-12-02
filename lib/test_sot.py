import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # 360VOT/lib
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from sot.tools.tracker import get_tracker
import sot.SiamX.lib.models.models as models
import cv2
from framework_torch import OmniFrameworkTorch
from easydict import EasyDict as edict
import torch
import numpy as np
import json


class SiamX_360Framework:
    def __init__(self, arch="SiamX", resume=None):
        self.device = "cuda"
        self.fw = None          

        self.net = models.__dict__[arch](align=False).cuda().eval()
        self.net = self._load_ckpt(resume)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 127, 127).cuda()
            self.net.template(dummy)

        info = edict()
        info.arch = arch
        info.align = False
        self.sot = get_tracker("base", info)

        self.state = None
        self.last_mask = None 

    def _load_ckpt(self, path):
        print("Loading model:", path)
        ckpt = torch.load(path, map_location="cuda")
        self.net.load_state_dict(ckpt, strict=False)
        return self.net

    def _ensure_fw(self, H, W):
        if self.fw is None:
            self.fw = OmniFrameworkTorch(
                ori_width=W,
                ori_height=H,
                save_inter=False,
                device=self.device
            )

    def init_with_bfov(self, img, bfov):
        H, W = img.shape[:2]
        self._ensure_fw(H, W)

        clon = float(bfov["clon"])
        clat = float(bfov["clat"])
        theta = float(bfov["fov_h"])
        phi   = float(bfov["fov_v"])
        rot   = float(bfov.get("rotation", 0.0))

        rgb_tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float().to(self.device) / 255.0

        rgb_sr, msk_sr = self.fw.obtain_search_region_from_bfov(
            rgb_tensor, clon, clat, theta, phi, rot
        )

        if rgb_sr.dim() == 4:
            rgb_sr = rgb_sr[0]
        sr_np = (rgb_sr.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        sr_np = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)

        ys, xs = torch.where(msk_sr > 0)
        if len(xs) == 0:
            raise RuntimeError("BFoV search region empty, check BFoV format!")

        x1, x2 = xs.min().item(), xs.max().item()
        y1, y2 = ys.min().item(), ys.max().item()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w  = x2 - x1
        h  = y2 - y1

        self.state = self.sot.init(sr_np, np.array([cx,cy],np.float32),
                                  np.array([w,h],np.float32), self.net)

        # 保存 pano 空间掩膜，用于后续 track
        self.last_mask = self.fw.bfov_to_mask(clon, clat, theta, phi, rot).to(self.device)


    def track(self, img):
        H, W = img.shape[:2]
        self._ensure_fw(H, W)

        if self.last_mask is None:
            raise RuntimeError("Call init_with_bfov() first!")

        rgb_tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float().to(self.device) / 255.0
        rgb_sr, _ = self.fw.obtain_search_region(rgb_tensor, self.last_mask)

        if rgb_sr.dim() == 4:
            rgb_sr = rgb_sr[0]
        sr_np = (rgb_sr.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        sr_np = cv2.cvtColor(sr_np, cv2.COLOR_RGB2BGR)

        self.state = self.sot.track(self.state, sr_np)

        cx, cy = self.state["target_pos"]
        w,  h  = self.state["target_sz"]
        if w<=1 or h<=1:
            print("⚠ BBox degenerate -> fallback last")
            return self.last_mask.cpu().numpy()

        Hs, Ws = rgb_sr.shape[1:]
        mask_sr = torch.zeros((Hs, Ws),dtype=torch.uint8,device=self.device)
        x1,x2 = int(cx-w/2), int(cx+w/2)
        y1,y2 = int(cy-h/2), int(cy+h/2)
        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(Ws,x2),min(Hs,y2)

        mask_sr[y1:y2,x1:x2]=1
        prd_mask_np, valid = self.fw.reproject_search_region(mask_sr)
        if (not valid) or prd_mask_np.sum()==0:
            print("⚠ reprojection fail -> fallback")
            return self.last_mask.cpu().numpy()

        prd_mask = torch.from_numpy(prd_mask_np.astype(np.uint8)).to(self.device)
        self.last_mask = prd_mask

        return prd_mask_np.astype(np.uint8)



def run_360_sot(img_dir, json_path, resume_path):
    with open(json_path,'r') as f:
        label = json.load(f)

    tracker = SiamX_360Framework(resume=resume_path)

    os.makedirs(os.path.join(THIS_DIR,"output"),exist_ok=True)

    keys = sorted(label.keys())
    first = keys[0]

    img = cv2.imread(os.path.join(img_dir, first))
    bfov = label[first]["bfov"]
    tracker.init_with_bfov(img, bfov)

    for k in keys:
        frame = cv2.imread(os.path.join(img_dir,k))
        mask = tracker.track(frame)

        ys,xs = np.where(mask>0)
        if len(xs)==0: continue
        x1,x2,y1,y2=xs.min(),xs.max(),ys.min(),ys.max()
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        save_path = os.path.join(THIS_DIR,"output",k)
        cv2.imwrite(save_path,frame)
        print("[Save]", save_path)



if __name__ == "__main__":
    json_path = os.path.join(THIS_DIR,"0010/label.json")
    image_dir = os.path.join(THIS_DIR,"0010/image")
    resume = os.path.join(THIS_DIR,"SiamX.pth")
    run_360_sot(image_dir,json_path,resume)