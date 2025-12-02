import json
import cv2
import os

def json_bbox_to_xyxy(bbox):
    """ Convert cx, cy, w, h to (x1, y1, x2, y2) """
    cx = bbox['cx']
    cy = bbox['cy']
    w = bbox['w']
    h = bbox['h']
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    return x1, y1, x2, y2

def draw_first_bbox(json_path, image_dir, output_dir):
    # 1. 读取 JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    #取前10个key
    keys = list(data.keys())[:10]
    # 3. 取 bbox
    for first_key in keys:
        bbox = data[first_key]["bbox"]

        # 4. 转成左上-右下
        x1, y1, x2, y2 = json_bbox_to_xyxy(bbox)

        # 5. 读取对应图片
        image_path = os.path.join(image_dir, first_key)
        img = cv2.imread(image_path)

        if img is None:
            raise FileNotFoundError("Image not found: " + image_path)

        # 6. 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # 7. 保存
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, first_key)
        cv2.imwrite(save_path, img)

        print("Saved:", save_path)


if __name__ == "__main__":
    # 修改你的路径
    json_path   = "../test/0010/label.json"               # JSON 文件路径
    image_dir   = "../test/0010/image"                  # PNG/JPG 图片所在目录
    output_dir  = "./output"                  # 保存输出目录

    draw_first_bbox(json_path, image_dir, output_dir)
