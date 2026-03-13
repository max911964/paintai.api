from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
import requests as req
from io import BytesIO
from scipy.ndimage import distance_transform_edt

app = Flask(__name__)
CORS(app)

def process_painting(image_url, difficulty):
    config = {
        "easy":   {"k": 8,  "blur": 3, "min_region": 300, "max_w": 900},
        "medium": {"k": 14, "blur": 2, "min_region": 150, "max_w": 1100},
        "hard":   {"k": 20, "blur": 1, "min_region": 60,  "max_w": 1300},
    }
    cfg = config.get(difficulty, config["medium"])

    r = req.get(image_url, timeout=15)
    img = Image.open(BytesIO(r.content)).convert("RGB")
    ratio = cfg["max_w"] / img.width
    img = img.resize((cfg["max_w"], int(img.height * ratio)), Image.LANCZOS)
    img_np = np.array(img)

    filtered = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(filtered, cv2.COLOR_RGB2LAB)
    h, w = lab.shape[:2]
    pixels = lab.reshape(-1, 3).astype(np.float32)

    kmeans = MiniBatchKMeans(n_clusters=cfg["k"], random_state=42, n_init=3)
    labels = kmeans.fit_predict(pixels)
    centers_lab = kmeans.cluster_centers_

    centers_rgb = []
    for c in centers_lab:
        lab_pixel = np.uint8([[c]])
        rgb = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)[0][0]
        centers_rgb.append(rgb.tolist())

    label_map = labels.reshape(h, w).astype(np.int32)
    final_label_map = label_map.copy()

    regions = []
    for color_idx in range(cfg["k"]):
        mask = (label_map == color_idx).astype(np.uint8)
        num_labels, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        for region_id in range(1, num_labels):
            area = stats[region_id, cv2.CC_STAT_AREA]
            if area < cfg["min_region"]:
                final_label_map[cc_labels == region_id] = -1
            else:
                cx, cy = int(centroids[region_id][0]), int(centroids[region_id][1])
                regions.append({
                    "id": f"{color_idx}_{region_id}",
                    "colorIndex": color_idx,
                    "centroid": [cx, cy],
                    "area": int(area)
                })

    orphan_mask = (final_label_map == -1).astype(np.uint8)
    if orphan_mask.sum() > 0:
        dist, idx = distance_transform_edt(orphan_mask, return_indices=True)
        final_label_map[orphan_mask == 1] = final_label_map[
            idx[0][orphan_mask == 1], idx[1][orphan_mask == 1]
        ]

    boundary = np.zeros((h, w), dtype=np.uint8)
    boundary[:-1, :] |= (final_label_map[:-1, :] != final_label_map[1:, :]).astype(np.uint8)
    boundary[1:, :]  |= (final_label_map[:-1, :] != final_label_map[1:, :]).astype(np.uint8)
    boundary[:, :-1] |= (final_label_map[:, :-1] != final_label_map[:, 1:]).astype(np.uint8)
    boundary[:, 1:]  |= (final_label_map[:, :-1] != final_label_map[:, 1:]).astype(np.uint8)

    palette = []
    for i, rgb in enumerate(centers_rgb):
        palette.append({
            "index": i,
            "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
            "r": rgb[0], "g": rgb[1], "b": rgb[2]
        })

    return {
        "width": w,
        "height": h,
        "labelMap": final_label_map.tolist(),
        "boundaryMap": boundary.tolist(),
        "regions": regions,
        "palette": palette
    }

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    result = process_painting(data['imageUrl'], data['difficulty'])
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
