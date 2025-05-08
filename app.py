import os
import json
import cv2
import torch
import numpy as np
import gdown
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

import gdown_download 

# Flask setup
app = Flask(__name__)

# Direktori upload
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "model_final.pth"
ANNOTATION_PATH = "annotations_coco_resized.json"

# ✅ Load anotasi
with open(ANNOTATION_PATH, "r") as f:
    annotations = json.load(f)

image_sizes = {img["file_name"]: (img["width"], img["height"]) for img in annotations["images"]}

# ✅ Konfigurasi Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = MODEL_PATH
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.DEVICE = "cpu"  # Ubah ke "cuda" jika pakai GPU

MetadataCatalog.get("tuna_dataset").thing_classes = ["ikan_tuna_segar", "ikan_tuna_tidak_segar"]
class_labels = MetadataCatalog.get("tuna_dataset").thing_classes

predictor = DefaultPredictor(cfg)

# ✅ Resize image
def resize_image(image_path):
    if not image_sizes:
        return image_path
    target_width, target_height = list(image_sizes.values())[0]
    image = cv2.imread(image_path)
    if image is None:
        return None
    resized = cv2.resize(image, (target_width, target_height))
    resized_path = os.path.join(UPLOAD_FOLDER, "resized_" + os.path.basename(image_path))
    cv2.imwrite(resized_path, resized)
    return resized_path

@app.route("/")
def home():
    return "<h1>API Deteksi Ikan Tuna (Render/Railway)</h1>"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    resized_path = resize_image(filepath)
    if resized_path is None:
        return jsonify({"error": "Failed to process image"}), 400

    image = cv2.imread(resized_path)
    outputs = predictor(image)

    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    pred_scores = outputs["instances"].scores.cpu().numpy()

    if len(pred_classes) > 0:
        best_idx = np.argmax(pred_scores)
        results = [{
            "label": class_labels[pred_classes[best_idx]],
            "confidence": round(float(pred_scores[best_idx]) * 100, 2)
        }]
    else:
        results = []

    os.remove(filepath)
    os.remove(resized_path)
    return jsonify({"predictions": results, "message": "Deteksi selesai"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
