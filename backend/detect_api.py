import os
import cv2
from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

YOLO_MODEL_PATH = "runs/detect/outputs/yolo_brain_tumor/weights/best.pt"
UPLOAD_DIR = "static/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load YOLO once when server starts
model = YOLO(YOLO_MODEL_PATH)


@app.route("/detect", methods=["POST"])
def detect():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        image_path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(image_path)

        image = cv2.imread(image_path)
        if image is None:
            return jsonify({"error": "Could not read uploaded image"}), 400

        results = model(image, conf=0.25)

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = result.names[cls_id]

                detections.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name
                })

        return jsonify({
            "image_path": image_path.replace("\\", "/"),
            "detections": detections
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "service": "YOLO Detection API",
        "status": "running"
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True)