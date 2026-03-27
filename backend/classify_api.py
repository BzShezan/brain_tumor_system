import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import transforms, models

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESNET_MODEL_PATH = "outputs/brain_tumor_resnet18.pth"
UPLOAD_DIR = "static/uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

class_names = ["glioma", "meningioma", "notumor", "pituitary"]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def load_resnet_model(model_path, num_classes=4):
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, num_classes)
    )

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# Load once
model = load_resnet_model(RESNET_MODEL_PATH)


@app.route("/classify", methods=["POST"])
def classify():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No cropped image provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        temp_path = os.path.join(UPLOAD_DIR, "temp_crop.jpg")
        file.save(temp_path)

        crop = cv2.imread(temp_path)
        if crop is None:
            return jsonify({"error": "Could not read crop image"}), 400

        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(crop_pil).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        return jsonify({
            "class_name": class_names[pred.item()],
            "confidence": float(conf.item())
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "service": "ResNet Classification API",
        "status": "running"
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=True)