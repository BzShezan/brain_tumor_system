import os
import cv2
import requests
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

UPLOAD_DIR = "static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

RESULT_IMAGE_NAME = "result.jpg"
RESULT_IMAGE_PATH = os.path.join(UPLOAD_DIR, RESULT_IMAGE_NAME)


def get_final_decision(det_class_name, det_conf, cls_label, cls_conf):
    if cls_label == "notumor" and det_class_name != "notumor":
        return (
            f"Suspicious region detected: YOLO suggests {det_class_name} "
            f"({det_conf:.2f}), but classifier predicts notumor ({cls_conf:.2f})"
        )

    if cls_label == det_class_name:
        return f"Confirmed: {det_class_name} | det={det_conf:.2f}, cls={cls_conf:.2f}"

    return (
        f"Likely {det_class_name} by YOLO ({det_conf:.2f}) / "
        f"classifier suggests {cls_label} ({cls_conf:.2f})"
    )


def draw_messages(image, messages, start_x=10, start_y=30, line_gap=28):
    y = start_y
    for msg in messages:
        cv2.putText(
            image,
            msg,
            (start_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )
        y += line_gap


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if "image" not in request.files:
                return render_template("index.html", error="No image uploaded")

            file = request.files["image"]
            if file.filename == "":
                return render_template("index.html", error="Please select an image")

            upload_path = os.path.join(UPLOAD_DIR, file.filename)
            file.save(upload_path)

            image = cv2.imread(upload_path)
            if image is None:
                return render_template("index.html", error="Uploaded image could not be read")

            # Call Detection API
            with open(upload_path, "rb") as f:
                detect_response = requests.post(
                    "http://127.0.0.1:5001/detect",
                    files={"image": f},
                    timeout=120
                )

            if detect_response.status_code != 200:
                return render_template(
                    "index.html",
                    error=f"Detection API failed: {detect_response.text}"
                )

            detect_data = detect_response.json()
            detections = detect_data.get("detections", [])

            final_messages = []

            # No detection case
            if len(detections) == 0:
                final_messages.append("No tumor detected by YOLO (possible miss – review required)")
                draw_messages(image, final_messages)

            else:
                for idx, det in enumerate(detections, start=1):
                    x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                    det_conf = det["confidence"]
                    det_class = det["class_name"]

                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        final_messages.append(f"#{idx}: invalid crop region")
                        continue

                    crop_path = os.path.join(UPLOAD_DIR, f"crop_{idx}.jpg")
                    cv2.imwrite(crop_path, crop)

                    # Call Classification API
                    with open(crop_path, "rb") as cf:
                        cls_response = requests.post(
                            "http://127.0.0.1:5002/classify",
                            files={"image": cf},
                            timeout=120
                        )

                    if cls_response.status_code != 200:
                        final_messages.append(f"#{idx}: classification failed")
                        continue

                    cls_data = cls_response.json()
                    cls_label = cls_data["class_name"]
                    cls_conf = cls_data["confidence"]

                    final_decision = get_final_decision(det_class, det_conf, cls_label, cls_conf)
                    final_messages.append(f"#{idx}: {final_decision}")

                    # Color logic
                    if cls_label == det_class:
                        color = (0, 255, 0)      # green
                    elif cls_label == "notumor":
                        color = (0, 0, 255)      # red
                    else:
                        color = (0, 165, 255)    # orange

                    # Draw box
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                    short_label = f"YOLO:{det_class} {det_conf:.2f} | ResNet:{cls_label} {cls_conf:.2f}"
                    cv2.putText(
                        image,
                        short_label,
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

                draw_messages(image, final_messages)

            cv2.imwrite(RESULT_IMAGE_PATH, image)

            return render_template(
                "index.html",
                result_image=f"static/uploads/{RESULT_IMAGE_NAME}",
                messages=final_messages
            )

        except requests.exceptions.ConnectionError:
            return render_template(
                "index.html",
                error="Could not connect to Detection/Classify API. Make sure both APIs are running."
            )
        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)