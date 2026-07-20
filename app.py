import os
import io
import json
import base64
import numpy as np
from PIL import Image
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.models import load_model
from pillow_heif import register_heif_opener

register_heif_opener()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "khilona_model.keras")
META_PATH = os.path.join(BASE_DIR, "models", "model_meta.json")

model = None
meta = None


def load_keras_model():
    global model, meta
    if model is not None:
        return
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run 'python train.py' first."
        )
    model = load_model(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
    else:
        meta = {
            "classes": ["blue", "purple", "yellow"],
            "belts": ["A", "B", "C"],
            "img_size": 64,
        }
    print("Model ready for inference.")


def preprocess_image(img: Image.Image):
    img_size = meta["img_size"]
    img = img.convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(arr):
    prediction = model.predict(arr, verbose=0)
    pred_idx = int(np.argmax(prediction))
    confidence = float(prediction[0][pred_idx])
    all_confidences = {
        meta["classes"][i]: float(prediction[0][i])
        for i in range(len(meta["classes"]))
    }
    if confidence < 0.6:
        return {
            "color": "unknown",
            "belt": "None",
            "confidence": confidence,
            "all_confidences": all_confidences,
        }
    return {
        "color": meta["classes"][pred_idx],
        "belt": meta["belts"][pred_idx],
        "confidence": confidence,
        "all_confidences": all_confidences,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    load_keras_model()

    if "image" in request.files:
        file = request.files["image"]
        img = Image.open(file.stream)
    elif request.is_json and "image_base64" in request.get_json():
        data = request.get_json()
        img_bytes = base64.b64decode(data["image_base64"].split(",")[-1])
        img = Image.open(io.BytesIO(img_bytes))
    else:
        return jsonify({"error": "No image provided. Send 'image' file or 'image_base64'."}), 400

    arr = preprocess_image(img)
    result = predict(arr)

    img_buffer = io.BytesIO()
    img.thumbnail((300, 300))
    img.save(img_buffer, format="JPEG", quality=85)
    img_b64 = base64.b64encode(img_buffer.getvalue()).decode()

    result["preview"] = f"data:image/jpeg;base64,{img_b64}"
    return jsonify(result)


@app.route("/api/predict_frame", methods=["POST"])
def api_predict_frame():
    load_keras_model()

    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "No frame data"}), 400

    img_bytes = base64.b64decode(data["frame"].split(",")[-1])
    img = Image.open(io.BytesIO(img_bytes))
    arr = preprocess_image(img)
    result = predict(arr)
    return jsonify(result)


@app.route("/api/model_info")
def api_model_info():
    load_keras_model()
    return jsonify({
        "classes": meta["classes"],
        "belts": meta["belts"],
        "img_size": meta["img_size"],
        "test_accuracy": meta.get("test_accuracy", None),
    })


if __name__ == "__main__":
    load_keras_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
