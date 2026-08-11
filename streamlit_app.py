import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from pillow_heif import register_heif_opener

register_heif_opener()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "khilona_model.keras")
META_PATH = os.path.join(BASE_DIR, "models", "model_meta.json")

SWATCH = {"blue": "#3b82f6", "purple": "#a855f7", "yellow": "#eab308"}
GLOW = {"blue": "rgba(59,130,246,.15)", "purple": "rgba(168,85,247,.15)", "yellow": "rgba(234,179,8,.15)"}


@st.cache_resource
def load_keras_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at {MODEL_PATH}. Run 'python train.py' first.")
        st.stop()
    model = load_model(MODEL_PATH)
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
    else:
        meta = {"classes": ["blue", "purple", "yellow"], "belts": ["A", "C", "B"], "img_size": 64}
    return model, meta


def to_rgb(img: Image.Image):
    """Match train.py's load_rgb: composite transparent PNGs onto white
    instead of letting .convert("RGB") drop alpha and expose black."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def preprocess_image(img: Image.Image, img_size: int):
    img = to_rgb(img).resize((img_size, img_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, meta, arr):
    prediction = model(arr, training=False).numpy()
    pred_idx = int(np.argmax(prediction))
    confidence = float(prediction[0][pred_idx])
    all_confidences = {meta["classes"][i]: float(prediction[0][i]) for i in range(len(meta["classes"]))}
    if confidence < 0.6:
        return {"color": "unknown", "belt": "None", "confidence": confidence, "all_confidences": all_confidences}
    return {
        "color": meta["classes"][pred_idx],
        "belt": meta["belts"][pred_idx],
        "confidence": confidence,
        "all_confidences": all_confidences,
    }


st.set_page_config(page_title="Khilona Color Detector", page_icon="🎨", layout="centered")

st.markdown("""
<style>
:root {
    --bg-card: #12121a;
    --border: #1e1e2e;
    --text-muted: #71717a;
    --radius: 12px;
}
.stApp { background: #0a0a0f; }
[data-testid="stHeader"] { background: transparent; }

.khilona-header {
    padding: 4px 0 20px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.khilona-header h1 {
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0;
}
.khilona-header p {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 4px;
}

.khilona-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
}

.khilona-swatch {
    width: 100%;
    height: 64px;
    border-radius: var(--radius);
    margin-bottom: 16px;
    box-shadow: 0 0 32px 4px var(--glow-color, transparent);
}

.khilona-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 4px;
}
.khilona-color {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 12px;
}
.khilona-belt {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-bottom: 16px;
}
.khilona-belt b { color: #e4e4e7; }

[data-testid="stFileUploaderDropzone"], [data-testid="stCameraInput"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
}

[data-testid="stTabs"] button {
    font-weight: 600;
}

section[data-testid="stSidebar"] {
    background: #0d0d13;
    border-right: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="khilona-header">
    <h1>🎨 Khilona Color Detector</h1>
    <p>Point a camera or upload a photo to classify toy color</p>
</div>
""", unsafe_allow_html=True)

model, meta = load_keras_model()

with st.sidebar:
    st.markdown("#### Model Info")
    st.markdown(f"**Classes** &nbsp; {', '.join(c.capitalize() for c in meta['classes'])}")
    if meta.get("test_accuracy") is not None:
        st.markdown(f"**Test accuracy** &nbsp; {meta['test_accuracy'] * 100:.1f}%")
    st.markdown(f"**Input size** &nbsp; {meta['img_size']}x{meta['img_size']}")

tab_camera, tab_upload = st.tabs(["Camera", "Upload"])

img = None
with tab_camera:
    cam_file = st.camera_input("Take a photo of the toy", label_visibility="collapsed")
    if cam_file is not None:
        img = Image.open(cam_file)

with tab_upload:
    up_file = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "bmp", "heic", "heif"], label_visibility="collapsed"
    )
    if up_file is not None:
        img = Image.open(up_file)

if img is not None:
    img = to_rgb(img)
    arr = preprocess_image(img, meta["img_size"])
    result = predict(model, meta, arr)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image(img, caption="Input", use_container_width=True)

    with col2:
        color = result["color"]
        swatch_color = SWATCH.get(color, "#888888")
        glow_color = GLOW.get(color, "transparent")

        if color == "unknown":
            st.markdown(f"""
            <div class="khilona-card">
                <div class="khilona-swatch" style="background:{swatch_color};--glow-color:{glow_color};"></div>
                <div class="khilona-label">Result</div>
                <div class="khilona-color">Not sure</div>
                <div class="khilona-belt">Confidence too low: <b>{result['confidence']*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="khilona-card">
                <div class="khilona-swatch" style="background:{swatch_color};--glow-color:{glow_color};"></div>
                <div class="khilona-label">Predicted Color</div>
                <div class="khilona-color">{color.capitalize()}</div>
                <div class="khilona-belt">Belt <b>{result['belt']}</b> &nbsp;·&nbsp; Confidence <b>{result['confidence']*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(result["confidence"])

    with st.expander("All class confidences"):
        for cls, conf in sorted(result["all_confidences"].items(), key=lambda x: -x[1]):
            st.markdown(f"**{cls.capitalize()}** &nbsp; {conf * 100:.1f}%")
            st.progress(conf)
else:
    st.info("Take a photo or upload an image to get a prediction.")
