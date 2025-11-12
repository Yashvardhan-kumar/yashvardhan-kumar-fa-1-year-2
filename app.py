# app.py — PPE Detection – 100% offline, no yolov5 folder
import os
import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np

# -------------------------------------------------
# Paths
# -------------------------------------------------
ROOT = Path(__file__).parent
WEIGHTS = ROOT / "best.pt"          # <-- your trained model

st.set_page_config(page_title="PPE Detector", layout="centered")
st.title("PPE Detection")
st.write("Upload an image → detect helmets, vests, masks, violations.")

# -------------------------------------------------
# Load model (weights_only=False → safe because you trained it)
# -------------------------------------------------
@st.cache_resource
def load_model():
    if not WEIGHTS.exists():
        st.error("`best.pt` not found! Place it next to `app.py`.")
        st.stop()

    try:
        # ---- PyTorch ≥2.6 safe-globals (optional) ----
        # If you get "Unsupported global" errors, uncomment the next 3 lines:
        # from torch.serialization import add_safe_globals
        # from models.yolo import DetectionModel   # not needed – we bypass it
        # add_safe_globals([DetectionModel])

        ckpt = torch.load(
            WEIGHTS,
            map_location="cpu",
            weights_only=False          # <-- required for YOLOv5 .pt files
        )
    except Exception as e:
        st.error("Failed to load `best.pt`.")
        st.exception(e)
        st.info(
            "Checklist:\n"
            "1. `best.pt` is in the **same folder** as `app.py`\n"
            "2. It was **exported from YOLOv5** (`best.pt`, not ONNX)\n"
            "3. You trained it yourself → `weights_only=False` is safe"
        )
        st.stop()

    # Extract model + class names
    model = ckpt.get("model") or ckpt.get("ema") or ckpt
    names = ckpt.get("names") or getattr(model, "names", None)
    if names is None:
        st.error("Class names not found in `best.pt`.")
        st.stop()

    model.eval()
    st.success("Model loaded from `best.pt`")
    return model, names

MODEL, CLASS_NAMES = load_model()

# -------------------------------------------------
# Preprocess
# -------------------------------------------------
def preprocess(img_pil: Image.Image):
    # Resize + pad to 640×640 (YOLOv5 default)
    old_size = img_pil.size
    ratio = 640 / max(old_size)
    new_size = tuple(int(x * ratio) for x in old_size)
    img = img_pil.resize(new_size, Image.BILINEAR)

    # Pad to square
    delta_w = 640 - new_size[0]
    delta_h = 640 - new_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
    img = Image.fromarray(np.pad(np.array(img), (
        (padding[1], padding[3]), (padding[0], padding[2]), (0, 0)
    ), mode='constant'))

    # To tensor
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)  # 1×C×H×W
    return img, old_size, ratio, padding

# -------------------------------------------------
# Post-process + draw
# -------------------------------------------------
def postprocess(pred, img_pil, old_size, ratio, padding):
    pred = pred[0]  # remove batch dim
    pred = pred[pred[:, 4] > 0.25]  # conf filter
    if len(pred) == 0:
        return img_pil, pd.DataFrame()

    # Scale boxes back
    pad_l, pad_t = padding[0], padding[1]
    boxes = pred[:, :4].cpu().numpy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_l) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_t) / ratio

    # Clip
    w, h = old_size
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

    # DataFrame
    df = pd.DataFrame({
        "xmin": boxes[:, 0], "ymin": boxes[:, 1],
        "xmax": boxes[:, 2], "ymax": boxes[:, 3],
        "confidence": pred[:, 4].cpu().numpy(),
        "class": pred[:, 5].cpu().numpy().astype(int),
        "name": [CLASS_NAMES[int(c)] for c in pred[:, 5]]
    })

    # Draw
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()

    for _, r in df.iterrows():
        draw.rectangle([r.xmin, r.ymin, r.xmax, r.ymax], outline="red", width=3)
        label = f"{r.name} {r.confidence:.2f}"
        draw.text((r.xmin, r.ymin - 25), label, fill="red", font=font)

    return img_pil, df

# -------------------------------------------------
# Inference
# -------------------------------------------------
def detect(img_pil: Image.Image):
    img_tensor, old_size, ratio, padding = preprocess(img_pil)
    with torch.no_grad():
        pred = MODEL(img_tensor)  # list of tensors → take first
    return postprocess(pred, img_pil.copy(), old_size, ratio, padding)

# -------------------------------------------------
# UI
# -------------------------------------------------
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    tmp_path = ROOT / f"tmp_{int(time.time())}.jpg"
    img = Image.open(uploaded).convert("RGB")
    img.save(tmp_path)

    st.image(img, caption="Original", use_container_width=True)

    with st.spinner("Detecting PPE…"):
        ann_img, df = detect(img)

    st.image(ann_img, caption="Detections", use_container_width=True)
    tmp_path.unlink(missing_ok=True)

    if df.empty:
        st.warning("No objects detected.")
    else:
        st.subheader("Detection Details")
        color_map = {
            "Hardhat": "green", "Mask": "green", "Safety Vest": "green",
            "NO-Hardhat": "red", "NO-Mask": "red", "NO-Safety Vest": "red",
        }
        for _, r in df.iterrows():
            c = color_map.get(r["name"], "gray")
            st.markdown(
                f"<span style='color:{c};'><b>{r['name']}</b>: {r['confidence']:.2f}</span>",
                unsafe_allow_html=True
            )
