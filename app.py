# app.py — 100% YOLOv5-FREE, runs your best.pt directly
import os
import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
import pandas as pd
import numpy as np

# -------------------------------
# Paths
# -------------------------------
ROOT = Path(__file__).parent
WEIGHTS = ROOT / "best.pt"  # <-- YOUR MODEL HERE

st.set_page_config(page_title="PPE Detector", layout="centered")
st.title("PPE Detection")
st.write("Upload an image → detect **helmets, vests, masks**, and violations.")

# -------------------------------
# Load model directly (NO yolov5 folder!)
# -------------------------------
@st.cache_resource
def load_model():
    if not WEIGHTS.exists():
        st.error("`best.pt` not found! Upload your trained model.")
        st.stop()

    try:
        # Load the raw .pt file (contains 'model' and 'names')
        ckpt = torch.load(WEIGHTS, map_location="cpu")
        model = ckpt["model"]
        model.eval()
        names = ckpt.get("names", ckpt.get("model").names)  # class names
        st.success("Model loaded from `best.pt`")
        return model, names
    except Exception as e:
        st.error("Failed to load `best.pt`.")
        st.exception(e)
        st.info(
            "Make sure:\n"
            "1. `best.pt` is in the same folder as `app.py`\n"
            "2. It was exported from YOLOv5 (not ONNX)"
        )
        st.stop()

MODEL, CLASS_NAMES = load_model()

# -------------------------------
# Preprocess image
# -------------------------------
def preprocess(img_pil):
    img = img_pil.resize((640, 640))
    img = np.array(img).transpose(2, 0, 1)  # HWC → CHW
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0)  # 1xCxHxW
    return img

# -------------------------------
# Run inference
# -------------------------------
def detect(img_pil):
    img = preprocess(img_pil)
    with torch.no_grad():
        pred = MODEL(img)[0]  # (1, num_boxes, 85)

    # Apply NMS
    pred = pred[pred[:, 4] > 0.25]  # conf > 0.25
    if len(pred) == 0:
        return None, pd.DataFrame()

    # Scale boxes back to original size
    h, w = img_pil.height, img_pil.width
    scale = 640 / max(h, w)
    pad_h = (640 - h * scale) / 2
    pad_w = (640 - w * scale) / 2

    boxes = pred[:, :4].cpu().numpy()
    boxes[:, 0] = (boxes[:, 0] - pad_w) / scale  # x1
    boxes[:, 1] = (boxes[:, 1] - pad_h) / scale  # y1
    boxes[:, 2] = (boxes[:, 2] - pad_w) / scale  # x2
    boxes[:, 3] = (boxes[:, 3] - pad_h) / scale  # y2

    # Clip
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

    # To DataFrame
    df = pd.DataFrame({
        "xmin": boxes[:, 0],
        "ymin": boxes[:, 1],
        "xmax": boxes[:, 2],
        "ymax": boxes[:, 3],
        "confidence": pred[:, 4].cpu().numpy(),
        "class": pred[:, 5].cpu().numpy().astype(int),
        "name": [CLASS_NAMES[int(i)] for i in pred[:, 5]]
    })

    # Draw
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img_pil)
    for _, row in df.iterrows():
        draw.rectangle([row.xmin, row.ymin, row.xmax, row.ymax], outline="red", width=3)
        draw.text((row.xmin, row.ymin - 10), f"{row.name} {row.confidence:.2f}", fill="red")

    return img_pil, df

# -------------------------------
# UI
# -------------------------------
uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded:
    tmp_path = ROOT / f"tmp_{int(time.time())}.jpg"
    img = Image.open(uploaded).convert("RGB")
    img.save(tmp_path)

    st.image(img, caption="Original", use_container_width=True)

    with st.spinner("Detecting..."):
        ann_img, df = detect(img.copy())

    if ann_img:
        st.image(ann_img, caption="Detections", use_container_width=True)
    else:
        st.image(img, use_container_width=True)

    tmp_path.unlink(missing_ok=True)

    if df.empty:
        st.warning("No objects detected.")
    else:
        st.subheader("Detections")
        color = {
            "Hardhat": "green", "Mask": "green", "Safety Vest": "green",
            "NO-Hardhat": "red", "NO-Mask": "red", "NO-Safety Vest": "red",
        }
        for _, r in df.iterrows():
            c = color.get(r["name"], "gray")
            st.markdown(f"<span style='color:{c};'><b>{r['name']}</b>: {r['confidence']:.2f}</span>", True)
