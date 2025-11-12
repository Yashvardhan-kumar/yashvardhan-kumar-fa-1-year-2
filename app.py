# app.py – PPE Detection – ONLY best.pt + this file
import os
import time
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# 1. Patch the missing YOLOv5 class (required for torch.load)
# ------------------------------------------------------------------
class DetectionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.nc = kwargs.get("nc", 80)
        self.names = kwargs.get("names", [str(i) for i in range(self.nc)])

    def forward(self, x, *args, **kwargs):
        return x

# Register it so torch.load can find it
import sys
sys.modules["models.yolo"] = type("module", (), {})
sys.modules["models.yolo"].DetectionModel = DetectionModel

# ------------------------------------------------------------------
# 2. Paths
# ------------------------------------------------------------------
ROOT = Path(__file__).parent
WEIGHTS = ROOT / "best.pt"

st.set_page_config(page_title="PPE Detector", layout="centered")
st.title("PPE Detection")
st.write("Upload an image → detect **helmets, vests, masks**, and safety violations.")

# ------------------------------------------------------------------
# 3. Load model (weights_only=False – safe because you trained it)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not WEIGHTS.exists():
        st.error("`best.pt` not found! Upload it to the repo root.")
        st.stop()

    try:
        ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    except Exception as e:
        st.error("Failed to load `best.pt`.")
        st.exception(e)
        st.stop()

    model = ckpt.get("model") or ckpt.get("ema") or ckpt
    names = ckpt.get("names") or getattr(model, "names", None)
    if names is None:
        st.error("Class names missing in `best.pt`.")
        st.stop()

    model.eval()
    st.success("Model loaded from `best.pt`")
    return model, names

MODEL, CLASS_NAMES = load_model()

# ------------------------------------------------------------------
# 4. Preprocess (YOLOv5 style)
# ------------------------------------------------------------------
def preprocess(img_pil):
    w, h = img_pil.size
    r = 640 / max(w, h)
    nw, nh = int(w * r), int(h * r)
    img = img_pil.resize((nw, nh), Image.BILINEAR)

    pad_w = 640 - nw
    pad_h = 640 - nh
    pad_l = pad_w // 2
    pad_t = pad_h // 2

    img = np.array(img)
    img = np.pad(img, ((pad_t, pad_h - pad_t), (pad_l, pad_w - pad_l), (0, 0)), mode='constant')
    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0)
    return img, (w, h), r, (pad_l, pad_t)

# ------------------------------------------------------------------
# 5. Post-process + draw
# ------------------------------------------------------------------
def postprocess(pred, img_pil, old_size, ratio, padding):
    pred = pred[0]
    pred = pred[pred[:, 4] > 0.25]
    if len(pred) == 0:
        return img_pil, pd.DataFrame()

    pad_l, pad_t = padding
    boxes = pred[:, :4].cpu().numpy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_l) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_t) / ratio

    w, h = old_size
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

    df = pd.DataFrame({
        "xmin": boxes[:, 0], "ymin": boxes[:, 1],
        "xmax": boxes[:, 2], "ymax": boxes[:, 3],
        "confidence": pred[:, 4].cpu().numpy(),
        "name": [CLASS_NAMES[int(c)] for c in pred[:, 5]]
    })

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

# ------------------------------------------------------------------
# 6. Inference
# ------------------------------------------------------------------
def detect(img_pil):
    img_tensor, old_size, ratio, padding = preprocess(img_pil)
    with torch.no_grad():
        pred = MODEL(img_tensor)
    return postprocess(pred, img_pil.copy(), old_size, ratio, padding)

# ------------------------------------------------------------------
# 7. UI
# ------------------------------------------------------------------
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    tmp_path = ROOT / f"tmp_{int(time.time())}.jpg"
    img = Image.open(uploaded).convert("RGB")
    img.save(tmp_path)

    st.image(img, caption="Original", use_container_width=True)

CHD    with st.spinner("Detecting..."):
        ann_img, df = detect(img)

    st.image(ann_img, caption="Detections", use_container_width=True)
    tmp_path.unlink(missing_ok=True)

    if df.empty:
        st.warning("No objects detected.")
    else:
        st.subheader("Detections")
        colors = {
            "Hardhat": "green", "Mask": "green", "Safety Vest": "green",
            "NO-Hardhat": "red", "NO-Mask": "red", "NO-Safety Vest": "red",
        }
        for _, r in df.iterrows():
            c = colors.get(r["name"], "gray")
            st.markdown(f"<span style='color:{c};'><b>{r['name']}</b>: {r['confidence']:.2f}</span>", True)
