# app.py — PPE Detection using torch.hub (no local yolov5 folder required)
import os
import time
from pathlib import Path
import streamlit as st
from PIL import Image
import pandas as pd
import torch

REPO_ROOT = Path(__file__).parent
WEIGHTS = REPO_ROOT / "best.pt"  # ensure best.pt is in repo root

st.set_page_config(page_title="PPE Detection App", layout="centered")
st.title("PPE Detection App")
st.write("Upload an image to detect helmets, vests, masks, and other PPE items.")

# -------------------------
# Load model via torch.hub (with skip_validation to avoid rate limits)
# -------------------------
@st.cache_resource
def load_model_hub():
    if not WEIGHTS.exists():
        st.error(f"Model weights not found: '{WEIGHTS.name}'. Upload `best.pt` to repo root.")
        st.stop()

    try:
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(WEIGHTS),
            force_reload=False,
            trust_repo=True,
            skip_validation=True  # Critical: avoids GitHub API rate limit
        )
        model.eval()
        return {"backend": "hub", "model": model}
    except Exception as e:
        st.error("Failed to load YOLOv5 via torch.hub.")
        st.markdown(
            "<details><summary>Technical error (click to expand)</summary>"
            f"<pre>{str(e)}</pre></details>",
            unsafe_allow_html=True,
        )
        st.info(
            "Quick fixes:\n\n"
            "• Add `skip_validation=True` (already done), or\n"
            "• Clone a minimal `yolov5/` folder locally and use `source='local'`, or\n"
            "• Check if `best.pt` is uploaded correctly."
        )
        st.stop()

state = load_model_hub()
MODEL = state["model"]

# -------------------------
# Helper: run detection using hub-style API
# -------------------------
def run_detection_on_path(img_path, model, conf_thres=0.25):
    try:
        results = model(img_path, size=640, augment=False)
    except Exception as e:
        raise RuntimeError(f"Model inference failed: {e}") from e

    # Render annotated image
    try:
        imgs = results.render()
        annotated = imgs[0]
        pil_ann = Image.fromarray(annotated)
    except Exception:
        pil_ann = None

    # Detection table
    try:
        df = results.pandas().xyxy[0]
    except Exception:
        df = pd.DataFrame()

    return pil_ann, df

# -------------------------
# UI
# -------------------------
uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if not uploaded:
    st.info("Upload an image to run PPE detection.")
else:
    # Save uploaded image temporarily
    tmp_name = f"tmp_{int(time.time())}.jpg"
    tmp_path = REPO_ROOT / tmp_name
    image = Image.open(uploaded).convert("RGB")
    image.save(tmp_path)

    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("Detecting...")

    try:
        annotated_img, df = run_detection_on_path(str(tmp_path), MODEL, conf_thres=0.25)
    except Exception as e:
        st.error("Inference failed. See details below.")
        st.exception(e)
        tmp_path.unlink(missing_ok=True)
        st.stop()

    # Display results
    if annotated_img is not None:
        st.image(annotated_img, caption="Detection Results", use_container_width=True)
    else:
        st.warning("Annotated image not available; showing original.")
        st.image(image, use_container_width=True)

    # Cleanup
    tmp_path.unlink(missing_ok=True)

    # Detection details
    if df.empty:
        st.warning("No objects detected.")
    else:
        st.subheader("Detection Details")
        color_map = {
            "Hardhat": "green", "Mask": "green", "Safety Vest": "green",
            "NO-Hardhat": "red", "NO-Mask": "red", "NO-Safety Vest": "red",
            "Person": "yellow", "Safety Cone": "yellow",
            "Machinery": "yellow", "Vehicle": "yellow",
        }
        sections = {
            "Safe Equipment": [],
            "Other Objects": [],
            "Unsafe Conditions": []
        }

        if "name" not in df.columns and "class" in df.columns:
            df = df.rename(columns={"class": "name"})

        for _, row in df.iterrows():
            label = str(row.get("name", ""))
            conf = float(row.get("confidence", 0.0))
            color = color_map.get(label, "white")
            html_line = f"<span style='color:{color}; font-size:18px;'><b>{label}</b> — {conf:.2f}</span>"

            if label.startswith("NO-"):
                sections["Unsafe Conditions"].append(html_line)
            elif label in ["Hardhat", "Mask", "Safety Vest"]:
                sections["Safe Equipment"].append(html_line)
            else:
                sections["Other Objects"].append(html_line)

        for title, items in sections.items():
            if items:
                st.markdown(f"### {title}")
                for line in items:
                    st.markdown(line, unsafe_allow_html=True)

        st.write("---")
        st.subheader("Detection Summary")
        try:
            counts = df["name"].value_counts()
            for obj, count in counts.items():
                color = color_map.get(obj, "white")
                st.markdown(f"<span style='color:{color};'>• {obj}: {count}</span>", unsafe_allow_html=True)
        except Exception:
            pass
