# app.py — PPE Detection using torch.hub (no local yolov5 folder required)
import os
import time
from pathlib import Path

import streamlit as st
from PIL import Image
import pandas as pd
import torch
import numpy as np

REPO_ROOT = Path(__file__).parent
WEIGHTS = REPO_ROOT / "best.pt"  # ensure best.pt is in repo root

st.set_page_config(page_title="🦺 PPE Detection App", layout="centered")
st.title("🦺 PPE Detection App")
st.write("Upload an image to detect helmets, vests, masks, and other PPE items.")

# -------------------------
# Load model via torch.hub
# -------------------------
@st.cache_resource
def load_model_hub():
    if not WEIGHTS.exists():
        st.error(f"Model weights not found: '{WEIGHTS.name}'. Upload best.pt to repo root.")
        st.stop()

    try:
        # trust_repo=True avoids the interactive trust prompt in newer torch versions
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(WEIGHTS),
            force_reload=False,
            trust_repo=True,   # automatically trust the repo (no prompt)
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
            "Two quick remedies:\n\n"
            "• Upload a minimal `yolov5/` folder next to app.py (preferred, offline), or\n"
            "• Check Streamlit logs for dependency problems (e.g., missing system libs like libGL)."
        )
        st.stop()

state = load_model_hub()
MODEL = state["model"]

# -------------------------
# Helper: run detection using hub-style API
# -------------------------
def run_detection_on_path(img_path, model, conf_thres=0.25):
    """
    Runs the hub model on the given file path. Returns (annotated_image_pil, pandas_df).
    """
    try:
        results = model(img_path, size=640, augment=False)
    except Exception as e:
        # try a simple wrapper error message
        raise RuntimeError(f"Model inference failed: {e}") from e

    # results.render() produces list of np arrays in RGB
    try:
        imgs = results.render()  # modifies results.imgs to annotated images
        annotated = imgs[0]  # first image
        # convert to PIL
        pil_ann = Image.fromarray(annotated)
    except Exception:
        pil_ann = None

    # detection table (pandas)
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
    # save temp file and run hub model
    tmp_name = f"tmp_{int(time.time())}.jpg"
    tmp_path = REPO_ROOT / tmp_name
    image = Image.open(uploaded).convert("RGB")
    image.save(tmp_path)

    st.image(image, caption="📸 Uploaded Image", use_container_width=True)
    st.write("🔍 Detecting...")

    try:
        annotated_img, df = run_detection_on_path(str(tmp_path), MODEL, conf_thres=0.25)
    except Exception as e:
        st.error("Inference failed. See details below.")
        st.exception(e)
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        st.stop()

    # show annotated image if available, else original
    if annotated_img is not None:
        st.image(annotated_img, caption="🧠 Detection Results", use_container_width=True)
    else:
        st.warning("Annotated image not available; showing original.")
        st.image(image, use_container_width=True)

    # cleanup temp
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    # prepare detection details
    if df is None or df.empty:
        st.warning("⚪ No objects detected.")
    else:
        st.subheader("📋 Detection Details")

        color_map = {
            "Hardhat": "green",
            "Mask": "green",
            "Safety Vest": "green",
            "NO-Hardhat": "red",
            "NO-Mask": "red",
            "NO-Safety Vest": "red",
            "Person": "yellow",
            "Safety Cone": "yellow",
            "Machinery": "yellow",
            "Vehicle": "yellow",
        }

        sections = {"🟢 Safe Equipment": [], "🟡 Other Objects": [], "🔴 Unsafe Conditions": []}

        # ensure name column exists
        if "name" not in df.columns and "class" in df.columns:
            df = df.rename(columns={"class": "name"})

        for _, row in df.iterrows():
            label = str(row.get("name", ""))
            conf = float(row.get("confidence", 0.0))
            color = color_map.get(label, "white")
            html_line = f"<span style='color:{color}; font-size:18px;'><b>{label}</b> — Confidence: {conf:.2f}</span>"

            if label.startswith("NO-"):
                sections["🔴 Unsafe Conditions"].append(html_line)
            elif label in ["Hardhat", "Mask", "Safety Vest"]:
                sections["🟢 Safe Equipment"].append(html_line)
            else:
                sections["🟡 Other Objects"].append(html_line)

        for title, items in sections.items():
            if items:
                st.markdown(f"### {title}")
                for html_line in items:
                    st.markdown(html_line, unsafe_allow_html=True)

        st.write("---")
        st.subheader("🧾 Detection Summary")
        try:
            counts = df["name"].value_counts()
            for obj, count in counts.items():
                color = color_map.get(obj, "white")
                st.markdown(f"<span style='color:{color}; font-size:16px;'>• {obj}: {count}</span>",
                            unsafe_allow_html=True)
        except Exception:
            pass
