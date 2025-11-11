import streamlit as st
import os
from PIL import Image
import time
import pandas as pd
from yolov5.models.common import DetectMultiBackend
import torch

# -------------------------------
# Load your trained YOLOv5 model
# -------------------------------
@st.cache_resource
def load_model():
    weights_path = os.path.join(os.path.dirname(__file__), "best.pt")
    model = DetectMultiBackend(weights_path, device="cpu")
    return model

st.set_page_config(page_title="🦺 PPE Detection App", layout="centered")
st.title("🦺 PPE Detection App")
st.write("Upload an image to detect helmets, vests, masks, and more!")

model = load_model()

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    img = Image.open(uploaded_file)
    img_path = f"temp_{int(time.time())}_{uploaded_file.name}"
    img.save(img_path)

    st.image(img, caption="📸 Uploaded Image", use_container_width=True)
    st.write("🔍 Detecting...")

    # Run detection
    results = model(img_path)

    # Save to unique results folder
    results_dir = os.path.join('results', f"detection_{int(time.time())}")
    os.makedirs(results_dir, exist_ok=True)
    results.save(save_dir=results_dir)

    # Get latest image saved
    detected_files = sorted(
        [os.path.join(results_dir, f) for f in os.listdir(results_dir)],
        key=os.path.getmtime,
        reverse=True
    )

    detected_img_path = detected_files[0] if detected_files else None

    if detected_img_path:
        st.image(detected_img_path, caption="🧠 Detection Results", use_container_width=True)

    # Extract detection info
    try:
        df = results.pandas().xyxy[0]
    except:
        st.warning("⚪ No objects detected.")
        df = pd.DataFrame()

    if len(df) == 0:
        st.warning("⚪ No objects detected.")
    else:
        st.subheader("📋 Detection Details")

        # Color map
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
            "Vehicle": "yellow"
        }

        # Grouping
        sections = {
            "🟢 Safe Equipment": [],
            "🟡 Other Objects": [],
            "🔴 Unsafe Conditions": []
        }

        for _, row in df.iterrows():
            label = row['name']
            conf = row['confidence']
            color = color_map.get(label, "white")
            text = f"<span style='color:{color}; font-size:18px;'><b>{label}</b> — Confidence: {conf:.2f}</span>"

            if label.startswith("NO-"):
                sections["🔴 Unsafe Conditions"].append(text)
            elif label in ["Hardhat", "Mask", "Safety Vest"]:
                sections["🟢 Safe Equipment"].append(text)
            else:
                sections["🟡 Other Objects"].append(text)

        # Display sections
        for title, items in sections.items():
            if items:
                st.markdown(f"### {title}")
                for t in items:
                    st.markdown(t, unsafe_allow_html=True)

        # Summary
        st.write("---")
        st.subheader("🧾 Detection Summary")

        counts = df['name'].value_counts()
        for obj, count in counts.items():
            color = color_map.get(obj, "white")
            st.markdown(
                f"<span style='color:{color}; font-size:16px;'>• {obj}: {count}</span>",
                unsafe_allow_html=True
            )

    # Cleanup
    if os.path.exists(img_path):
        os.remove(img_path)
