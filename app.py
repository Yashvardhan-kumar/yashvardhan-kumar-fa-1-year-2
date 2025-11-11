import streamlit as st
import os
import time
from PIL import Image
import torch

# --- Streamlit page ---
st.set_page_config(page_title="🦺 PPE Detection App", layout="centered")
st.title("🦺 PPE Detection App")
st.write("Upload an image to detect helmets, vests, masks, and more!")

# --- Load YOLOv5 model (local best.pt in the same folder) ---
@st.cache_resource
def load_model():
    weights_path = os.path.join(os.path.dirname(__file__), "best.pt")
    if not os.path.exists(weights_path):
        st.stop()  # stop app with a clear message
    model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=weights_path,
        force_reload=False
    )
    model.eval()
    return model

model = load_model()

# --- Uploader ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # save uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    img_path = f"temp_{int(time.time())}_{uploaded_file.name}"
    img.save(img_path)

    st.image(img, caption="📸 Uploaded Image", use_container_width=True)
    st.write("🔍 Detecting...")

    # results directory (unique per run)
    results_dir = os.path.join("results", f"detection_{int(time.time())}")
    os.makedirs(results_dir, exist_ok=True)

    # run detection
    results = model(img_path)
    results.save(save_dir=results_dir)

    # show the saved, annotated image
    detected_files = sorted(
        [os.path.join(results_dir, f) for f in os.listdir(results_dir)],
        key=os.path.getmtime,
        reverse=True
    )
    detected_img_path = detected_files[0] if detected_files else None
    if detected_img_path:
        st.image(detected_img_path, caption="🧠 Detection Results", use_container_width=True)

    # details table
    try:
        detected_objects = results.pandas().xyxy[0]
    except Exception:
        detected_objects = None

    if detected_objects is None or len(detected_objects) == 0:
        st.warning("⚪ No objects detected.")
    else:
        st.subheader("📋 Detection Details")

        color_map = {
            "Hardhat": "green", "Mask": "green", "Safety Vest": "green",
            "NO-Hardhat": "red", "NO-Mask": "red", "NO-Safety Vest": "red",
            "Person": "yellow", "Safety Cone": "yellow",
            "Machinery": "yellow", "Vehicle": "yellow"
        }

        sections = {"🟢 Safe Equipment": [], "🟡 Other Objects": [], "🔴 Unsafe Conditions": []}

        for _, row in detected_objects.iterrows():
            label = str(row.get("name", ""))
            conf = float(row.get("confidence", 0.0))
            color = color_map.get(label, "white")
            line = (
                f"<span style='color:{color}; font-size:18px;'>"
                f"<b>{label}</b> — Confidence: {conf:.2f}</span>"
            )
            if label.startswith("NO-"):
                sections["🔴 Unsafe Conditions"].append(line)
            elif label in ["Hardhat", "Mask", "Safety Vest"]:
                sections["🟢 Safe Equipment"].append(line)
            else:
                sections["🟡 Other Objects"].append(line)

        for title, items in sections.items():
            if items:
                st.markdown(f"### {title}")
                for html in items:
                    st.markdown(html, unsafe_allow_html=True)

        st.write("---")
        st.subheader("🧾 Detection Summary")
        counts = detected_objects['name'].value_counts()
        for obj, count in counts.items():
            color = color_map.get(obj, "white")
            st.markdown(
                f"<span style='color:{color}; font-size:16px;'>• {obj}: {count}</span>",
                unsafe_allow_html=True
            )

    # cleanup
    try:
        if os.path.exists(img_path):
            os.remove(img_path)
    except Exception:
        pass
