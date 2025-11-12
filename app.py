# app.py (UI refresh only - logic unchanged)
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import io
import os

# --- page config + small theme touch ---
st.set_page_config(page_title="PPE Compliance Detector", layout="centered", initial_sidebar_state="collapsed")

# --- custom CSS to change look & feel (colors, fonts, cards) ---
CUSTOM_CSS = """
/* overall background */
[data-testid="stAppViewContainer"] {
  background: linear-gradient(180deg, #0f1720 0%, #0b1320 100%);
  color: #e6eef8;
  font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
}

/* center content + top padding */
main > div blockquote + div {
  padding-top: 18px;
}

/* header */
h1 {
  font-size: 36px !important;
  color: #f4f9ff !important;
  letter-spacing: -0.5px;
}

/* subtitle */
h2, h3 {
  color: #dbeafe !important;
}

/* boxes (error/info) slightly restyled */
.stAlert {
  border-radius: 12px !important;
  padding: 12px 18px !important;
  font-weight: 600;
}

/* custom card container */
.ppe-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.04);
  border-radius: 14px;
  padding: 18px;
  box-shadow: 0 6px 18px rgba(2,6,23,0.6);
  margin-bottom: 18px;
}

/* uploader styling */
.upload-wrap {
  background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius: 12px;
  border: 1px dashed rgba(255,255,255,0.06);
  padding: 14px;
}

/* button style */
.stButton > button {
  background: linear-gradient(90deg, #00c2a8, #2dd4bf);
  color: #022c2a;
  font-weight: 700;
  border-radius: 10px;
  padding: 8px 14px;
}

/* small muted text */
.muted {
  color: rgba(230,238,248,0.6);
  font-size: 13px;
}

/* compliance pills */
.pill {
  display:inline-block;
  padding:6px 10px;
  border-radius:999px;
  margin:4px 6px 4px 0;
  font-weight:600;
}
.pill.good { background: rgba(34,197,94,0.12); color:#bbf7d0; border:1px solid rgba(34,197,94,0.12);}
.pill.bad  { background: rgba(239,68,68,0.12); color:#ffd7d7; border:1px solid rgba(239,68,68,0.12);}

/* image captions */
[data-testid="stImage"] figcaption {
  color: #cfe8ff !important;
  font-weight:600;
}

/* download button tweak */
.stDownloadButton>button {
  background: linear-gradient(90deg, #f59e0b, #fb923c);
  color: #1a1207;
  border-radius: 9px;
  font-weight:700;
}
"""

st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# --- Title area (keeps same text but nicer layout) ---
col1, col2 = st.columns([0.1, 1])
with col1:
    st.markdown("<div style='font-size:36px'>🦺</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1>PPE Compliance Detection</h1>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Upload a photo and the model will mark PPE presence and compliance.</div>", unsafe_allow_html=True)

# small map kept from your original
compliance_map = {
    'Hardhat': '✅ Compliant',
    'Safety Vest': '✅ Compliant',
    'Mask': '✅ Compliant',
    'NO-Hardhat': '❌ Missing Hardhat',
    'NO-Safety Vest': '❌ Missing Vest',
    'NO-Mask': '❌ Missing Mask',
    'Person': '👤 Worker',
    'machinery': '⚙️ Machinery',
    'vehicle': '🚗 Vehicle',
    'Safety Cone': '🟠 Cone'
}

MODEL_PATH = "best.pt"  # ensure this exists in repo root or change to URL

# lazy cv2 importer (won't crash app if cv2 / native libs missing)
def try_import_cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None

cv2 = try_import_cv2()

# safe model loader (gives readable errors)
@st.cache_resource(show_spinner=False)
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at '{path}'. Put your best.pt in repo root or provide a URL and change MODEL_PATH.")
    try:
        # load via torch.hub (ultralytics yolov5)
        return torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=False)
    except Exception as e:
        raise RuntimeError("Model load failed: " + str(e))

# try to load model, show friendly message if fails
model = None
try:
    model = load_model()
except Exception as e:
    # intentionally using the same messages so logic/flow unchanged
    st.error("Model load: " + str(e))
    st.info("You can still upload an image, but detection won't run until model is loaded successfully.")

# uploader card (styled)
st.markdown("<div class='ppe-card'>", unsafe_allow_html=True)
st.markdown("<h3>Upload an image (jpg/png)</h3>", unsafe_allow_html=True)
st.markdown("<div class='muted'>Drag & drop an image or use Browse. Max 200MB.</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

def draw_boxes_pil(img_pil, df):
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    for _, row in df.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = row['name']
        color = (0,255,0) if not str(label).startswith('NO-') else (255,0,0)
        draw.rectangle([x1,y1,x2,y2], outline=color, width=2)
        draw.text((x1, max(0,y1-12)), compliance_map.get(label, label), fill=color, font=font)
    return img_pil

def draw_boxes_cv2(img_np, df, cv2):
    out = img_np.copy()
    for _, row in df.iterrows():
        x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        label = row['name']
        color = (0,255,0) if not str(label).startswith('NO-') else (0,0,255)  # BGR
        cv2.rectangle(out, (x1,y1),(x2,y2), color, 2)
        cv2.putText(out, compliance_map.get(label, label), (x1, max(10,y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if model is None:
        st.warning("Model not loaded — detection skipped. Fix model (best.pt) or check logs.")
    else:
        with st.spinner("Running detection..."):
            try:
                results = model(img)
                df = results.pandas().xyxy[0]
            except Exception as e:
                st.error("Inference failed: " + str(e))
                st.stop()

        # annotate and display
        if cv2 is not None:
            # convert PIL to numpy BGR for cv2 drawing
            arr = np.array(img)  # RGB
            bgr = arr[..., ::-1]
            annotated_bgr = draw_boxes_cv2(bgr, df, cv2)
            annotated_rgb = annotated_bgr[..., ::-1]
            st.image(annotated_rgb, caption="🧠 Detection Result (cv2)", use_column_width=True)
            # optional download
            buf = io.BytesIO()
            Image.fromarray(annotated_rgb).save(buf, format="PNG")
            buf.seek(0)
            st.download_button("Download annotated image", data=buf, file_name="annotated.png", mime="image/png")
        else:
            # PIL drawing fallback
            annotated = draw_boxes_pil(img.copy(), df)
            st.image(annotated, caption="🧠 Detection Result (PIL fallback)", use_column_width=True)
            buf = io.BytesIO()
            annotated.save(buf, format="PNG")
            buf.seek(0)
            st.download_button("Download annotated image", data=buf, file_name="annotated.png", mime="image/png")

        # summary
        st.subheader("Compliance summary")
        counts = df['name'].value_counts()
        # nicer inline pills for each label (visual only)
        pill_html = ""
        for label, cnt in counts.items():
            is_ok = not str(label).startswith('NO-')
            pill_class = "good" if is_ok else "bad"
            pretty = compliance_map.get(label, label)
            pill_html += f"<span class='pill {pill_class}'>{pretty} × {cnt}</span>"
        st.markdown(pill_html, unsafe_allow_html=True)
