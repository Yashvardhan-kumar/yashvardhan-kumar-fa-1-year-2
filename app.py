import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
import os

# ============================
# Page Config
# ============================
st.set_page_config(
    page_title="PPE Compliance Detector",
    page_icon="Hard hat",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================
# Dark-Theme + Glassmorphism CSS
# ============================
st.markdown(
    """
<style>
    /* Global */
    .stApp {background:#0a0e17; color:#e2e8f0;}
    .main {padding:2rem;}

    /* Title */
    .big-title {
        font-size:3.2rem;
        font-weight:800;
        background:linear-gradient(90deg,#4facfe,#00f2fe);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        text-align:center;
        margin-bottom:.5rem;
    }
    .subtitle {
        text-align:center;
        color:#94a3b8;
        font-size:1.15rem;
        margin-bottom:2.5rem;
    }

    /* Upload Card */
    .upload-card {
        background:rgba(30,41,59,0.75);
        backdrop-filter:blur(12px);
        border:2px dashed #3b82f6;
        border-radius:18px;
        padding:2.5rem;
        text-align:center;
        transition:all .3s ease;
    }
    .upload-card:hover {
        border-color:#60a5fa;
        background:rgba(51,65,85,0.75);
        transform:translateY(-4px);
    }
    .upload-icon {font-size:3rem; color:#60a5fa;}

    /* Result Card */
    .result-card {
        background:rgba(30,41,59,0.75);
        backdrop-filter:blur(12px);
        border-radius:16px;
        padding:1.5rem;
        border:1px solid #334155;
        box-shadow:0 8px 32px rgba(0,0,0,.4);
        margin:1.5rem 0;
    }

    /* Metric Tile */
    .metric-tile {
        background:rgba(30,41,59,0.75);
        border-radius:14px;
        padding:1rem;
        text-align:center;
        border:1px solid #334155;
    }
    .metric-tile .stMetricValue {font-size:2rem !important;}
    .metric-tile .stMetricDelta {font-size:1rem !important;}

    /* Colors */
    .compliant {color:#34d399; font-weight:600;}
    .non-compliant {color:#f87171; font-weight:600;}
    .neutral {color:#a78bfa; font-weight:600;}

    /* Footer */
    .footer {
        text-align:center;
        margin-top:4rem;
        color:#64748b;
        font-size:.9rem;
    }

    /* Hide Streamlit junk */
    #MainMenu, footer {visibility:hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ============================
# Load Model (unchanged)
# ============================
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file `{model_path}` not found. Place it in the app folder.")
        st.stop()
    return torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=False)

model = load_model()

# ============================
# Compliance Map (unchanged)
# ============================
compliance_map = {
    "Hardhat": "Hardhat Worn",
    "Safety Vest": "Safety Vest Worn",
    "Mask": "Mask Worn",
    "NO-Hardhat": "Missing Hardhat",
    "NO-Safety Vest": "Missing Safety Vest",
    "NO-Mask": "Missing Mask",
    "Person": "Worker Detected",
    "machinery": "Machinery",
    "vehicle": "Vehicle",
    "Safety Cone": "Safety Cone",
}

# ============================
# Header
# ============================
st.markdown('<h1 class="big-title">PPE Compliance Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time safety gear verification for construction sites</p>', unsafe_allow_html=True)

# ============================
# Upload Area
# ============================
with st.container():
    st.markdown(
        """
        <div class="upload-card">
            <div class="upload-icon">Upload image</div>
            <h4 style="color:#60a5fa; margin:1rem 0 0.5rem;">Drop an image here</h4>
            <p style="color:#94a3b8;">Supported: JPG, JPEG, PNG</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# ============================
# Process Image (logic unchanged)
# ============================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    colA, colB = st.columns([1, 1])

    # ---- Original ----
    with colA:
        st.image(image, caption="Original Image", use_column_width=True)

    # ---- Detection ----
    with st.spinner("Running YOLOv5 inference..."):
        results = model(image)
        df = results.pandas().xyxy[0]

        annotated = img_np.copy()
        for _, row in df.iterrows():
            label = row["name"]
            x1, y1, x2, y2 = map(int, [row["xmin"], row["ymin"], row["xmax"], row["ymax"]])
            is_compliant = label in ["Hardhat", "Safety Vest", "Mask"]
            color = (0, 255, 0) if is_compliant else (255, 0, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            txt = compliance_map.get(label, label)
            cv2.putText(annotated, txt, (x1, y1 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    with colB:
        st.markdown("### Detection Result")
        st.image(annotated, use_column_width=True)

    # ---- Download Button ----
    _, dl_col, _ = st.columns([1, 2, 1])
    with dl_col:
        success, buf = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        st.download_button(
            label="Download Annotated Image",
            data=buf.tobytes(),
            file_name="ppe_result.jpg",
            mime="image/jpeg",
        )

    # ============================
    # Summary Card (unchanged logic, nicer UI)
    # ============================
    st.markdown("### Compliance Summary")
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)

    compliant_count = 0
    violations = 0
    workers = len(df[df["name"] == "Person"])

    for label in df["name"].unique():
        cnt = (df["name"] == label).sum()
        txt = compliance_map.get(label, label)

        if any(x in txt for x in ["Hardhat Worn", "Safety Vest Worn", "Mask Worn"]):
            st.markdown(f"<p class='compliant'>Check {txt}: <strong>{cnt}</strong></p>", unsafe_allow_html=True)
            compliant_count += cnt
        elif "Missing" in txt:
            st.markdown(f"<p class='non-compliant'>Warning {txt}: <strong>{cnt}</strong></p>", unsafe_allow_html=True)
            violations += cnt
        else:
            st.markdown(f"<p class='neutral'>{txt}: <strong>{cnt}</strong></p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ============================
    # Metrics (icons + tiles)
    # ============================
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            """
            <div class="metric-tile">
                <div style="font-size:2.2rem;">Person</div>
                <div style="font-size:1rem; color:#94a3b8;">Total Workers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("", workers)
    with m2:
        st.markdown(
            """
            <div class="metric-tile">
                <div style="font-size:2.2rem;">Check</div>
                <div style="font-size:1rem; color:#94a3b8;">Compliant Items</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("", compliant_count)
    with m3:
        st.markdown(
            """
            <div class="metric-tile">
                <div style="font-size:2.2rem;">Warning</div>
                <div style="font-size:1rem; color:#94a3b8;">Violations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("", violations, delta=violations)

    # ============================
    # Final Status
    # ============================
    if violations == 0 and workers > 0:
        st.success("**All workers are fully compliant!** Great job keeping the site safe.")
    elif violations > 0:
        st.error(f"**{violations} PPE violation(s) detected!** Immediate action required.")
    else:
        st.info("No workers detected in the image.")

else:
    st.info("Upload a construction-site image to begin the compliance check.")

# ============================
# Footer
# ============================
st.markdown(
    """
    <div class="footer">
        <p>PPE Compliance Detector • Powered by YOLOv5 + Streamlit</p>
        <p>Safety first, always.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
