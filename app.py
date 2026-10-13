import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from collections import Counter
from io import BytesIO

st.set_page_config(
    page_title="RoadMind Lite | Road Scene Understanding",
    page_icon="🚦",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; background: linear-gradient(120deg,#102a43,#1f526b); color: white;}
.hero h1 {color: white; margin-bottom: .25rem;}
.hero p {color: #d9e8f0; margin-bottom: 0;}
.small-note {font-size: .88rem; color: #64748b;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🚦 RoadMind Lite</h1>
  <p>Interactive baseline for road-scene object detection and visual scene exploration.</p>
</div>
""", unsafe_allow_html=True)

st.write("")
st.write(
    "RoadMind Lite uses a pretrained object detector to identify objects in a road-scene image. "
    "It is a starting point for research into road-user behaviour understanding—not a safety-critical system."
)

@st.cache_resource(show_spinner="Loading pretrained detection model…")
def load_model():
    # The small pretrained model is downloaded on first run.
    return YOLO("yolo11n.pt")

with st.sidebar:
    st.header("Analysis settings")
    confidence = st.slider(
        "Detection confidence threshold",
        min_value=0.10, max_value=0.90, value=0.35, step=0.05,
        help="Higher thresholds show fewer detections, generally with greater model confidence."
    )
    st.caption("Model: YOLO11n pretrained on COCO classes")
    st.divider()
    st.subheader("Research direction")
    st.write("Next steps: video tracking, temporal behaviour classification, domain-shift evaluation, and uncertainty analysis.")

uploaded = st.file_uploader(
    "Upload a road-scene image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Use a road or traffic image that you have permission to analyse."
)

if uploaded is None:
    st.info("Upload an image to run detection. For best results, use a clear street scene with visible road users.")
    st.markdown("### Research questions this prototype motivates")
    st.markdown("""
    - Which road-user classes are detected reliably, and which are missed?
    - How does the confidence threshold change the number and type of detections?
    - How might temporal information improve understanding beyond single-frame detection?
    """)
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except Exception:
    st.error("The uploaded file could not be read as an image.")
    st.stop()

left, right = st.columns([1, 1], gap="large")
with left:
    st.subheader("Input image")
    st.image(image, use_container_width=True)

try:
    model = load_model()
    with st.spinner("Analysing road scene…"):
        result = model.predict(image, conf=confidence, verbose=False)[0]
    annotated = result.plot()
    # Ultralytics returns a BGR array; convert to RGB for Streamlit.
    import cv2
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    rows = []
    names = result.names
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for box, score, cls_id in zip(xyxy, confs, clss):
            rows.append({
                "class": names.get(int(cls_id), str(cls_id)),
                "confidence": round(float(score), 4),
                "x_min": round(float(box[0]), 1),
                "y_min": round(float(box[1]), 1),
                "x_max": round(float(box[2]), 1),
                "y_max": round(float(box[3]), 1),
            })

    with right:
        st.subheader("Detected objects")
        st.image(annotated_rgb, use_container_width=True)

    df = pd.DataFrame(rows)
    st.divider()
    st.subheader("Detection summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Objects detected", len(df))
    c2.metric("Unique classes", int(df["class"].nunique()) if not df.empty else 0)
    c3.metric("Mean confidence", f"{df['confidence'].mean():.1%}" if not df.empty else "—")

    if df.empty:
        st.warning("No objects were detected above this threshold. Try lowering the confidence threshold or using another image.")
    else:
        counts = df["class"].value_counts().rename_axis("Class").reset_index(name="Count")
        chart_col, table_col = st.columns([0.85, 1.15], gap="large")
        with chart_col:
            st.markdown("**Objects by class**")
            st.bar_chart(counts.set_index("Class"))
        with table_col:
            st.markdown("**Detection details**")
            st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download detections as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="roadmind_detections.csv",
            mime="text/csv",
        )

    st.divider()
    st.subheader("Interpretation and limitations")
    st.markdown("""
    - This is **single-image object detection**, not full road-user behaviour understanding.
    - A detected object does not establish its intention, attention, or future movement.
    - Performance can change with camera angle, weather, lighting, image quality, and geographic context.
    - The pretrained model is a baseline. Research claims require a defined dataset, ground truth, and quantitative evaluation.
    """)
    st.caption("Research prototype by Victor Taiwo · Built with Streamlit and Ultralytics YOLO.")
except Exception as exc:
    st.error("The model could not complete inference. Check the app logs and package versions.")
    st.exception(exc)
