import streamlit as st
import torch
import cv2
import tempfile
import os
import shutil
from pathlib import Path

from src.model import VideoDetector
from scripts.infer import infer


# =========================
# Fix: make infer() find the model where it expects it
# infer() looks for: models/best_model.pth
# If your model is already there, we do nothing.
# If you later move it elsewhere, this function can copy it safely.
# =========================
def ensure_infer_model_path():
    base_dir = Path(__file__).resolve().parent
    src = base_dir / "models" / "best_model.pth"
    dst_dir = base_dir / "models"
    dst = dst_dir / "best_model.pth"

    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Prevent copying a file onto itself (SameFileError)
        try:
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
        except Exception:
            # If resolve fails for any reason, fall back to safe behavior
            if str(src) != str(dst):
                shutil.copy2(src, dst)

ensure_infer_model_path()


# =========================
# Model loading
# =========================
@st.cache_resource
def load_model():
    try:
        base_dir = Path(__file__).resolve().parent
        model_path = base_dir / "models" / "best_model.pth"

        if not model_path.exists():
            return None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VideoDetector()
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except:
        return None


# =========================
# Page config (emoji removed)
# =========================
st.set_page_config(
    page_title="AI Video Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# Global styles (anti-empty-rectangles included)
# =========================
st.markdown(
    """
<style>
/* Layout */
.block-container { padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1200px; }

/* HERO */
.hero {
    border: 1px solid rgba(255,255,255,0.10);
    background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.025));
    border-radius: 20px;
    padding: 1.35rem 1.45rem;
    box-shadow: 0 10px 24px rgba(0,0,0,0.22);
}
.hero-title{ font-size: 2.25rem; font-weight: 900; margin:0; letter-spacing: 0.2px; line-height:1.1; }
.hero-sub{ opacity: 0.86; margin-top: 0.45rem; margin-bottom: 0.95rem; font-size: 1.02rem; }

.badges{ display:flex; gap:0.55rem; flex-wrap:wrap; }
.badge{
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(0,0,0,0.18);
    border-radius: 999px;
    padding: 0.34rem 0.70rem;
    font-size: 0.90rem;
    opacity: 0.95;
}

/* CARDS */
.card{
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.03);
    border-radius: 20px;
    padding: 1.10rem 1.20rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 24px rgba(0,0,0,0.18);
}
.card-title{ font-size: 1.08rem; font-weight: 800; margin-bottom: 0.65rem; }
.muted{ opacity:0.78; font-size:0.94rem; line-height:1.35; }

/* RESULTS */
.result{ border-left: 6px solid rgba(255,255,255,0.12); }
.result.real{ border-left-color: rgba(0,200,120,0.95); }
.result.fake{ border-left-color: rgba(255,80,80,0.95); }

/* Button */
.stButton > button[kind="primary"]{
    border-radius: 14px !important;
    font-weight: 800 !important;
    padding: 0.75rem 1rem !important;
}

/* Make expanders look nicer */
[data-testid="stExpander"]{
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 16px !important;
  background: rgba(0,0,0,0.14) !important;
}

/* KILL EMPTY RECTANGLES */
.stTabs [data-baseweb="tab-list"] + div:empty { display: none !important; }
[data-testid="stVerticalBlock"] div:empty { display: none !important; }
[data-testid="stMarkdownContainer"]:empty { display: none !important; }
</style>
""",
    unsafe_allow_html=True
)


# =========================
# Helpers
# =========================
MAX_MB = 200.0

def explain_mode(mode_key: str) -> str:
    if mode_key == "f1":
        return (
            "**F1-Optimal (Balanced)**\n\n"
            "- Optimizes the F1-score.\n"
            "- Best trade-off between precision and recall.\n"
            "- Recommended for demos."
        )
    return (
        "**Recall-Constrained (High Security)**\n\n"
        "- Prioritizes recall.\n"
        "- Fewer missed AI videos.\n"
        "- May flag some real videos."
    )


def confidence_bucket(conf: float) -> str:
    if conf >= 0.85:
        return "High"
    if conf >= 0.65:
        return "Medium"
    return "Low"


def safe_video_size_mb(uploaded_file):
    try:
        return uploaded_file.size / (1024 * 1024)
    except Exception:
        return None


def grab_frame_at_ratio(video_path: str, ratio: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        ok, frame = cap.read()
        cap.release()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None

    idx = max(0, min(total - 1, int(total * ratio)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# =========================
# Main app
# =========================
def main():
    # Load model once (for status + disable analyze if missing)
    model_pack = load_model()
    model_ready = model_pack is not None
    device_label = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    model_status = "Loaded" if model_ready else "Missing model file (models/best_model.pth)"

    # ===== Sidebar
    with st.sidebar:
        st.markdown("**CHOOSE YOUR MODE HERE**")
        st.header("Controls")

        mode_label = st.radio(
            "Detection mode",
            ["F1-Optimal (Balanced)", "Recall-Constrained (High Security)"]
        )
        mode = "f1" if "F1" in mode_label else "recall"

        with st.expander("Mode explanation", expanded=True):
            st.markdown(explain_mode(mode))

        st.divider()
        st.header("Metrics (indicative)")
        if mode == "f1":
            st.metric("Accuracy", "82.1%")
            st.metric("Fake Detection", "86.2%")
            st.metric("Real Detection", "77.8%")
        else:
            st.metric("Accuracy", "75.9%")
            st.metric("Fake Detection", "91.1%")
            st.metric("Real Detection", "59.3%")

        st.caption(f"Device: {device_label} · Model: {model_status}")

    # Push the top cards slightly lower
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ===== Hero + Quick summary
    hleft, hright = st.columns([1.55, 1.0], gap="large")

    with hleft:
        st.markdown(
            """
<div class="hero">
  <div class="hero-title">AI Video Detector</div>
  <div class="hero-sub">Detect AI-generated or manipulated videos using deep learning (ResNet50 + LSTM).</div>
  <div class="badges">
    <div class="badge">Binary output</div>
    <div class="badge">Confidence score</div>
    <div class="badge">Local inference</div>
    <div class="badge">MP4 / AVI / MOV / MKV / WebM</div>
  </div>
</div>
""",
            unsafe_allow_html=True
        )

    with hright:
        st.markdown(
            """
<div class="card">
  <div class="card-title">Quick summary</div>
  <div class="muted" style="margin-bottom:0.55rem;">
    Run a local check in seconds. Pick a mode based on your risk tolerance.
  </div>
  <ol class="muted" style="margin:0; padding-left: 1.2rem;">
    <li><b>Upload</b> a video (≤ 200 MB)</li>
    <li><b>Select</b> a detection mode</li>
    <li><b>Analyze</b> and read confidence</li>
  </ol>
  <div class="muted" style="margin-top:0.75rem;"><b>Tip:</b> Avoid heavy compression.</div>
</div>
""",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ===== Main body
    left, right = st.columns([1.6, 1.0], gap="large")

    with left:
        st.markdown(
            """
<div class="card">
  <div class="card-title">Upload and Analyze</div>
  <div class="muted">Upload a video file, then run the detector. Results appear below.</div>
</div>
""",
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Upload video",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            label_visibility="collapsed"
        )

        video_path = None
        if uploaded_file:
            file_size_mb = safe_video_size_mb(uploaded_file)
            if (file_size_mb is not None) and (file_size_mb > MAX_MB):
                st.error(f"File too large: {file_size_mb:.1f} MB. Max allowed is {MAX_MB:.0f} MB.")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    video_path = tmp.name

        analyze_disabled = (video_path is None) or (not model_ready)
        if not model_ready:
            st.warning("Model file is missing. Please add `models/best_model.pth` to enable analysis.")

        analyze = st.button("Analyze video", type="primary", disabled=analyze_disabled)

        if video_path:
            f1 = grab_frame_at_ratio(video_path, 0.05)
            f2 = grab_frame_at_ratio(video_path, 0.50)
            f3 = grab_frame_at_ratio(video_path, 0.95)

            st.markdown('<div class="card-title" style="margin-top:0.7rem;">Preview</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            if f1 is not None: c1.image(f1, caption="Start", use_container_width=True)
            if f2 is not None: c2.image(f2, caption="Middle", use_container_width=True)
            if f3 is not None: c3.image(f3, caption="End", use_container_width=True)

        if analyze and video_path and model_ready:
            status = st.empty()
            step = st.progress(0)

            try:
                status.info("1/3 Sampling frames...")
                step.progress(25)

                status.info("2/3 Running inference...")
                step.progress(60)

                label, confidence, threshold = infer(video_path, mode)

                status.info("3/3 Preparing output...")
                step.progress(90)

                bucket = confidence_bucket(confidence)
                cls = "real" if label == "Real" else "fake"
                title = "REAL VIDEO" if label == "Real" else "AI-GENERATED / MANIPULATED"

                st.markdown(
                    f"""
<div class="card result {cls}">
  <div class="card-title">{title}</div>
  <div class="muted">Mode: <b>{mode_label}</b> · Threshold: <b>{threshold}</b></div>
  <div style="margin-top:0.55rem;"><b>Confidence:</b> {confidence:.1%} <span class="muted">({bucket})</span></div>
</div>
""",
                    unsafe_allow_html=True
                )

                st.progress(float(max(0.0, min(1.0, confidence))))
                status.success("Done.")
                step.progress(100)

            except Exception as e:
                status.error(f"Analysis failed: {e}")
                step.empty()
            finally:
                try:
                    os.unlink(video_path)
                except:
                    pass

    with right:
        tabs = st.tabs(["How it works", "Technical", "Examples"])

        with tabs[0]:
            st.markdown('<div class="card"><div class="card-title">How it works</div>', unsafe_allow_html=True)
            st.markdown(
                """
1. **Upload** your video file  
2. **Choose** detection mode  
3. Click **Analyze** to process  
4. **Review** the results  

**The system automatically:**
- Extracts video frames  
- Analyzes temporal patterns  
- Applies AI detection  
- Provides confidence scores  
                """
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with tabs[1]:
            st.markdown('<div class="card"><div class="card-title">Technical Details</div>', unsafe_allow_html=True)
            st.markdown(
                """
- **Model:** ResNet50 + LSTM  
- **Training:** Focal Loss + Class Balancing  
- **Accuracy:** ~82%+ on unseen data (indicative)  
- **Supports:** Multiple video formats  
- **Privacy:** Local inference only  
                """
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with tabs[2]:
            st.markdown(
                """
<div class="card">
  <div class="card-title">Sample Results</div>
  <div class="muted">Examples are illustrative (same 2-class output).</div>
</div>
""",
                unsafe_allow_html=True
            )

            examples = [
                {"title": "Real Video", "label": "Real", "conf": 0.78, "note": "Stable motion and natural textures."},
                {"title": "AI Content", "label": "Fake", "conf": 0.91, "note": "Temporal inconsistencies detected."},
                {"title": "Hybrid Content", "label": "Fake", "conf": 0.62, "note": "Ambiguous content; manual review recommended."},
            ]

            for ex in examples:
                with st.expander(ex["title"]):
                    x, y, z = st.columns(3)
                    x.metric("Prediction", ex["label"])
                    y.metric("Confidence", f"{ex['conf']:.0%}")
                    z.metric("Level", confidence_bucket(ex["conf"]))
                    st.progress(float(ex["conf"]))
                    st.caption(ex["note"])


if __name__ == "__main__":
    main()
