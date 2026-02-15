import streamlit as st
import torch
import cv2
import tempfile
import os
import shutil
import base64
from pathlib import Path

from src.model import VideoDetector
from src.gdrive import get_model_path
from scripts.infer import infer


# =========================
# Download model from Google Drive if not available locally
# =========================
def ensure_infer_model_path():
    """
    Ensures the model file is available locally.
    Downloads from Google Drive if the file is missing/invalid.
    """
    base_dir = Path(__file__).resolve().parent
    dst_dir = base_dir / "models"
    dst = dst_dir / "best_model.pth"

    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        model_path = get_model_path()

        # If get_model_path returned a different path, copy it to the expected location
        if model_path != dst:
            shutil.copy2(model_path, dst)
        return None
    except Exception as e:
        return str(e)

MODEL_SETUP_ERROR = ensure_infer_model_path()


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
    except Exception:
        return None


# =========================
# Page config (emoji removed)
# =========================
st.set_page_config(
    page_title="AI Video Detector",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================
# Global styles (anti-empty-rectangles included)
# =========================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800;900&family=IBM+Plex+Sans:wght@400;500;600&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20,400,0,0&display=swap');

:root{
  --bg-1: #061726;
  --bg-2: #0f2b2a;
  --panel: rgba(255,255,255,0.06);
  --panel-strong: rgba(255,255,255,0.09);
  --border: rgba(170, 224, 209, 0.22);
  --text-soft: rgba(232,245,244,0.78);
  --accent: #58e4c7;
  --accent-2: #6ec9ff;
}

/* App background + layout */
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(80rem 60rem at 8% -12%, rgba(95, 220, 199, 0.22), rgba(0,0,0,0)),
    radial-gradient(70rem 54rem at 102% 8%, rgba(91, 161, 252, 0.20), rgba(0,0,0,0)),
    linear-gradient(145deg, var(--bg-1), var(--bg-2));
}
.block-container { padding-top: 2.2rem; padding-bottom: 5.5rem; max-width: 1200px; }

/* Typography */
html, body, .stApp { font-family: "IBM Plex Sans", sans-serif; }
h1, h2, h3, .hero-title, .card-title, .stMetricLabel, .stButton button { font-family: "Sora", sans-serif !important; }
/* Keep Streamlit icon glyphs from rendering as raw text like _arrow_right */
.material-symbols-rounded,
[class*="material-symbols"],
.material-icons {
  font-family: "Material Symbols Rounded" !important;
  font-style: normal;
  font-weight: 400;
}

/* HERO */
.hero-top-left{
    display: inline-flex;
    align-items: center;
    gap: 0.42rem;
    border: 1px solid var(--border);
    background: rgba(5, 24, 31, 0.45);
    border-radius: 999px;
    padding: 0.28rem 0.74rem;
    margin: 0 0 0.58rem 0.1rem;
    color: rgba(232,245,244,0.92);
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.22px;
}
.hero {
    border: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.04));
    border-radius: 20px;
    padding: 1.45rem 1.5rem;
    box-shadow: 0 12px 30px rgba(0,0,0,0.32);
    backdrop-filter: blur(5px);
}
.hero-brand{
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
}
.hero-copy{
    width: 100%;
}
.hero-title-row{
    display: flex;
    align-items: center;
    gap: 0.72rem;
    flex-wrap: nowrap;
}
.hero-logo{
    width: 64px;
    min-width: 64px;
    height: 64px;
    object-fit: cover;
    display: block;
    border-radius: 12px;
    border: 1px solid var(--border);
    box-shadow: 0 6px 14px rgba(0,0,0,0.28);
    background: rgba(2, 13, 24, 0.6);
    padding: 2px;
}
.hero-title{
    font-size: 2.35rem;
    font-weight: 900;
    margin:0;
    letter-spacing: 0.2px;
    line-height:1.08;
    color: #f3fffd;
}
.hero-brand-name{
    background: linear-gradient(90deg, #58e4c7 0%, #6ec9ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 14px rgba(88,228,199,0.20);
}
.hero-sub{ opacity: 0.86; margin-top: 0.45rem; margin-bottom: 0.95rem; font-size: 1.02rem; }

.badges{ display:flex; gap:0.55rem; flex-wrap:wrap; }
.badge{
    border: 1px solid var(--border);
    background: rgba(7, 28, 35, 0.48);
    border-radius: 999px;
    padding: 0.34rem 0.70rem;
    font-size: 0.90rem;
    opacity: 0.96;
}

/* CARDS */
.card{
    border: 1px solid var(--border);
    background: var(--panel);
    border-radius: 20px;
    padding: 1.10rem 1.20rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 24px rgba(0,0,0,0.22);
    backdrop-filter: blur(6px);
}
.card-title{ font-size: 1.08rem; font-weight: 800; margin-bottom: 0.65rem; }
.muted{ color: var(--text-soft); font-size:0.94rem; line-height:1.35; }

/* RESULTS */
.result{ border-left: 6px solid rgba(255,255,255,0.12); }
.result.real{ border-left-color: rgba(0,200,120,0.95); }
.result.fake{ border-left-color: rgba(255,80,80,0.95); }

/* Button */
.stButton > button[kind="primary"]{
    border-radius: 12px !important;
    font-weight: 800 !important;
    padding: 0.75rem 1rem !important;
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    border: none !important;
    color: #05202d !important;
    box-shadow: 0 8px 18px rgba(57, 189, 225, 0.28);
}
.stButton > button[kind="primary"]:hover{
    filter: brightness(1.06);
    transform: translateY(-1px);
}

/* Tabs + metrics */
[data-baseweb="tab"]{
  font-family: "Sora", sans-serif !important;
  font-weight: 700 !important;
}
[data-baseweb="tab-highlight"]{
  background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important;
}
[data-testid="stMetric"]{
  background: var(--panel-strong);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.5rem 0.65rem;
}

/* Hide sidebar and its toggle control entirely */
[data-testid="stSidebar"]{ display: none !important; }
[data-testid="collapsedControl"]{ display: none !important; }

/* Make expanders look nicer */
[data-testid="stExpander"]{
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  background: rgba(5,24,31,0.34) !important;
}

/* Footer credits */
.credits-footer{
  position: fixed;
  left: 0;
  right: 0;
  bottom: 0;
  z-index: 999;
  text-align: center;
  padding: 0.62rem 0.8rem;
  font-family: "Sora", sans-serif;
  font-weight: 600;
  letter-spacing: 0.2px;
  color: rgba(240, 255, 251, 0.90);
  border-top: 1px solid var(--border);
  background: rgba(4,16,24,0.78);
  backdrop-filter: blur(6px);
}

/* Motion */
@keyframes fadeRise {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
.hero, .card, .result { animation: fadeRise 0.42s ease-out both; }

/* Mobile */
@media (max-width: 760px){
  .block-container{ padding-top: 1.4rem; padding-bottom: 4.2rem; }
  .hero-top-left{ font-size: 0.74rem; padding: 0.24rem 0.62rem; margin-bottom: 0.5rem; }
  .hero-brand{ gap: 0.75rem; }
  .hero-title-row{ gap: 0.55rem; }
  .hero-logo{ width: 48px; min-width: 48px; height: 48px; border-radius: 10px; }
  .hero-title{ font-size: 1.78rem; line-height: 1.12; }
  .hero-sub{ font-size: 0.95rem; }
  .credits-footer{ font-size: 0.82rem; padding: 0.52rem 0.6rem; }
}

/* Hide occasional empty containers */
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
LOGO_CANDIDATES = [
    Path(__file__).resolve().parent / "assets" / "verivid-logo.png",
    Path(__file__).resolve().parent / "assets" / "logo.png",
]


def get_logo_data_uri() -> str | None:
    """Return data URI for a local logo image if available."""
    mime_by_ext = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
    }
    for logo_path in LOGO_CANDIDATES:
        if not logo_path.exists():
            continue
        ext = logo_path.suffix.lower()
        mime = mime_by_ext.get(ext)
        if not mime:
            continue
        b64 = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"
    return None

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
    model_issue = None
    if MODEL_SETUP_ERROR:
        model_issue = f"Startup download failed: {MODEL_SETUP_ERROR}"
    elif not model_ready:
        model_path = Path(__file__).resolve().parent / "models" / "best_model.pth"
        if not model_path.exists():
            model_issue = "Missing model file at models/best_model.pth"
        else:
            model_issue = "Model file exists but failed to load"

    model_status = "Loaded" if model_ready else "Unavailable"

    # Push the top cards slightly lower
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ===== Hero + Quick summary
    hleft, hright = st.columns([1.55, 1.0], gap="large")
    logo_data_uri = get_logo_data_uri()
    logo_html = (
        f'<img class="hero-logo" src="{logo_data_uri}" alt="VeriVid logo">'
        if logo_data_uri
        else ""
    )

    with hleft:
        st.markdown(
            f"""
<div class="hero-top-left">AI video detector platform</div>
<div class="hero">
  <div class="hero-brand">
    <div class="hero-copy">
      <div class="hero-title-row">
        {logo_html}
        <div class="hero-title"><span class="hero-brand-name">VeriVid</span> : AI video detector</div>
      </div>
      <div class="hero-sub">Detect AI-generated or manipulated videos using deep learning (ResNet50 + LSTM).</div>
      <div class="badges">
        <div class="badge">Binary output</div>
        <div class="badge">Confidence score</div>
        <div class="badge">Local inference</div>
        <div class="badge">MP4 / AVI / MOV / MKV / WebM</div>
      </div>
    </div>
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
    <li><b>Upload</b> a video (&lt;= 200 MB)</li>
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
  <div class="card-title">Detection Mode</div>
  <div class="muted">Pick your mode, then upload and analyze.</div>
</div>
""",
            unsafe_allow_html=True
        )

        mode_label = st.radio(
            "Detection mode",
            ["F1-Optimal (Balanced)", "Recall-Constrained (High Security)"],
            horizontal=True
        )
        mode = "f1" if "F1" in mode_label else "recall"

        with st.expander("Mode explanation", expanded=False):
            st.markdown(explain_mode(mode))

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
        st.caption(f"Runtime: {device_label} | Model: {model_status}")

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
            if model_issue:
                st.warning(f"Model unavailable: {model_issue}")
            else:
                st.warning("Model unavailable. Check logs for model download/load errors.")

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
  <div class="muted">Mode: <b>{mode_label}</b> | Threshold: <b>{threshold}</b></div>
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

    st.markdown(
        """
<div class="credits-footer">
  Developped by Wiam Bouimejane and Wafae Waaziz
</div>
""",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
