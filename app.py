import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import subprocess
import time

# Import your existing model and inference functions
from src.model import VideoDetector
from scripts.infer import extract_frames, infer

# Model caching for better performance
@st.cache_resource
def load_model():
    """Load the AI detector model once and cache it"""
    try:
        model_path = 'models/best_model.pth'
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.info("Please ensure the model is trained and saved to models/best_model.pth")
            return None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VideoDetector()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Page configuration
st.set_page_config(
    page_title="AI Video Detector",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid;
        margin: 1rem 0;
    }
    .real-result {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    .fake-result {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        text-align: center;
        margin: 0.5rem;
    }
    .upload-section {
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎥 AI Video Detector</h1>', unsafe_allow_html=True)
    st.markdown("### Detect AI-generated and manipulated videos with advanced deep learning")

    # Sidebar with information
    with st.sidebar:
        st.header("📊 About")
        st.info("""
        This AI detector uses a ResNet50 + LSTM architecture trained on thousands of videos
        to distinguish between real and AI-generated content.

        **Features:**
        - Supports multiple video formats
        - Two detection modes
        - Real-time analysis
        - High accuracy on unseen data
        """)

        st.header("🎯 Detection Modes")
        mode_info = st.radio(
            "Choose detection mode:",
            ["F1-Optimal (Balanced)", "Recall-Constrained (High Security)"],
            help="F1-Optimal: Best balance | Recall-Constrained: Catches more fakes"
        )

        # Map to internal mode names
        mode = "f1" if "F1-Optimal" in mode_info else "recall"

        st.header("📈 Performance Metrics")
        if mode == "f1":
            st.metric("Accuracy", "82.1%")
            st.metric("Fake Detection", "86.2%")
            st.metric("Real Detection", "77.8%")
        else:
            st.metric("Accuracy", "75.9%")
            st.metric("Fake Detection", "91.1%")
            st.metric("Real Detection", "59.3%")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Your Video")
        st.markdown("Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WebM")

        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'],
            help="Upload a video to analyze for AI manipulation"
        )

        if uploaded_file is not None:
            st.markdown('</div>', unsafe_allow_html=True)

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name

            # Display video info
            st.success(f"✅ Video uploaded: {uploaded_file.name}")

            # Show video preview (first frame)
            try:
                cap = cv2.VideoCapture(video_path)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="Video Preview (First Frame)", use_column_width=True)
                cap.release()
            except Exception as e:
                st.warning(f"Could not generate preview: {e}")

            # Analysis button
            if st.button("🔍 Analyze Video", type="primary", use_container_width=True):
                with st.spinner("🎬 Analyzing video... This may take a few moments."):

                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        # Step 1: Extract frames
                        status_text.text("📹 Extracting video frames...")
                        progress_bar.progress(25)

                        # Step 2: Load model and analyze
                        status_text.text("🧠 Running AI detection...")
                        progress_bar.progress(50)

                        # Run inference
                        label, confidence, threshold = infer(video_path, mode)

                        progress_bar.progress(75)
                        status_text.text("📊 Processing results...")

                        # Step 3: Display results
                        progress_bar.progress(100)
                        status_text.empty()

                        # Result display
                        result_class = "real-result" if label == "Real" else "fake-result"
                        st.markdown(f'''
                        <div class="result-box {result_class}">
                            <h2 style="margin-top: 0;">{"✅ REAL VIDEO" if label == "Real" else "🚨 AI-GENERATED CONTENT"}</h2>
                            <p><strong>Confidence:</strong> {confidence:.1%}</p>
                            <p><strong>Detection Mode:</strong> {mode_info}</p>
                            <p><strong>Threshold Used:</strong> {threshold}</p>
                        </div>
                        ''', unsafe_allow_html=True)

                        # Detailed metrics
                        st.subheader("📈 Detailed Analysis")

                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Prediction", label)
                        with col_b:
                            st.metric("Confidence", f"{confidence:.1%}")
                        with col_c:
                            st.metric("Threshold", threshold)

                        # Explanation
                        st.subheader("💡 What This Means")
                        if label == "Real":
                            st.info("""
                            **Real Video Detected**
                            The analysis suggests this video contains authentic, unmodified content.
                            However, very sophisticated AI manipulations might still evade detection.
                            """)
                        else:
                            st.warning("""
                            **AI Content Detected**
                            The video appears to contain AI-generated or manipulated content.
                            This could include deepfakes, AI-generated scenes, or other synthetic media.
                            """)

                        # Mode-specific explanation
                        if mode == "f1":
                            st.info("**F1-Optimal Mode**: Provides the best balance between detecting fakes and avoiding false alarms.")
                        else:
                            st.warning("**High-Security Mode**: Optimized to catch as many fake videos as possible, but may flag some real videos.")

                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.info("Try uploading a different video or check the file format.")

                    finally:
                        # Clean up
                        try:
                            os.unlink(video_path)
                        except:
                            pass
                        progress_bar.empty()

        else:
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 🎯 How It Works")
        st.markdown("""
        1. **Upload** your video file
        2. **Choose** detection mode
        3. **Click Analyze** to process
        4. **Review** the results

        The system automatically:
        - Extracts video frames
        - Analyzes temporal patterns
        - Applies AI detection
        - Provides confidence scores
        """)

        st.markdown("### 🔧 Technical Details")
        st.markdown("""
        - **Model**: ResNet50 + LSTM
        - **Training**: Focal Loss + Class Balancing
        - **Accuracy**: 82%+ on unseen data
        - **Supports**: Multiple video formats
        """)

        # Sample results showcase
        st.markdown("### 📊 Sample Results")
        sample_results = {
            "Real Video": {"confidence": 0.78, "mode": "F1-Optimal"},
            "AI Deepfake": {"confidence": 0.91, "mode": "Recall-Constrained"},
            "Hybrid Content": {"confidence": 0.67, "mode": "Recall-Constrained"}
        }

        for video_type, data in sample_results.items():
            with st.expander(f"📹 {video_type}"):
                st.metric("Confidence", f"{data['confidence']:.0%}")
                st.caption(f"Mode: {data['mode']}")

if __name__ == "__main__":
    main()
