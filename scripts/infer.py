import torch
import cv2
import torchvision.transforms as T
from src.model import VideoDetector
from src.gdrive import get_model_path
import argparse
import os
import numpy as np
import subprocess
import tempfile
from pathlib import Path

# Try to import moviepy for better video support
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

# Check for FFMPEG availability
def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

FFMPEG_AVAILABLE = check_ffmpeg()

# ---------------- CONFIG ----------------
NUM_FRAMES = 30
IMG_SIZE = 224

# Threshold configurations
THRESHOLDS = {
    'f1': 0.420,      # F1-optimal: 85.44% F1, 93.62% precision, 78.57% recall
    'recall': 0.290   # Recall-constrained: 91.07% recall, 67.11% precision, 77.27% F1
}
# ----------------------------------------

def extract_frames_ffmpeg(video_path, num_frames=30):
    """Extract frames using FFMPEG directly - most reliable method"""
    try:
        # Get video duration using ffprobe
        probe_cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format',
            '-show_streams', video_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)

        if probe_result.returncode != 0:
            raise Exception(f"FFprobe failed: {probe_result.stderr}")

        import json
        probe_data = json.loads(probe_result.stdout)

        # Extract video info
        duration = float(probe_data['format']['duration'])
        video_stream = next(s for s in probe_data['streams'] if s['codec_type'] == 'video')
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        fps = eval(video_stream['r_frame_rate'])

        print(f"✓ FFMPEG opened video: {duration:.1f}s, {fps:.1f} FPS, {width}x{height}")

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract frames using ffmpeg
            frame_pattern = os.path.join(temp_dir, 'frame_%04d.png')
            extract_cmd = [
                'ffmpeg', '-i', video_path, '-vf',
                f'fps={num_frames}/{duration}', frame_pattern,
                '-v', 'quiet', '-y'
            ]

            result = subprocess.run(extract_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFMPEG extraction failed: {result.stderr}")

            # Read extracted frames
            frames = []
            for i in range(1, num_frames + 1):
                frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
                if os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)

            if len(frames) == 0:
                raise Exception("No frames were extracted")

            print(f"✓ FFMPEG extracted {len(frames)} frames")

            # Ensure we have exactly num_frames
            if len(frames) < num_frames:
                last_frame = frames[-1]
                while len(frames) < num_frames:
                    frames.append(last_frame.copy())

            return frames[:num_frames]

    except Exception as e:
        print(f"FFMPEG extraction failed: {e}")
        raise


def extract_frames_moviepy(video_path, num_frames=30):
    """Extract frames using MoviePy - good alternative"""
    if not MOVIEPY_AVAILABLE:
        raise Exception("MoviePy not available")

    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        fps = clip.fps

        print(f"✓ MoviePy opened video: {duration:.1f}s, {fps:.1f} FPS")

        # Sample frames uniformly
        frame_times = np.linspace(0, duration, num_frames, endpoint=False)
        frames = []

        for t in frame_times:
            try:
                frame = clip.get_frame(t)
                # MoviePy returns RGB, convert to numpy array
                frame = np.array(frame)
                frames.append(frame)
            except Exception as e:
                print(f"Warning: Failed to extract frame at {t:.2f}s: {e}")
                continue

        clip.close()

        if len(frames) == 0:
            raise Exception("No frames were extracted")

        print(f"✓ MoviePy extracted {len(frames)} frames")

        # Ensure we have exactly num_frames
        if len(frames) < num_frames:
            last_frame = frames[-1]
            while len(frames) < num_frames:
                frames.append(last_frame.copy())

        return frames[:num_frames]

    except Exception as e:
        print(f"MoviePy extraction failed: {e}")
        raise


def extract_frames_opencv(video_path, num_frames=30):
    """Extract frames using OpenCV - fast but limited codec support"""
    # Try different backends in order of preference
    backends = [
        cv2.CAP_FFMPEG,      # FFMPEG backend
        cv2.CAP_ANY,         # Any available backend
        cv2.CAP_DSHOW,       # DirectShow (Windows)
        cv2.CAP_MSMF         # Microsoft Media Foundation
    ]

    cap = None
    for backend in backends:
        try:
            cap = cv2.VideoCapture(video_path, backend)
            if cap.isOpened():
                print(f"✓ OpenCV opened video with backend: {backend}")
                break
        except:
            continue

    if cap is None or not cap.isOpened():
        # Try without specifying backend as last resort
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("OpenCV could not open video")

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"✓ OpenCV opened video: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")

    if total_frames <= 0 or total_frames > 10000:  # Sanity check
        # Fallback: try to read frames manually
        print("Using manual frame reading...")
        frame_count = 0
        while cap.isOpened() and frame_count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1
    else:
        # Sample frames uniformly
        frame_indices = torch.linspace(0, total_frames - 1, num_frames).long()
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx.item())
            ret, frame = cap.read()
            if ret and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise Exception("No frames extracted with OpenCV")

    print(f"✓ OpenCV extracted {len(frames)} frames")

    # Ensure we have exactly num_frames
    if len(frames) < num_frames:
        last_frame = frames[-1]
        while len(frames) < num_frames:
            frames.append(last_frame.copy())

    return frames[:num_frames]


def extract_frames(video_path, num_frames=30):
    """
    Universal video frame extraction - tries multiple methods automatically.
    Supports any video format by falling back through different extraction methods.
    """
    # Ensure it's a video file
    if not video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')):
        raise ValueError(f"Unsupported video format: {video_path}. Supported: MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Try extraction methods in order of reliability
    methods = [
        ("FFMPEG", extract_frames_ffmpeg, FFMPEG_AVAILABLE),
        ("MoviePy", extract_frames_moviepy, MOVIEPY_AVAILABLE),
        ("OpenCV", extract_frames_opencv, True)
    ]

    last_error = None

    for method_name, method_func, is_available in methods:
        if not is_available:
            print(f"⚠ Skipping {method_name} (not available)")
            continue

        try:
            print(f"🔄 Trying {method_name} extraction...")
            frames = method_func(video_path, num_frames)
            print(f"✅ Successfully extracted {len(frames)} frames using {method_name}")
            return frames

        except Exception as e:
            error_msg = f"{method_name} failed: {str(e)}"
            print(f"❌ {error_msg}")
            last_error = error_msg
            continue

    # If all methods failed
    raise RuntimeError(f"All video extraction methods failed. Last error: {last_error}\n\n"
                      "Troubleshooting:\n"
                      "1. Install FFMPEG: Download from https://ffmpeg.org/download.html and add to PATH\n"
                      "2. Install MoviePy: pip install moviepy\n"
                      "3. Try converting video to H.264 MP4: ffmpeg -i input.mp4 -c:v libx264 output.mp4\n"
                      "4. Check if video is corrupted or uses unsupported codec")


def infer(video_path, mode='f1'):
    """
    Infer video classification with different threshold modes.

    Args:
        video_path: Path to video file
        mode: 'f1' for F1-optimal (balanced) or 'recall' for recall-constrained (high recall)

    Returns:
        label: "Real" or "AI-generated"
        confidence: Probability score
        threshold_used: The threshold value used
    """
    if mode not in THRESHOLDS:
        raise ValueError(f"Invalid mode '{mode}'. Choose 'f1' or 'recall'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = THRESHOLDS[mode]
    model_path = str(get_model_path())

    # Load model
    model = VideoDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Same normalization as ResNet training
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    frames = extract_frames(video_path, NUM_FRAMES)
    frames = torch.stack([transform(f) for f in frames])
    frames = frames.unsqueeze(0).to(device)  # (1, T, C, H, W)

    with torch.no_grad():
        outputs = model(frames)
        probs = torch.softmax(outputs, dim=1)

        # Get probability of being fake (AI-generated)
        fake_prob = probs[0, 1].item()

        # Apply threshold decision
        if fake_prob >= threshold:
            label = "AI-generated"
            confidence = fake_prob
        else:
            label = "Real"
            confidence = 1 - fake_prob

    return label, confidence, threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AI Video Detector Inference')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--mode', choices=['f1', 'recall'], default='f1',
                       help='Detection mode: f1 (balanced) or recall (high recall)')

    args = parser.parse_args()

    # Run inference
    label, confidence, threshold = infer(args.video_path, args.mode)

    # Display results
    mode_names = {
        'f1': 'F1-Optimal (Balanced)',
        'recall': 'Recall-Constrained (High Recall)'
    }

    print("🎥 AI Video Detector Results")
    print("=" * 40)
    print(f"Video: {args.video_path}")
    print(f"Mode: {mode_names[args.mode]}")
    print(f"Threshold: {threshold}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.1%}")
    print()

    # Mode-specific explanations
    if args.mode == 'f1':
        print("📊 F1-Optimal Mode:")
        print("   • Best overall balance between precision and recall")
        print("   • 93.6% precision, 78.6% recall, 85.4% F1 score")
        print("   • Recommended for most applications")
    else:
        print("📊 Recall-Constrained Mode:")
        print("   • Maximizes fake video detection")
        print("   • 91.1% recall, 67.1% precision, 77.3% F1 score")
        print("   • Use when missing fake videos is critical")
