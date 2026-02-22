"""
Computer Vision Analysis Module for AI Video Detector.

This script explains model decisions by combining:
1. Threshold-based prediction analysis (aligned with inference mode)
2. Temporal frame importance
3. Spatial GradCAM attention maps
4. Frequency-domain cues

The goal is to answer: why this video was detected (or missed), what the model
focused on, and how the decision was produced.
"""

import sys
from pathlib import Path
import json
import textwrap
from typing import Dict, List, Optional
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.gdrive import get_model_path
from src.model import VideoDetector
from scripts.infer import THRESHOLDS, IMG_SIZE, NUM_FRAMES, center_crop_square, extract_frames

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-10


# ==================== SHARED HELPERS ====================

def _build_transform() -> T.Compose:
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


FRAME_TRANSFORM = _build_transform()


def preprocess_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Match inference preprocessing: center-crop each frame to square."""
    return [center_crop_square(frame) for frame in frames]


def frames_to_tensor(frames: List[np.ndarray]) -> torch.Tensor:
    """Convert RGB frames to model input tensor (1, T, C, H, W)."""
    frame_tensors = [FRAME_TRANSFORM(frame) for frame in frames]
    return torch.stack(frame_tensors).unsqueeze(0).to(DEVICE)


def get_threshold(mode: str) -> float:
    if mode not in THRESHOLDS:
        raise ValueError(f"Invalid mode '{mode}'. Expected one of: {list(THRESHOLDS)}")
    return float(THRESHOLDS[mode])


def load_video_detector() -> VideoDetector:
    """Load trained detector lazily (no import-time download side effects)."""
    model_path = str(get_model_path())
    model = VideoDetector().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def describe_frequency_level(high_freq_ratio: float) -> str:
    if high_freq_ratio >= 0.35:
        return "high"
    if high_freq_ratio >= 0.20:
        return "moderate"
    return "low"


def describe_temporal_peak(temporal_distribution: Dict[str, float]) -> str:
    return max(temporal_distribution, key=temporal_distribution.get)


def describe_spatial_focus(attention_stat: Dict[str, float]) -> str:
    area_ratio = attention_stat["hot_region_ratio"]
    cx = attention_stat["hot_centroid_x"]
    cy = attention_stat["hot_centroid_y"]

    if area_ratio < 0.08:
        spread = "very localized"
    elif area_ratio < 0.20:
        spread = "moderately localized"
    else:
        spread = "diffuse"

    if 0.33 <= cx <= 0.66 and 0.33 <= cy <= 0.66:
        location = "center"
    else:
        location = "off-center"

    return f"{spread} and mostly {location}"


def build_conclusion(
    prediction_result: Dict,
    temporal_result: Dict,
    frequency_result: Dict,
    gradcam_result: Dict,
) -> str:
    top_frame = temporal_result["most_important_frame"]
    top_attention_stat = gradcam_result["frame_attention_stats"][top_frame]

    temporal_peak = describe_temporal_peak(temporal_result["temporal_distribution"])
    frequency_level = describe_frequency_level(frequency_result["mean_high_freq_ratio"])
    spatial_focus = describe_spatial_focus(top_attention_stat)

    margin = prediction_result["decision_margin"]
    if prediction_result["prediction"] == "AI-generated":
        decision_line = f"Fake probability is above threshold by {margin:.4f}."
    else:
        decision_line = f"Fake probability is below threshold by {abs(margin):.4f}."

    truth_line = ""
    if prediction_result["is_correct"] is not None:
        truth_line = f" Ground truth match: {prediction_result['is_correct']}."

    return (
        f"Decision: {prediction_result['prediction']} (confidence {prediction_result['confidence']:.1%}). "
        f"{decision_line} The model relied most on frame {top_frame} and attention was {spatial_focus}. "
        f"Temporal evidence is strongest in the {temporal_peak} segment. "
        f"Frequency content is {frequency_level} in high-frequency bands."
        f"{truth_line}"
    )


# ==================== GRADCAM IMPLEMENTATION ====================

class GradCAM:
    """GradCAM for the CNN backbone inside VideoDetector."""

    def __init__(self, model: VideoDetector):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_module, _inputs, output):
            self.activations = output.detach()

        def backward_hook(_module, _grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer = self.model.cnn.layer4[-1].conv2
        self._hook_handles.append(target_layer.register_forward_hook(forward_hook))
        self._hook_handles.append(target_layer.register_full_backward_hook(backward_hook))

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _normalize(cam: np.ndarray) -> np.ndarray:
        cam = cam - cam.min()
        max_val = cam.max()
        if max_val > 0:
            cam = cam / max_val
        return cam

    @staticmethod
    def _summarize_attention(cam: np.ndarray) -> Dict[str, float]:
        hot_threshold = float(np.percentile(cam, 90))
        hot_mask = cam >= hot_threshold
        hot_ratio = float(hot_mask.mean())

        ys, xs = np.where(hot_mask)
        if len(xs) == 0 or len(ys) == 0:
            return {
                "hot_region_ratio": 0.0,
                "hot_centroid_x": 0.5,
                "hot_centroid_y": 0.5,
            }

        h, w = cam.shape[:2]
        return {
            "hot_region_ratio": hot_ratio,
            "hot_centroid_x": float(xs.mean() / max(1, (w - 1))),
            "hot_centroid_y": float(ys.mean() / max(1, (h - 1))),
        }

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        output = self.model(input_tensor)

        self.model.zero_grad()
        output[0, target_class].backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks did not capture gradients/activations.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze(1)

        cam_np = cam.detach().cpu().numpy()
        return np.array([self._normalize(single_cam) for single_cam in cam_np])

    def analyze_video(self, frames: List[np.ndarray], target_class: int) -> Dict:
        input_tensor = frames_to_tensor(frames)
        cams = self.generate_cam(input_tensor, target_class)

        h, w = frames[0].shape[:2]
        resized_cams = [cv2.resize(cam, (w, h)) for cam in cams]
        frame_attention_stats = [self._summarize_attention(cam) for cam in resized_cams]

        return {
            "cams": resized_cams,
            "target_class": int(target_class),
            "frame_attention_stats": frame_attention_stats,
        }


# ==================== TEMPORAL ANALYSIS ====================

class TemporalAnalyzer:
    """Find which frames contributed most to a selected class decision."""

    def __init__(self, model: VideoDetector):
        self.model = model
        self.model.eval()

    def compute_frame_importance(self, frames: List[np.ndarray], target_class: int) -> np.ndarray:
        frames_tensor = frames_to_tensor(frames)
        frames_tensor.requires_grad_(True)

        output = self.model(frames_tensor)

        self.model.zero_grad()
        output[0, target_class].backward()

        if frames_tensor.grad is None:
            raise RuntimeError("Unable to compute frame gradients for temporal analysis.")

        grad = frames_tensor.grad[0]
        importance = grad.abs().mean(dim=(1, 2, 3)).detach().cpu().numpy()

        total = float(importance.sum())
        if total <= EPS:
            importance = np.full(len(importance), 1.0 / max(1, len(importance)), dtype=np.float64)
        else:
            importance = importance / total

        return importance

    def analyze_temporal_patterns(self, frames: List[np.ndarray], target_class: int) -> Dict:
        importances = self.compute_frame_importance(frames, target_class)

        n = len(importances)
        cut1 = n // 3
        cut2 = 2 * n // 3

        top_k = min(3, n)
        top_indices = np.argsort(importances)[-top_k:][::-1]

        return {
            "importances": importances,
            "most_important_frame": int(np.argmax(importances)),
            "least_important_frame": int(np.argmin(importances)),
            "most_important_value": float(importances.max()),
            "least_important_value": float(importances.min()),
            "top_frames": [int(i) for i in top_indices],
            "temporal_distribution": {
                "early": float(importances[:cut1].sum()),
                "middle": float(importances[cut1:cut2].sum()),
                "late": float(importances[cut2:].sum()),
            },
        }


# ==================== FREQUENCY DOMAIN ANALYSIS ====================

class FrequencyAnalyzer:
    """Compute robust low/mid/high frequency energy statistics for each frame."""

    @staticmethod
    def compute_fft_features(frame: np.ndarray) -> Dict:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        magnitude_log = np.log1p(magnitude)

        h, w = gray.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_norm = r / (r.max() + EPS)

        low_mask = r_norm <= 0.33
        mid_mask = (r_norm > 0.33) & (r_norm < 0.66)
        high_mask = r_norm >= 0.66

        total_energy = float(magnitude.sum() + EPS)
        low_energy = float(magnitude[low_mask].sum())
        mid_energy = float(magnitude[mid_mask].sum())
        high_energy = float(magnitude[high_mask].sum())

        r_int = r.astype(np.int32)
        radial_sum = np.bincount(r_int.ravel(), weights=magnitude.ravel())
        radial_count = np.bincount(r_int.ravel())
        radial_profile = radial_sum / np.maximum(radial_count, 1)
        radial_profile = radial_profile.astype(np.float64)
        if radial_profile.max() > 0:
            radial_profile /= radial_profile.max()

        center_radius = max(1, min(h, w) // 40)
        center_patch = magnitude[
            max(0, cy - center_radius) : min(h, cy + center_radius + 1),
            max(0, cx - center_radius) : min(w, cx + center_radius + 1),
        ]

        return {
            "magnitude_log": magnitude_log,
            "radial_profile": radial_profile,
            "low_freq_ratio": low_energy / total_energy,
            "mid_freq_ratio": mid_energy / total_energy,
            "high_freq_ratio": high_energy / total_energy,
            "center_frequency_energy": float(center_patch.mean()) if center_patch.size > 0 else 0.0,
        }

    def analyze_video(self, frames: List[np.ndarray]) -> Dict:
        frame_features = []

        for idx, frame in enumerate(frames):
            features = self.compute_fft_features(frame)
            frame_features.append(
                {
                    "frame_idx": idx,
                    "low_freq_ratio": features["low_freq_ratio"],
                    "mid_freq_ratio": features["mid_freq_ratio"],
                    "high_freq_ratio": features["high_freq_ratio"],
                    "center_energy": features["center_frequency_energy"],
                }
            )

        high_freq_ratios = np.array([item["high_freq_ratio"] for item in frame_features], dtype=np.float64)
        low_freq_ratios = np.array([item["low_freq_ratio"] for item in frame_features], dtype=np.float64)

        return {
            "frame_features": frame_features,
            "mean_high_freq_ratio": float(high_freq_ratios.mean()),
            "std_high_freq_ratio": float(high_freq_ratios.std()),
            "mean_low_freq_ratio": float(low_freq_ratios.mean()),
            "freq_temporal_variation": float(high_freq_ratios.std()),
        }


# ==================== PREDICTION + ERROR ANALYSIS ====================

class ErrorAnalyzer:
    """Explain predictions with consistent confidence and decision margin."""

    def __init__(self, model: VideoDetector):
        self.model = model
        self.model.eval()

    def _predict(self, frames: List[np.ndarray], mode: str) -> Dict:
        threshold = get_threshold(mode)
        input_tensor = frames_to_tensor(frames)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)

        real_prob = float(probs[0, 0].item())
        fake_prob = float(probs[0, 1].item())

        predicted_fake = fake_prob >= threshold
        predicted_label = "AI-generated" if predicted_fake else "Real"
        confidence = fake_prob if predicted_fake else real_prob
        target_class = 1 if predicted_fake else 0

        return {
            "prediction": predicted_label,
            "target_class": target_class,
            "threshold_used": threshold,
            "decision_margin": float(fake_prob - threshold),
            "confidence": float(confidence),
            "probabilities": {"real": real_prob, "fake": fake_prob},
        }

    def analyze_prediction(
        self,
        frames: List[np.ndarray],
        true_label: Optional[str] = None,
        mode: str = "f1",
    ) -> Dict:
        prediction = self._predict(frames, mode)

        temporal_analyzer = TemporalAnalyzer(self.model)
        temporal_analysis = temporal_analyzer.analyze_temporal_patterns(frames, prediction["target_class"])

        freq_analyzer = FrequencyAnalyzer()
        freq_analysis = freq_analyzer.analyze_video(frames)

        is_correct = None
        if true_label is not None:
            is_correct = prediction["prediction"] == true_label

        return {
            "prediction": prediction["prediction"],
            "true_label": true_label,
            "is_correct": is_correct,
            "confidence": prediction["confidence"],
            "probabilities": prediction["probabilities"],
            "threshold_used": prediction["threshold_used"],
            "decision_margin": prediction["decision_margin"],
            "target_class": prediction["target_class"],
            "mode": mode,
            "temporal_analysis": temporal_analysis,
            "frequency_analysis": freq_analysis,
        }

    def compare_videos(
        self,
        video1_frames: List[np.ndarray],
        video1_label: Optional[str],
        video2_frames: List[np.ndarray],
        video2_label: Optional[str],
        mode: str = "f1",
    ) -> Dict:
        analysis1 = self.analyze_prediction(video1_frames, video1_label, mode)
        analysis2 = self.analyze_prediction(video2_frames, video2_label, mode)

        return {
            "video1": analysis1,
            "video2": analysis2,
            "comparison": {
                "confidence_diff": abs(analysis1["confidence"] - analysis2["confidence"]),
                "fake_prob_diff": abs(
                    analysis1["probabilities"]["fake"] - analysis2["probabilities"]["fake"]
                ),
                "freq_ratio_diff": abs(
                    analysis1["frequency_analysis"]["mean_high_freq_ratio"]
                    - analysis2["frequency_analysis"]["mean_high_freq_ratio"]
                ),
                "temporal_var_diff": abs(
                    analysis1["frequency_analysis"]["freq_temporal_variation"]
                    - analysis2["frequency_analysis"]["freq_temporal_variation"]
                ),
            },
        }


# ==================== REPORTING ====================

def compare_analysis_results(result_a: Dict, result_b: Dict) -> Dict:
    pred_a = result_a["prediction"]
    pred_b = result_b["prediction"]
    freq_a = result_a["frequency"]
    freq_b = result_b["frequency"]

    comparison = {
        "video_a": result_a["video_path"],
        "video_b": result_b["video_path"],
        "prediction_diff": {
            "video_a_label": pred_a["prediction"],
            "video_b_label": pred_b["prediction"],
            "fake_probability_gap": abs(pred_a["probabilities"]["fake"] - pred_b["probabilities"]["fake"]),
            "confidence_gap": abs(pred_a["confidence"] - pred_b["confidence"]),
        },
        "frequency_diff": {
            "high_freq_ratio_gap": abs(
                freq_a["mean_high_freq_ratio"] - freq_b["mean_high_freq_ratio"]
            ),
            "high_freq_variation_gap": abs(
                freq_a["freq_temporal_variation"] - freq_b["freq_temporal_variation"]
            ),
        },
        "temporal_diff": {
            "video_a_top_frame": pred_a["temporal_analysis"]["most_important_frame"],
            "video_b_top_frame": pred_b["temporal_analysis"]["most_important_frame"],
        },
    }

    comparison["conclusion"] = (
        "The model separated these videos using a combination of threshold margin, "
        "frame-level temporal importance, and high-frequency behavior. "
        "Use each video's conclusion text for detailed per-video reasoning."
    )
    return comparison


# ==================== MAIN ANALYSIS FUNCTION ====================

def analyze_video_comprehensive(
    video_path: str,
    true_label: Optional[str] = None,
    output_dir: str = "results/cv_analysis",
    mode: str = "f1",
) -> Dict:
    """Run full CV analysis and produce plot + JSON report."""
    print(f"\n{'=' * 60}")
    print("COMPREHENSIVE COMPUTER VISION ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Video: {video_path}")
    print(f"Mode: {mode}")
    print(f"True Label: {true_label}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n1. Loading model...")
    model = load_video_detector()

    print("2. Extracting and preprocessing frames...")
    frames = preprocess_frames(extract_frames(video_path, NUM_FRAMES))
    print(f"   Extracted {len(frames)} frames")

    print("3. Running prediction + temporal + frequency analysis...")
    error_analyzer = ErrorAnalyzer(model)
    prediction_result = error_analyzer.analyze_prediction(frames, true_label, mode)

    temporal_result = prediction_result["temporal_analysis"]
    freq_result = prediction_result["frequency_analysis"]

    print("4. Running GradCAM for predicted class...")
    gradcam = GradCAM(model)
    try:
        gradcam_result = gradcam.analyze_video(frames, target_class=prediction_result["target_class"])
    finally:
        gradcam.close()

    print("5. Building evidence-based conclusion...")
    conclusion = build_conclusion(prediction_result, temporal_result, freq_result, gradcam_result)

    # -------------------- Visualization --------------------
    print("6. Generating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    ax = axes[0, 0]
    frame_idx = np.arange(len(temporal_result["importances"]))
    ax.bar(frame_idx, temporal_result["importances"], color="steelblue", alpha=0.75)
    ax.axhline(
        y=1 / len(temporal_result["importances"]),
        color="red",
        linestyle="--",
        label="Uniform importance",
    )
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Importance Score")
    ax.set_title("Temporal Importance", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)

    ax = axes[0, 1]
    freq_frame_idx = [item["frame_idx"] for item in freq_result["frame_features"]]
    high_freq = [item["high_freq_ratio"] for item in freq_result["frame_features"]]
    ax.plot(freq_frame_idx, high_freq, "g-", linewidth=2, marker="o", markersize=4)
    ax.axhline(
        y=freq_result["mean_high_freq_ratio"],
        color="red",
        linestyle="--",
        label=f"Mean: {freq_result['mean_high_freq_ratio']:.3f}",
    )
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("High-Frequency Energy Ratio")
    ax.set_title("Frequency-Domain Trend", fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)

    ax = axes[1, 0]
    key_frame_idx = temporal_result["most_important_frame"]
    key_frame = frames[key_frame_idx]
    key_cam = gradcam_result["cams"][key_frame_idx]
    cam_colored = plt.cm.jet(key_cam)[:, :, :3]
    overlay = (0.6 * key_frame / 255.0) + (0.4 * cam_colored)
    ax.imshow((overlay * 255).astype(np.uint8))
    ax.set_title(f"GradCAM Overlay (Frame {key_frame_idx})", fontsize=18, fontweight="bold")
    ax.axis("off")

    ax = axes[1, 1]
    ax.axis("off")
    summary_text = "ANALYSIS SUMMARY\n" + "=" * 30 + "\n\n"
    summary_text += f"Prediction: {prediction_result['prediction']}\n"
    summary_text += f"Confidence: {prediction_result['confidence']:.1%}\n"
    summary_text += f"Decision margin: {prediction_result['decision_margin']:.4f}\n"
    summary_text += f"Most important frame: {temporal_result['most_important_frame']}\n"
    summary_text += f"Top frames: {temporal_result['top_frames']}\n"
    summary_text += f"Mean high-freq ratio: {freq_result['mean_high_freq_ratio']:.4f}\n"
    summary_text += f"Temporal dist (E/M/L): {temporal_result['temporal_distribution']['early']:.2f} / "
    summary_text += f"{temporal_result['temporal_distribution']['middle']:.2f} / "
    summary_text += f"{temporal_result['temporal_distribution']['late']:.2f}\n\n"
    summary_text += "Conclusion:\n"
    summary_text += textwrap.fill(conclusion, width=62)

    ax.text(
        0.02,
        0.98,
        summary_text,
        transform=ax.transAxes,
        fontsize=12,
        va="top",
        fontfamily="monospace",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout(pad=2.0)
    output_path = Path(output_dir) / f"{Path(video_path).stem}_cv_analysis.png"
    plt.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close()
    print(f"   Saved visualization to {output_path}")

    results = {
        "video_path": video_path,
        "true_label": true_label,
        "mode": mode,
        "prediction": prediction_result,
        "temporal": temporal_result,
        "frequency": {
            "mean_high_freq_ratio": freq_result["mean_high_freq_ratio"],
            "std_high_freq_ratio": freq_result["std_high_freq_ratio"],
            "mean_low_freq_ratio": freq_result["mean_low_freq_ratio"],
            "freq_temporal_variation": freq_result["freq_temporal_variation"],
        },
        "gradcam": {
            "num_frames": len(gradcam_result["cams"]),
            "target_class": gradcam_result["target_class"],
            "frame_attention_stats": gradcam_result["frame_attention_stats"],
        },
        "conclusion": conclusion,
        "processing_trace": [
            "video -> frame extraction",
            "center-crop + normalization",
            "CNN features + LSTM temporal fusion",
            "threshold-based decision",
            "GradCAM + temporal gradients + FFT cues for explanation",
        ],
        "visualization_path": str(output_path),
    }

    json_path = Path(output_dir) / f"{Path(video_path).stem}_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"   Saved results to {json_path}")

    print("\nFinal conclusion:")
    print(conclusion)
    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Computer Vision Analysis for AI Video Detector")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument(
        "--true_label",
        choices=["Real", "AI-generated"],
        help="Ground truth label (optional)",
    )
    parser.add_argument("--mode", choices=["f1", "recall"], default="f1", help="Detection mode")
    parser.add_argument(
        "--output_dir",
        default="results/cv_analysis",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--compare_with",
        help="Optional second video path for side-by-side analysis",
    )
    parser.add_argument(
        "--compare_with_label",
        choices=["Real", "AI-generated"],
        help="Ground truth label for --compare_with video (optional)",
    )

    args = parser.parse_args()

    result_a = analyze_video_comprehensive(
        args.video_path,
        true_label=args.true_label,
        mode=args.mode,
        output_dir=args.output_dir,
    )

    if args.compare_with:
        result_b = analyze_video_comprehensive(
            args.compare_with,
            true_label=args.compare_with_label,
            mode=args.mode,
            output_dir=args.output_dir,
        )

        comparison = compare_analysis_results(result_a, result_b)
        compare_name = f"compare_{Path(args.video_path).stem}_vs_{Path(args.compare_with).stem}.json"
        compare_path = Path(args.output_dir) / compare_name
        with open(compare_path, "w", encoding="utf-8") as f:
            json.dump(convert_to_serializable(comparison), f, indent=2)

        print("\nComparison summary:")
        print(comparison["conclusion"])
        print(f"Comparison file saved to {compare_path}")
