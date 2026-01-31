#!/usr/bin/env python3
"""
Demo script showing both detection modes of the AI Video Detector
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from infer import infer

def demo_dual_mode(video_path):
    """
    Demonstrate both detection modes on the same video
    """
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return

    print("🎥 AI Video Detector - Dual Mode Demo")
    print("=" * 50)
    print(f"Testing video: {video_path}")
    print()

    # Test both modes
    modes = ['f1', 'recall']

    for mode in modes:
        try:
            label, confidence, threshold = infer(video_path, mode)

            mode_names = {
                'f1': 'F1-Optimal (Balanced)',
                'recall': 'Recall-Constrained (High Recall)'
            }

            print(f"📊 {mode_names[mode]}:")
            print(f"   Threshold: {threshold}")
            print(f"   Prediction: {label}")
            print(f"   Confidence: {confidence:.1%}")

            if mode == 'f1':
                print("   → Best overall balance (93.6% precision, 78.6% recall)")
            else:
                print("   → Maximizes fake detection (91.1% recall, 67.1% precision)")

            print()

        except Exception as e:
            print(f"❌ Error with {mode} mode: {str(e)}")
            print()

if __name__ == "__main__":
    # Example usage - replace with your video path
    test_video = "path/to/your/test_video.mp4"

    demo_dual_mode(test_video)

    print("💡 Usage Tips:")
    print("• F1 mode: Use for general applications (best balance)")
    print("• Recall mode: Use when missing fake videos is critical")
    print("• Run: python scripts/demo_dual_mode.py")
