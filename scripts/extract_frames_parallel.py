"""
Pre-extract all frames from videos using parallel CPU workers.
Trades disk space for training speed (50-70% faster training).

One-time preprocessing step before training.
Expected runtime: ~20-30 minutes (uses all CPU cores).
Output: data/frames/ directory with pre-extracted frame arrays.
"""

import json
import os
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

def extract_frames_from_video(video_path, output_dir, num_frames=30):
    """
    Extract num_frames evenly sampled frames from a video and save as .npy file.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frame arrays
        num_frames: Number of frames to extract
    
    Returns:
        (success, video_id, error_msg)
    """
    try:
        video_id = Path(video_path).stem
        output_file = os.path.join(output_dir, f"{video_id}.npy")
        
        # Skip if already extracted
        if os.path.exists(output_file):
            return True, video_id, None
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, video_id, "Cannot open video"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < num_frames:
            cap.release()
            return False, video_id, f"Video has {total_frames} frames, need {num_frames}"
        
        # Sample frame indices evenly
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                cap.release()
                return False, video_id, f"Cannot read frame {idx}"
            
            # Resize to 224x224 and convert BGR -> RGB
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        
        # Stack frames and save as .npy (more efficient than individual files)
        frames_array = np.stack(frames, axis=0)  # Shape: (30, 224, 224, 3)
        np.save(output_file, frames_array)
        
        return True, video_id, None
    
    except Exception as e:
        return False, video_id, str(e)


def extract_all_frames(data_dir='data', num_workers=None):
    """
    Extract frames from all videos using parallel processing.
    
    Args:
        data_dir: Root data directory
        num_workers: Number of parallel workers (default: CPU count)
    """
    
    # Determine number of workers
    if num_workers is None:
        num_workers = os.cpu_count()
    
    print(f"Using {num_workers} workers for parallel frame extraction\n")
    
    # Load splits
    splits_file = os.path.join(data_dir, 'splits.json')
    if not os.path.exists(splits_file):
        print(f"Error: {splits_file} not found. Run preprocess_data.py first.")
        return
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    # Create output directories
    frames_dir = os.path.join(data_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    # Collect all video paths (handle absolute and already-prefixed paths)
    all_videos = []
    for split_name in ['train', 'val', 'test']:
        for video_path, label in splits[split_name]:
            # If path is absolute or already starts with 'data', use as is
            if os.path.isabs(video_path) or video_path.startswith('data'):
                full_path = video_path
            else:
                full_path = os.path.join(data_dir, video_path)
            all_videos.append(full_path)
    
    print(f"Total videos to process: {len(all_videos)}\n")
    
    # Extract frames in parallel
    success_count = 0
    failed_videos = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(extract_frames_from_video, video_path, frames_dir): video_path
            for video_path in all_videos
        }
        
        with tqdm(as_completed(futures), total=len(futures), desc="Extracting frames") as pbar:
            for future in pbar:
                success, video_id, error = future.result()
                
                if success:
                    success_count += 1
                else:
                    failed_videos.append((video_id, error))
                
                pbar.update(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("FRAME EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"✓ Successfully extracted: {success_count}/{len(all_videos)}")
    print(f"✓ Output directory: {frames_dir}")
    print(f"✓ Each video: 30 frames × 224×224 × 3 channels")
    
    if failed_videos:
        print(f"\n⚠ Failed videos ({len(failed_videos)}):")
        for video_id, error in failed_videos[:10]:
            print(f"  - {video_id}: {error}")
        if len(failed_videos) > 10:
            print(f"  ... and {len(failed_videos) - 10} more")
    
    # Estimate disk usage
    frame_array_size = 30 * 224 * 224 * 3 * 4  # 4 bytes per float32
    total_size_gb = (success_count * frame_array_size) / (1024**3)
    print(f"\n📊 Disk usage: ~{total_size_gb:.1f} GB")
    print("\nNext step: Use updated dataset.py to load pre-extracted frames")


if __name__ == '__main__':
    import sys
    
    num_workers = int(sys.argv[1]) if len(sys.argv) > 1 else None
    extract_all_frames(num_workers=num_workers)
