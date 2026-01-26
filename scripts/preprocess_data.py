import os
import cv2
import glob
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import warnings

# Suppress OpenCV warnings about corrupted files
warnings.filterwarnings('ignore')
cv2.setLogLevel(0)  # Suppress OpenCV debug messages

def validate_videos(data_dir='data'):
    """Validate all videos and remove corrupted ones."""
    print("=" * 60)
    print("VALIDATING DATASET")
    print("=" * 60)
    
    real_dir = os.path.join(data_dir, 'real')
    fake_dir = os.path.join(data_dir, 'fake')
    
    valid_videos = {'real': [], 'fake': []}
    corrupted = {'real': [], 'fake': []}
    
    for category, directory in [('real', real_dir), ('fake', fake_dir)]:
        video_files = glob.glob(os.path.join(directory, '*.mp4'))
        print(f"\n{category.upper()} videos: {len(video_files)} found")
        
        for video_path in video_files:
            try:
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Try to read first frame to verify video is not corrupted
                ret, frame = cap.read()
                cap.release()
                
                # Check validity
                if not ret or frame is None:
                    raise ValueError("Could not read first frame")
                if frame_count < 30:
                    raise ValueError(f"Too short ({frame_count} frames, need 30+)")
                if fps <= 0:
                    raise ValueError(f"Invalid FPS: {fps}")
                if width <= 0 or height <= 0:
                    raise ValueError(f"Invalid dimensions: {width}x{height}")
                
                valid_videos[category].append(video_path)
                print(f"  ✓ {Path(video_path).name} ({frame_count} frames @ {fps:.1f} FPS, {width}x{height})")
            
            except Exception as e:
                corrupted[category].append(video_path)
                print(f"  ✗ {Path(video_path).name} - {str(e)}")
        
        print(f"  Summary: {len(valid_videos[category])} valid, {len(corrupted[category])} corrupted")
    
    return valid_videos, corrupted

def check_balance(valid_videos):
    """Check real/fake balance."""
    real_count = len(valid_videos['real'])
    fake_count = len(valid_videos['fake'])
    total = real_count + fake_count
    
    print("\n" + "=" * 60)
    print("DATA BALANCE CHECK")
    print("=" * 60)
    print(f"Real videos: {real_count} ({real_count/total*100:.1f}%)")
    print(f"Fake videos: {fake_count} ({fake_count/total*100:.1f}%)")
    print(f"Total videos: {total}")
    
    if abs(real_count - fake_count) / total > 0.2:
        print("\n  WARNING: Imbalanced classes (>20% difference)")
        print("   Consider collecting more videos to balance.")
    else:
        print("\n✓ Classes are well-balanced")

def create_splits(valid_videos, output_dir='data', train_ratio=0.7, val_ratio=0.15):
    """Create train/val/test splits and save metadata."""
    print("\n" + "=" * 60)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("=" * 60)
    
    test_ratio = 1 - train_ratio - val_ratio
    
    splits = {'train': [], 'val': [], 'test': []}
    
    for category in ['real', 'fake']:
        videos = valid_videos[category]
        
        # First split: train + val vs test
        train_val, test = train_test_split(
            videos, test_size=test_ratio, random_state=42
        )
        
        # Second split: train vs val
        train, val = train_test_split(
            train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=42
        )
        
        # Store with labels
        label = 0 if category == 'real' else 1
        splits['train'].extend([(v, label) for v in train])
        splits['val'].extend([(v, label) for v in val])
        splits['test'].extend([(v, label) for v in test])
        
        print(f"\n{category.upper()}:")
        print(f"  Train: {len(train)} ({len(train)/len(videos)*100:.1f}%)")
        print(f"  Val:   {len(val)} ({len(val)/len(videos)*100:.1f}%)")
        print(f"  Test:  {len(test)} ({len(test)/len(videos)*100:.1f}%)")
    
    # Save metadata
    metadata = {
        'train': [(str(p), int(l)) for p, l in splits['train']],
        'val': [(str(p), int(l)) for p, l in splits['val']],
        'test': [(str(p), int(l)) for p, l in splits['test']],
        'split_ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio}
    }
    
    metadata_file = os.path.join(output_dir, 'splits.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Splits saved to {metadata_file}")
    
    print("\nTOTAL SPLIT SUMMARY:")
    print(f"  Train: {len(splits['train'])} videos")
    print(f"  Val:   {len(splits['val'])} videos")
    print(f"  Test:  {len(splits['test'])} videos")
    
    return metadata_file

def main():
    data_dir = 'data'
    
    # Validate videos
    valid_videos, corrupted = validate_videos(data_dir)
    
    # Check balance
    check_balance(valid_videos)
    
    # Remove corrupted files (automatic)
    total_corrupted = len(corrupted['real']) + len(corrupted['fake'])
    if total_corrupted > 0:
        print(f"\n  {total_corrupted} corrupted videos found - Removing automatically...")
        for category in ['real', 'fake']:
            for video_path in corrupted[category]:
                try:
                    os.remove(video_path)
                    print(f"  Deleted: {Path(video_path).name}")
                except Exception as e:
                    print(f"  Failed to delete {Path(video_path).name}: {e}")
    
    # Create splits
    splits_file = create_splits(valid_videos, data_dir)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nNext step: Train the model using splits.json")
    print("Run: python scripts/train.py")

if __name__ == '__main__':
    main()
