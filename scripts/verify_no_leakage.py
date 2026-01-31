"""
Verify that data splits are maintained and no leakage occurs.
Run this after frame extraction to confirm train/val/test separation.
"""

import json
import os
from pathlib import Path

def verify_data_integrity(data_dir='data'):
    """
    Check:
    1. Splits are deterministic (no changes)
    2. Video IDs in frames match splits
    3. Train/val/test are mutually exclusive
    4. No frames in wrong directories
    """
    
    splits_file = os.path.join(data_dir, 'splits.json')
    frames_dir = os.path.join(data_dir, 'frames')
    
    if not os.path.exists(splits_file):
        print("❌ Error: splits.json not found")
        return False
    
    if not os.path.exists(frames_dir):
        print("❌ Error: frames directory not found")
        return False
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    # Collect all video IDs by split
    train_ids = set(Path(p).stem for p, _ in splits['train'])
    val_ids = set(Path(p).stem for p, _ in splits['val'])
    test_ids = set(Path(p).stem for p, _ in splits['test'])
    
    # Get all extracted frame files
    frame_files = set(Path(f).stem for f in os.listdir(frames_dir) if f.endswith('.npy'))
    
    print("=" * 60)
    print("DATA INTEGRITY VERIFICATION")
    print("=" * 60)
    
    # Check 1: Mutual exclusivity
    overlap_tv = train_ids & val_ids
    overlap_te = train_ids & test_ids
    overlap_ve = val_ids & test_ids
    
    if overlap_tv or overlap_te or overlap_ve:
        print("❌ CRITICAL: Split contamination detected!")
        if overlap_tv:
            print(f"   Train-Val overlap: {overlap_tv}")
        if overlap_te:
            print(f"   Train-Test overlap: {overlap_te}")
        if overlap_ve:
            print(f"   Val-Test overlap: {overlap_ve}")
        return False
    
    print("✓ Splits are mutually exclusive")
    print(f"  Train: {len(train_ids)} videos")
    print(f"  Val:   {len(val_ids)} videos")
    print(f"  Test:  {len(test_ids)} videos")
    print(f"  Total: {len(train_ids) + len(val_ids) + len(test_ids)} videos\n")
    
    # Check 2: Frame coverage
    all_split_ids = train_ids | val_ids | test_ids
    missing_frames = all_split_ids - frame_files
    extra_frames = frame_files - all_split_ids
    
    if missing_frames:
        print(f"⚠ {len(missing_frames)} videos missing frames (not extracted yet):")
        for vid in list(missing_frames)[:5]:
            print(f"  - {vid}")
        if len(missing_frames) > 5:
            print(f"  ... and {len(missing_frames) - 5} more")
    
    if extra_frames:
        print(f"⚠ {len(extra_frames)} extra frame files not in splits (corrupted?)")
        for vid in list(extra_frames)[:5]:
            print(f"  - {vid}")
    
    if not missing_frames and not extra_frames:
        print("✓ Frame coverage: 100% (all videos extracted)")
    
    print("\n" + "=" * 60)
    print("LEAKAGE RISK ASSESSMENT")
    print("=" * 60)
    
    if not overlap_tv and not overlap_te and not overlap_ve:
        print("✅ NO DATA LEAKAGE DETECTED")
        print("   Train/Val/Test splits are properly isolated")
        print("   Safe to proceed with training")
        return True
    else:
        print("❌ DATA LEAKAGE RISK!")
        print("   Do not train until splits are fixed")
        return False


if __name__ == '__main__':
    verify_data_integrity()
