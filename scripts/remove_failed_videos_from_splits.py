"""
Removes videos from splits.json that failed frame extraction.
Run this after frame extraction to clean up splits.json.
"""

import json
import os
from pathlib import Path

FAILED_VIDEO_IDS = [
    "4841885",
    "4440928",
    "213026_medium",
    "2324274",
    "3683300"
]

def clean_splits(splits_path='data/splits.json'):
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    def is_valid(video_path):
        vid = Path(video_path).stem
        return vid not in FAILED_VIDEO_IDS
    
    cleaned = {}
    for split in ['train', 'val', 'test']:
        cleaned[split] = [entry for entry in splits[split] if is_valid(entry[0])]
        print(f"{split}: {len(splits[split])} → {len(cleaned[split])} videos after cleanup")
    
    with open(splits_path, 'w') as f:
        json.dump(cleaned, f, indent=2)
    print("✓ splits.json updated. Failed videos removed.")

if __name__ == '__main__':
    clean_splits()
