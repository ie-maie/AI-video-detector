import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import glob
import json
from torchvision import transforms
import warnings
from pathlib import Path

# Suppress OpenCV warnings
warnings.filterwarnings('ignore')
cv2.setLogLevel(0)

class VideoDataset(Dataset):
    def __init__(self, data_dir='data', split='train', transform=None, seq_len=30, frame_size=(224, 224), use_preextracted=True):
        self.data_dir = data_dir
        self.split = split
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.use_preextracted = use_preextracted
        self.frames_dir = os.path.join(data_dir, 'frames')

        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            if split == 'train':
                # Data augmentation for training
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.RandomResizedCrop(size=self.frame_size, scale=(0.8, 1.0)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                # Only normalization for validation/test
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(self.frame_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        
        # Check if pre-extracted frames are available
        self.has_preextracted = os.path.exists(self.frames_dir) and len(os.listdir(self.frames_dir)) > 0
        
        # Load video paths and labels
        self.video_paths = []
        self.labels = []
        
        # Check if splits.json exists (from preprocessing)
        splits_file = os.path.join(data_dir, 'splits.json')
        if os.path.exists(splits_file):
            # Load from splits
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            
            if split not in splits:
                raise ValueError(f"Split '{split}' not found. Available: {list(splits.keys())}")
            
            for video_path, label in splits[split]:
                self.video_paths.append(video_path)
                self.labels.append(label)
        else:
            # Fallback: load all videos from folders
            real_dir = os.path.join(data_dir, 'real')
            fake_dir = os.path.join(data_dir, 'fake')
            
            for path in glob.glob(os.path.join(real_dir, '*.mp4')):
                self.video_paths.append(path)
                self.labels.append(0)  # 0: real
            
            for path in glob.glob(os.path.join(fake_dir, '*.mp4')):
                self.video_paths.append(path)
                self.labels.append(1)  # 1: fake
        
        if not self.video_paths:
            raise ValueError(f"No videos found for split '{split}'. Run: python scripts/preprocess_data.py")

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        video_id = Path(video_path).stem
        
        # Try to load pre-extracted frames first
        if self.use_preextracted and self.has_preextracted:
            frames_file = os.path.join(self.frames_dir, f"{video_id}.npy")
            if os.path.exists(frames_file):
                try:
                    frames_array = np.load(frames_file)  # Shape: (30, 224, 224, 3)
                    frames = [frames_array[i] for i in range(frames_array.shape[0])]

                    # Apply transforms to each frame
                    frames = [self.transform(frame) for frame in frames]
                    return torch.stack(frames), label
                except Exception as e:
                    # Fallback to on-the-fly extraction if pre-extracted fails
                    print(f"Warning: Failed to load pre-extracted {video_id}: {e}. Extracting on-the-fly...")
        
        # On-the-fly extraction (fallback or if pre-extraction disabled)
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Check if video metadata is valid
            if total_frames <= 0 or fps <= 0:
                cap.release()
                raise ValueError(f"Invalid video metadata: {total_frames} frames, {fps} FPS")
            
            # Sample seq_len frames evenly
            if total_frames >= self.seq_len:
                step = max(1, total_frames // self.seq_len)
                for i in range(0, total_frames, step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame = cv2.resize(frame, self.frame_size)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                    if len(frames) >= self.seq_len:
                        break
            else:
                # If fewer frames, read all and duplicate last
                while len(frames) < total_frames:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame = cv2.resize(frame, self.frame_size)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                    else:
                        break
                while len(frames) < self.seq_len:
                    if frames:
                        frames.append(frames[-1])
                    else:
                        raise ValueError("Could not extract any frames from video")
            
            cap.release()
            
            # Ensure we have exactly seq_len frames
            if len(frames) < self.seq_len:
                if frames:
                    while len(frames) < self.seq_len:
                        frames.append(frames[-1])
                else:
                    raise ValueError("No valid frames extracted")
            
            # Apply transforms to each frame
            if self.transform:
                frames = [self.transform(frame) for frame in frames]
            else:
                # Fallback: convert to tensor and normalize
                frames = [torch.tensor(frame).permute(2, 0, 1).float() / 255 for frame in frames]
            
            return torch.stack(frames), label
        
        except Exception as e:
            print(f"Warning: Failed to load {video_path}: {str(e)}")
            # Return a black video as fallback
            black_frames = [torch.zeros((3, self.frame_size[0], self.frame_size[1])) for _ in range(self.seq_len)]
            return torch.stack(black_frames), label