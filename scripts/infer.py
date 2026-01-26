import torch
from src.model import VideoDetector
import cv2

def infer(video_path):
    model = VideoDetector()
    model.load_state_dict(torch.load('models/model.pth'))
    model.eval()
    
    # Load and process video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    frames = frames[:30]  # Assume 30 frames
    # Preprocess frames
    frames = [torch.tensor(frame).permute(2,0,1).float() / 255 for frame in frames]
    frames = torch.stack(frames).unsqueeze(0)  # Add batch dim
    
    with torch.no_grad():
        output = model(frames)
        pred = torch.argmax(output, dim=1).item()
    
    return 'AI-generated' if pred == 1 else 'Real'

if __name__ == '__main__':
    result = infer('path/to/test.mp4')
    print(result)