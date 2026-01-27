import torch
import cv2
import torchvision.transforms as T
from src.model import VideoDetector

# ---------------- CONFIG ----------------
MODEL_PATH = "models/best_model.pth"
NUM_FRAMES = 30
IMG_SIZE = 224
# ----------------------------------------

def extract_frames(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames extracted from video")

    # Uniform sampling
    if len(frames) >= num_frames:
        idxs = torch.linspace(0, len(frames) - 1, num_frames).long()
        frames = [frames[i] for i in idxs]
    else:
        frames += [frames[-1]] * (num_frames - len(frames))

    return frames


def infer(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = VideoDetector().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
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
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()

    label = "AI-generated" if pred == 1 else "Real"
    return label, confidence


if __name__ == "__main__":
    video = "path/to/test.mp4"
    label, conf = infer(video)
    print(f"Prediction: {label} ({conf:.2%})")
