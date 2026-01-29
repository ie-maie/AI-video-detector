import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from src.dataset import VideoDataset
from src.model import VideoDetector
import matplotlib.pyplot as plt

def find_optimal_threshold_f1(model_path="models/best_model.pth", data_dir="data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = VideoDetector().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load validation data
    val_dataset = VideoDataset(data_dir, split='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    all_targets = []
    all_probs = []

    print("Extracting predictions from validation set...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # Probability of being fake

            all_targets.extend(targets.numpy())
            all_probs.extend(probs)

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Test different thresholds
    thresholds = np.linspace(0.01, 0.99, 99)
    f1_scores = []
    precisions = []
    recalls = []

    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        f1 = f1_score(all_targets, preds)
        precision = precision_score(all_targets, preds, zero_division=0)
        recall = recall_score(all_targets, preds, zero_division=0)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    # Find threshold with maximum F1 score
    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    optimal_precision = precisions[best_idx]
    optimal_recall = recalls[best_idx]

    print(f"\nF1-Optimized Threshold Results")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Precision: {optimal_precision:.4f}")
    print(f"Recall: {optimal_recall:.4f}")

    # Plot results
    plt.figure(figsize=(12, 5))

    # Plot 1: F1 Score vs Threshold
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1 Score')
    plt.axvline(x=optimal_threshold, color='red', linestyle='--',
                label=f'Optimal Threshold ({optimal_threshold:.3f})')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Precision vs Recall
    plt.subplot(1, 2, 2)
    plt.plot(recalls, precisions, 'b-', linewidth=2, alpha=0.7)
    plt.scatter([optimal_recall], [optimal_precision], color='red', s=100, zorder=5,
                label=f'F1 Optimal\nThreshold: {optimal_threshold:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall (F1 Optimal Point)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/threshold_optimization_f1.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save optimal threshold
    with open('models/optimal_threshold_f1.txt', 'w') as f:
        f.write(f"Optimal Threshold: {optimal_threshold:.6f}\n")
        f.write(f"F1 Score: {best_f1:.4f}\n")
        f.write(f"Precision: {optimal_precision:.4f}\n")
        f.write(f"Recall: {optimal_recall:.4f}\n")

    print(f"\nOptimal threshold saved to models/optimal_threshold_f1.txt")
    print(f"Plot saved to results/threshold_optimization_f1.png")

    return optimal_threshold, best_f1, optimal_precision, optimal_recall

if __name__ == "__main__":
    find_optimal_threshold_f1()
