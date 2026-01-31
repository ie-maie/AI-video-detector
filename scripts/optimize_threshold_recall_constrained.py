import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score
from src.dataset import VideoDataset
from src.model import VideoDetector
import matplotlib.pyplot as plt

def find_optimal_threshold_recall_constrained(model_path="models/best_model.pth",
                                             data_dir="data",
                                             min_recall=0.90):
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
    precisions = []
    recalls = []

    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        precision = precision_score(all_targets, preds, zero_division=0)
        recall = recall_score(all_targets, preds, zero_division=0)
        precisions.append(precision)
        recalls.append(recall)

    # Find thresholds that meet minimum recall constraint
    valid_thresholds = []
    valid_precisions = []

    for i, thresh in enumerate(thresholds):
        if recalls[i] >= min_recall:
            valid_thresholds.append(thresh)
            valid_precisions.append(precisions[i])

    if not valid_thresholds:
        print(f"No threshold found that maintains recall >= {min_recall}")
        print(f"Maximum achievable recall: {max(recalls):.4f}")
        return None, None, None, None

    # Find threshold with maximum precision under recall constraint
    best_idx = np.argmax(valid_precisions)
    optimal_threshold = valid_thresholds[best_idx]
    optimal_precision = valid_precisions[best_idx]
    optimal_recall = recalls[np.where(thresholds == optimal_threshold)[0][0]]

    print(f"\nRecall-Constrained Threshold Optimization (Min Recall: {min_recall})")
    print(f"Optimal Threshold: {optimal_threshold:.3f}")
    print(f"Precision: {optimal_precision:.4f}")
    print(f"Recall: {optimal_recall:.4f}")
    print(f"F1 Score: {2 * optimal_precision * optimal_recall / (optimal_precision + optimal_recall):.4f}")

    # Plot results
    plt.figure(figsize=(12, 5))

    # Plot 1: Precision and Recall vs Threshold
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, 'g-', label='Recall', linewidth=2)
    plt.axvline(x=optimal_threshold, color='red', linestyle='--',
                label=f'Optimal Threshold ({optimal_threshold:.3f})')
    plt.axhline(y=min_recall, color='orange', linestyle=':', alpha=0.7,
                label=f'Min Recall ({min_recall})')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision and Recall vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Precision vs Recall (with constraint)
    plt.subplot(1, 2, 2)
    plt.plot(recalls, precisions, 'b-', linewidth=2, alpha=0.7)
    plt.scatter([optimal_recall], [optimal_precision], color='red', s=100, zorder=5,
                label=f'Optimal Point\nThreshold: {optimal_threshold:.3f}')
    plt.axvline(x=min_recall, color='orange', linestyle=':', alpha=0.7,
                label=f'Min Recall Constraint ({min_recall})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall (with Constraint)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/threshold_optimization_recall_constrained.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save optimal threshold
    with open('models/optimal_threshold_recall_constrained.txt', 'w') as f:
        f.write(f"Min Recall Constraint: {min_recall}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.6f}\n")
        f.write(f"Precision: {optimal_precision:.4f}\n")
        f.write(f"Recall: {optimal_recall:.4f}\n")
        f.write(f"F1 Score: {2 * optimal_precision * optimal_recall / (optimal_precision + optimal_recall):.4f}\n")

    print(f"\nOptimal threshold saved to models/optimal_threshold_recall_constrained.txt")
    print(f"Plot saved to results/threshold_optimization_recall_constrained.png")

    return optimal_threshold, optimal_precision, optimal_recall, min_recall

if __name__ == "__main__":
    # You can adjust the minimum recall constraint here
    min_recall_constraint = 0.90  # 90% minimum recall
    find_optimal_threshold_recall_constrained(min_recall=min_recall_constraint)
