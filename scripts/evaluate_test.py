#!/usr/bin/env python3
"""
Evaluate the trained model on the test set from splits.json
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import VideoDataset
from src.model import VideoDetector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np
import warnings
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Suppress OpenCV warnings and logs
warnings.filterwarnings('ignore')
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

def evaluate_test_set(data_dir='data', batch_size=24, device=None):
    """
    Evaluate the model on the test set
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load test dataset
    print("Loading test dataset...")
    try:
        test_dataset = VideoDataset(data_dir, split='test')
        print(f"✓ Test set: {len(test_dataset)} videos")
    except ValueError as e:
        print(f"Error: {e}")
        print("Make sure splits.json contains a 'test' split")
        return

    # Detect pre-extracted frames availability
    frames_dir = os.path.join(data_dir, 'frames')
    has_preextracted = os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 0

    if has_preextracted:
        print("✓ Using pre-extracted frames (fast path)")
        num_workers = 14
    else:
        print("⚠ Pre-extracted frames not found. Using on-the-fly extraction (slower)")
        num_workers = 6

    print(f"✓ Using {num_workers} workers for testing\n")

    pin = device.type == 'cuda'

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin
    )

    # Load the trained model
    model = VideoDetector().to(device)
    model_path = 'models/best_model.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Loaded model from {model_path}")
    else:
        print(f"⚠ Model not found at {model_path}, using final model")
        model_path = 'models/model.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ Loaded model from {model_path}")
        else:
            print("❌ No trained model found!")
            return

    model.eval()

    # Define thresholds for both modes
    THRESHOLDS = {
        'f1': 0.420,      # F1-optimal threshold
        'recall': 0.290   # Recall-constrained threshold
    }

    # Evaluate on test set for both modes
    print("Evaluating on test set with both threshold modes...")

    results = {}

    for mode, threshold in THRESHOLDS.items():
        print(f"\n--- Evaluating {mode.upper()} Mode (threshold={threshold}) ---")

        all_targets = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                fake_probs = probs[:, 1].cpu().numpy()  # Probability of fake class

                # Apply threshold-based prediction
                preds = (fake_probs >= threshold).astype(int)

                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds)
                all_probs.extend(fake_probs)

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, average='weighted')
        recall = recall_score(all_targets, all_preds, average='weighted')
        f1 = f1_score(all_targets, all_preds, average='weighted')

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        print(f"Confusion Matrix:\n{cm}")

        # Classification report
        report = classification_report(
            all_targets,
            all_preds,
            target_names=['Real', 'Fake'],
            output_dict=True
        )
        df_report = pd.DataFrame(report).transpose()
        print("Detailed Classification Report:")
        print(df_report)

        # Store results
        results[mode] = {
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'targets': all_targets,
            'predictions': all_preds,
            'probabilities': all_probs
        }

    # Save comprehensive results
    with open('results/test_evaluation_dual_mode.json', 'w') as f:
        # Convert numpy arrays to lists and numpy types to Python types for JSON serialization
        json_results = {}
        for mode, data in results.items():
            json_results[mode] = {}
            for k, v in data.items():
                if hasattr(v, 'tolist'):  # numpy arrays
                    json_results[mode][k] = v.tolist()
                elif isinstance(v, (np.int64, np.int32, np.float64, np.float32)):  # numpy scalars
                    json_results[mode][k] = v.item()
                else:  # other types
                    json_results[mode][k] = v
        json.dump(json_results, f, indent=2)

    # Save individual classification reports
    for mode, data in results.items():
        df_report = pd.DataFrame(data['classification_report']).transpose()
        df_report.to_csv(f'results/test_classification_report_{mode}.csv')

    # Create plots
    plt.figure(figsize=(15, 5))

    # Confusion matrix plot
    plt.subplot(1, 3, 1)
    plt.imshow(cm, cmap='Blues')
    plt.title('Test Set Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red')

    # ROC curve
    plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test Set ROC Curve')
    plt.legend(loc='lower right')

    # Precision-Recall curve
    plt.subplot(1, 3, 3)
    from sklearn.metrics import precision_recall_curve
    precision_curve, recall_curve, _ = precision_recall_curve(all_targets, all_probs)
    plt.plot(recall_curve, precision_curve, label=f'PR curve (AUC = {auc(recall_curve, precision_curve):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Test Set Precision-Recall Curve')
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig('results/test_evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n✓ Test evaluation complete!")
    print("✓ Results saved to results/test_evaluation.json")
    print("✓ Classification report saved to results/test_classification_report.csv")
    print("✓ Plots saved to results/test_evaluation_plots.png")

    return results

if __name__ == "__main__":
    evaluate_test_set(batch_size=4)
