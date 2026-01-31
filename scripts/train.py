import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import VideoDataset
from src.model import VideoDetector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import warnings
import os

# Suppress OpenCV warnings and logs
warnings.filterwarnings('ignore')
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
 
def train(data_dir='data', batch_size=24, epochs=20, learning_rate=0.001, device=None):
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    use_amp = device.type == 'cuda'

    # Load datasets
    print("Loading datasets...")
    try:
        train_dataset = VideoDataset(data_dir, split='train')
        val_dataset = VideoDataset(data_dir, split='val')
        print(f"✓ Train set: {len(train_dataset)} videos")
        print(f"✓ Val set: {len(val_dataset)} videos\n")
    except ValueError as e:
        print(f"Error: {e}")
        print("Run preprocessing first: python scripts/preprocess_data.py")
        return
    
    # Detect pre-extracted frames availability
    frames_dir = os.path.join(data_dir, 'frames')
    has_preextracted = os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 0
    
    if has_preextracted:
        print("✓ Using pre-extracted frames (fast path)")
        num_workers_train = 14
        num_workers_val = 6
    else:
        print("⚠ Pre-extracted frames not found. Using on-the-fly extraction (slower)")
        num_workers_train = 6
        num_workers_val = 2

    print(f"✓ Using {num_workers_train} workers for training\n")

    pin = device.type == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers_train,
        pin_memory=pin,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_val,
        pin_memory=pin
    )
    
    # Initialize model
    model = VideoDetector().to(device)

    # Calculate class weights for balanced training (address class imbalance)
    train_labels = [label for _, label in train_dataset]
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    class_weights = torch.tensor([total_samples / (len(class_counts) * count) for count in class_counts], dtype=torch.float).to(device)
    print(f"Class weights: {class_weights}")

    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    # Freeze early ResNet layers (critical fix)
    for name, param in model.cnn.named_parameters():
        if "layer4" not in name:
            param.requires_grad = False


    optimizer = optim.Adam([
    {'params': model.cnn.layer4.parameters(), 'lr': learning_rate * 0.1},
    {'params': model.lstm.parameters(), 'lr': learning_rate},
    {'params': model.fc.parameters(), 'lr': learning_rate},
    ], weight_decay=5e-4)  # Increased L2 regularization to reduce overfitting

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.01)

    
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, classification_report
    import pandas as pd
    os.makedirs('results', exist_ok=True)

    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    val_f1s = []
    epochs_without_improvement = 0

    all_val_targets = []
    all_val_probs = []
    all_val_preds = []  # FIX 3

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(epochs):
        # ================= TRAIN =================
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(targets.detach().cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # ================= VALIDATION =================
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        val_probs = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # FIX 2: typo corrected
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()

                val_preds.extend(preds)
                val_targets.extend(targets.detach().cpu().numpy())
                val_probs.extend(probs)

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')

        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        # FIX 3: store full validation results
        all_val_targets = val_targets
        all_val_probs = val_probs
        all_val_preds = val_preds

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            saved = "✓ (saved)"
            epochs_without_improvement = 0  # Reset counter on improvement
        else:
            saved = ""
            epochs_without_improvement += 1

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"- Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f} {saved}"
        )

        # Step the scheduler at the end of each epoch
        scheduler.step()

        # Early stopping check
        patience = 8
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
            break

    # ================= PLOTS =================
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.savefig('results/loss_curves.png')
    plt.close()

    plt.figure()
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.savefig('results/accuracy_curves.png')
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(all_val_targets, np.array(all_val_preds))
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix (Val)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.savefig('results/confusion_matrix.png')
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(all_val_targets, all_val_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Val)')
    plt.legend(loc='lower right')
    plt.savefig('results/roc_curve.png')
    plt.close()

    # Classification report
    report = classification_report(
        all_val_targets,
        np.array(all_val_preds),
        target_names=['Real', 'Fake'],
        output_dict=True
    )
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv('results/classification_report.csv')

    print("\nFinal Classification Report:")
    print(df_report)

    print("\n" + "=" * 60)
    print("FINAL VALIDATION METRICS")
    print("=" * 60)
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Precision: {precision_score(all_val_targets, all_val_preds, average='weighted'):.4f}")
    print(f"Recall: {recall_score(all_val_targets, all_val_preds, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(all_val_targets, all_val_preds, average='weighted'):.4f}")
    print(f"\nConfusion Matrix:\n{cm}")

    torch.save(model.state_dict(), 'models/model.pth')
    print("\n✓ Best model saved to models/best_model.pth")
    print("✓ Final model saved to models/model.pth")


if __name__ == '__main__':
    train(batch_size=4, epochs=20)
