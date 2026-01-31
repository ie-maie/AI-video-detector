"""+++++++uuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuuugn$fb·+
Ga>nQuick training on a small subset for rapid iteration and validation.
Trains in ~5-10 minutes instead of hours.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from src.dataset import VideoDataset
from src.model import VideoDetector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

def quick_train(data_dir='data', batch_size=4, epochs=3, learning_rate=0.001, device=None):
    """Train on small subset for quick validation (5-10 min)."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load datasets
    print("Loading datasets (small subset)...")
    train_dataset = VideoDataset(data_dir, split='train')
    val_dataset = VideoDataset(data_dir, split='val')
    
    # Use only 10% of data for quick validation
    train_size = max(1, len(train_dataset) // 10)
    val_size = max(1, len(val_dataset) // 10)
    
    train_subset = Subset(train_dataset, list(range(train_size)))
    val_subset = Subset(val_dataset, list(range(val_size)))
    
    print(f"✓ Train set: {len(train_subset)} videos (10% of {len(train_dataset)})")
    print(f"✓ Val set: {len(val_subset)} videos (10% of {len(val_dataset)})\n")
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize model
    model = VideoDetector().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("=" * 60)
    print("QUICK TRAINING (Small Subset - 3 Epochs)")
    print("=" * 60)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(targets.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(targets.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted', zero_division=0)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'models/quick_model.pth')
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f} ✓ (saved)")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
    
    # Final metrics
    print("\n" + "=" * 60)
    print("QUICK TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Precision: {precision_score(val_targets, val_preds, average='weighted', zero_division=0):.4f}")
    print(f"Recall: {recall_score(val_targets, val_preds, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(val_targets, val_preds, average='weighted', zero_division=0):.4f}")
    print(f"\n✓ Quick model saved to models/quick_model.pth")
    print("➜ If satisfied, run: python scripts/train.py (full training)")

if __name__ == '__main__':
    quick_train(batch_size=4, epochs=3)
