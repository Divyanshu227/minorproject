"""
Parkinson's Disease Detection Pipeline
Using DaTscan SPECT Images with Subject-Level Attention Multiple-Instance Learning (MIL)
Dataset: NTUA Parkinson Dataset (Tagaris et al., 2018)
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# ---------------------------------------------------------
# 1. Reproducibility Seed
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ---------------------------------------------------------
# 2. Data Preparation & Curation
# ---------------------------------------------------------
def is_valid_slice(img_path, min_mean_threshold=3.0):
    """Filter out pitch-black background / empty padding slices."""
    try:
        im = Image.open(img_path).convert('L')
        arr = np.array(im)
        return arr.mean() >= min_mean_threshold
    except Exception:
        return False

def load_subject_dataset(base_dir):
    """
    Scans the NTUA dataset, filters subjects and slices, and returns a list of subject dictionaries.
    """
    pd_dir = os.path.join(base_dir, 'PD Patients')
    npd_dir = os.path.join(base_dir, 'Non PD Patients')
    
    # Excluded ambiguous subjects from clinical CSV
    excluded_subjects = {'Subject20', 'Subject22', 'Subject24', 'Subject58'}
    
    subjects_data = []
    
    # Process PD Patients (Class 1)
    for s in os.listdir(pd_dir):
        if not s.startswith('Subject') or s in excluded_subjects:
            continue
        dat_dir = os.path.join(pd_dir, s, '0.DAT')
        if not os.path.exists(dat_dir):
            continue
        
        valid_images = []
        for root, _, files in os.walk(dat_dir):
            for f in sorted(files):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    fp = os.path.join(root, f)
                    if is_valid_slice(fp):
                        valid_images.append(fp)
                        
        if len(valid_images) > 0:
            subjects_data.append({
                'subject_id': s,
                'label': 1,  # Parkinson's Disease
                'label_name': 'PD',
                'image_paths': valid_images
            })
            
    # Process Non-PD Control Patients (Class 0)
    for s in os.listdir(npd_dir):
        if not s.startswith('Subject') or s in excluded_subjects:
            continue
        dat_dir = os.path.join(npd_dir, s, '0.DAT')
        if not os.path.exists(dat_dir):
            continue
            
        valid_images = []
        for root, _, files in os.walk(dat_dir):
            for f in sorted(files):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    fp = os.path.join(root, f)
                    if is_valid_slice(fp):
                        valid_images.append(fp)
                        
        if len(valid_images) > 0:
            subjects_data.append({
                'subject_id': s,
                'label': 0,  # Non-PD Control / Essential Tremor / Healthy
                'label_name': 'Non-PD',
                'image_paths': valid_images
            })
            
    return subjects_data

# ---------------------------------------------------------
# 3. Subject-Level Dataset & Transforms
# ---------------------------------------------------------
class SubjectDaTDataset(Dataset):
    def __init__(self, subject_list, transform=None, max_slices_per_bag=32):
        self.subject_list = subject_list
        self.transform = transform
        self.max_slices_per_bag = max_slices_per_bag

    def __len__(self):
        return len(self.subject_list)

    def __getitem__(self, idx):
        item = self.subject_list[idx]
        image_paths = item['image_paths']
        label = torch.tensor(item['label'], dtype=torch.float32)
        subject_id = item['subject_id']
        
        # If subject has too many slices (e.g. 128-slice raw volume), sample top informative slices
        if len(image_paths) > self.max_slices_per_bag:
            step = len(image_paths) // self.max_slices_per_bag
            selected_paths = image_paths[::step][:self.max_slices_per_bag]
        else:
            selected_paths = image_paths

        tensors = []
        for p in selected_paths:
            im = Image.open(p).convert('RGB')
            if self.transform:
                im_t = self.transform(im)
            else:
                im_t = transforms.ToTensor()(im)
            tensors.append(im_t)
            
        # Stack into bag tensor: (num_slices, 3, 224, 224)
        bag_tensor = torch.stack(tensors)
        return bag_tensor, label, subject_id

def collate_mil_bag(batch):
    """Custom collator since each subject may have variable number of slices."""
    bag_tensor, label, subject_id = batch[0]
    return bag_tensor, label, subject_id

# ---------------------------------------------------------
# 4. Attention-Based Multiple Instance Learning (MIL) Model
# ---------------------------------------------------------
class AttentionMILNet(nn.Module):
    def __init__(self, feature_dim=512, attention_dim=128, dropout_rate=0.3):
        super(AttentionMILNet, self).__init__()
        
        # Pretrained ResNet-18 Backbone
        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Gated Attention Mechanism (Ilse et al., 2018)
        self.attention_V = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(attention_dim, 1)
        
        # Subject Classifier Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, bag):
        # bag: (N, 3, 224, 224)
        features = self.feature_extractor(bag) # (N, 512, 1, 1)
        features = features.view(features.size(0), -1) # (N, 512)
        
        # Calculate Attention Weights
        A_V = self.attention_V(features) # (N, attention_dim)
        A_U = self.attention_U(features) # (N, attention_dim)
        A = self.attention_weights(A_V * A_U) # (N, 1)
        A = torch.transpose(A, 1, 0) # (1, N)
        A = F.softmax(A, dim=1) # Softmax over slices in bag
        
        # Aggregate subject-level embedding: (1, 512)
        subject_embedding = torch.mm(A, features)
        
        # Classification logit: (1, 1)
        logits = self.classifier(subject_embedding)
        return logits.squeeze(0), A.squeeze(0)

# ---------------------------------------------------------
# 5. Training and Evaluation Loops
# ---------------------------------------------------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for bag_tensor, label, _ in dataloader:
        bag_tensor = bag_tensor.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(bag_tensor)
        loss = criterion(logits.view(-1), label.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= 0.5 else 0
        if pred == int(label.item()):
            correct += 1
        total += 1
        
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    subject_attentions = {}
    
    with torch.no_grad():
        for bag_tensor, label, sub_id in dataloader:
            bag_tensor = bag_tensor.to(device)
            label = label.to(device)
            
            logits, attention = model(bag_tensor)
            loss = criterion(logits.view(-1), label.view(-1))
            total_loss += loss.item()
            
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob >= 0.5 else 0
            
            all_probs.append(prob)
            all_preds.append(pred)
            all_labels.append(int(label.item()))
            subject_attentions[sub_id] = attention.cpu().numpy()
            
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_labels, all_preds, all_probs, subject_attentions

# ---------------------------------------------------------
# 6. Main Execution Pipeline
# ---------------------------------------------------------
def main():
    print("=" * 60)
    print(" PARKINSON'S DISEASE DETECTION: DaTscan SPECT MIL PIPELINE ")
    print("=" * 60)
    
    base_dir = r"c:\Users\divya\Desktop\minorproject\parkinsons-disease-prediction\ntua-parkinson-dataset-master\ntua-parkinson-dataset-master"
    output_dir = r"c:\Users\divya\Desktop\minorproject\parkinsons-disease-prediction"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on Device: {device}")
    
    # 1. Load Data
    subjects = load_subject_dataset(base_dir)
    print(f"Loaded {len(subjects)} curated subjects with valid DaTscan SPECT imaging.")
    
    pd_count = sum(1 for s in subjects if s['label'] == 1)
    npd_count = sum(1 for s in subjects if s['label'] == 0)
    print(f"Cohort Breakdown -> Parkinson's Disease (PD): {pd_count} | Non-PD Controls: {npd_count}")
    
    # 2. Stratified Subject-Level Splitting
    labels = [s['label'] for s in subjects]
    train_subs, temp_subs = train_test_split(subjects, test_size=0.30, stratify=labels, random_state=42)
    temp_labels = [s['label'] for s in temp_subs]
    val_subs, test_subs = train_test_split(temp_subs, test_size=0.50, stratify=temp_labels, random_state=42)
    
    print("\n--- Stratified Split Summary (Strict Subject-Level) ---")
    print(f"Train Cohort: {len(train_subs)} subjects (PD: {sum(s['label'] for s in train_subs)}, Non-PD: {len(train_subs)-sum(s['label'] for s in train_subs)})")
    print(f"Validation Cohort: {len(val_subs)} subjects (PD: {sum(s['label'] for s in val_subs)}, Non-PD: {len(val_subs)-sum(s['label'] for s in val_subs)})")
    print(f"Test Cohort: {len(test_subs)} subjects (PD: {sum(s['label'] for s in test_subs)}, Non-PD: {len(test_subs)-sum(s['label'] for s in test_subs)})")
    
    # 3. Data Transformations
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = SubjectDaTDataset(train_subs, transform=train_transforms)
    val_dataset = SubjectDaTDataset(val_subs, transform=eval_transforms)
    test_dataset = SubjectDaTDataset(test_subs, transform=eval_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_mil_bag)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_mil_bag)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_mil_bag)
    
    # 4. Model Setup
    model = AttentionMILNet(feature_dim=512, attention_dim=128, dropout_rate=0.3).to(device)
    
    # Class weighting for BCE
    pos_weight = torch.tensor([(len(train_subs) - sum(s['label'] for s in train_subs)) / sum(s['label'] for s in train_subs)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Fine-tuning: Optimizer with differential learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.feature_extractor.parameters(), 'lr': 2e-5},
        {'params': model.attention_V.parameters(), 'lr': 1e-4},
        {'params': model.attention_U.parameters(), 'lr': 1e-4},
        {'params': model.attention_weights.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
    
    # 5. Training Loop
    epochs = 25
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    model_save_path = os.path.join(output_dir, 'best_parkinsons_dat_model.pth')
    
    print("\n--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_labels, val_preds, val_probs, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.1f}%, F1: {val_f1:.4f}")
        
        # Save best checkpoint
        if val_loss < best_val_loss or (val_loss == best_val_loss and val_f1 > best_val_f1):
            best_val_loss = val_loss
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1
            }, model_save_path)
            
    print(f"\nTraining Complete! Best model checkpoint saved to: {model_save_path}")
    
    # 6. Test Set Evaluation
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best checkpoint from Epoch {checkpoint['epoch']} for Final Testing.")
    
    test_loss, test_acc, test_labels, test_preds, test_probs, _ = eval_epoch(model, test_loader, criterion, device)
    
    precision = precision_score(test_labels, test_preds, average='binary', zero_division=0)
    recall = recall_score(test_labels, test_preds, average='binary', zero_division=0)
    f1_bin = f1_score(test_labels, test_preds, average='binary', zero_division=0)
    f1_macro = f1_score(test_labels, test_preds, average='macro', zero_division=0)
    try:
        roc_auc = roc_auc_score(test_labels, test_probs)
    except Exception:
        roc_auc = 0.5
        
    print("\n" + "=" * 60)
    print(" FINAL TEST EVALUATION RESULTS (SUBJECT-LEVEL) ")
    print("=" * 60)
    print(f"Test Accuracy:         {test_acc * 100:.2f}%")
    print(f"Test Precision:        {precision:.4f}")
    print(f"Test Recall:           {recall:.4f}")
    print(f"Test F1-Score (Binary):{f1_bin:.4f}")
    print(f"Test F1-Score (Macro): {f1_macro:.4f}")
    print(f"Test ROC-AUC Score:    {roc_auc:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Non-PD Control', 'Parkinson\'s Disease'], zero_division=0))
    
    # 7. Generate and Save Visualizations
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Plot 1: Training & Validation Metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(1, epochs + 1), history['train_loss'], 'o-', label='Train Loss', color='#2563EB', linewidth=2)
    ax1.plot(range(1, epochs + 1), history['val_loss'], 's--', label='Validation Loss', color='#DC2626', linewidth=2)
    ax1.set_title('Cross-Entropy Loss vs. Epochs', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(frameon=True)
    
    ax2.plot(range(1, epochs + 1), [a * 100 for a in history['train_acc']], 'o-', label='Train Accuracy', color='#059669', linewidth=2)
    ax2.plot(range(1, epochs + 1), [a * 100 for a in history['val_acc']], 's--', label='Validation Accuracy', color='#D97706', linewidth=2)
    ax2.set_title('Subject Accuracy vs. Epochs', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.legend(frameon=True)
    plt.tight_layout()
    metrics_path = os.path.join(output_dir, 'training_metrics.png')
    plt.savefig(metrics_path, dpi=300)
    plt.close()
    print(f"Saved Training Curves to: {metrics_path}")
    
    # Plot 2: Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-PD Control', 'Parkinson\'s'],
                yticklabels=['Non-PD Control', 'Parkinson\'s'],
                annot_kws={'size': 14, 'fontweight': 'bold'})
    plt.title('Subject-Level Confusion Matrix (Test Set)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Predicted Label', fontsize=11, labelpad=8)
    plt.ylabel('Ground Truth Label', fontsize=11, labelpad=8)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix to: {cm_path}")
    
    # Plot 3: ROC Curve
    try:
        fpr, tpr, _ = roc_curve(test_labels, test_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='#7C3AED', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='#9CA3AF', lw=1.5, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
        plt.title('Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold', pad=12)
        plt.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        roc_path = os.path.join(output_dir, 'roc_curve.png')
        plt.savefig(roc_path, dpi=300)
        plt.close()
        print(f"Saved ROC Curve to: {roc_path}")
    except Exception as e:
        print(f"Skipping ROC plot: {e}")

if __name__ == '__main__':
    main()
