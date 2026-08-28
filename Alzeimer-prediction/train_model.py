import os
import random
import time
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, cohen_kappa_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from tqdm import tqdm

# --- 1. CONFIGURATION ---
RANDOM_SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join("Alzeimer-prediction", "Alzeimer", "combined_images")
MODEL_SAVE_PATH = os.path.join("Alzeimer-prediction", "best_alzheimer_model.pth")
CONF_MATRIX_PATH = os.path.join("Alzeimer-prediction", "confusion_matrix.png")
TRAIN_PLOT_PATH = os.path.join("Alzeimer-prediction", "training_metrics.png")
NOTEBOOK_PATH = os.path.join("Alzeimer-prediction", "train.ipynb")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

print(f"Using device: {DEVICE}")

# --- 2. DATASET PREPARATION ---
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
NUM_CLASSES = len(CLASS_NAMES)
class_to_idx = {cls_name: i for i, cls_name in enumerate(CLASS_NAMES)}

records = []
# Sample a representative balanced subset for efficient, high-performance training
SAMPLES_PER_CLASS = 1000

for cls_name in CLASS_NAMES:
    cls_folder = os.path.join(DATA_DIR, cls_name)
    if not os.path.exists(cls_folder):
        continue
    all_files = glob.glob(os.path.join(cls_folder, "*.jpg")) + glob.glob(os.path.join(cls_folder, "*.png"))
    all_files.sort()
    
    # Stratified/seeded selection
    random.seed(RANDOM_SEED)
    selected_files = random.sample(all_files, min(SAMPLES_PER_CLASS, len(all_files)))
    for p in selected_files:
        records.append({
            "image_path": p,
            "class_name": cls_name,
            "diagnosis": class_to_idx[cls_name]
        })

df = pd.DataFrame(records)
print(f"Total dataset records: {len(df)}")
print(df["class_name"].value_counts())

# Train (70%), Val (15%), Test (15%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["diagnosis"],
    random_state=RANDOM_SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["diagnosis"],
    random_state=RANDOM_SEED
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

# --- 3. PYTORCH DATASET & TRANSFORMS ---
class AlzheimerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["diagnosis"])
        if self.transform:
            image = self.transform(image)
        return image, label

train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = AlzheimerDataset(train_df, transform=train_transform)
val_dataset = AlzheimerDataset(val_df, transform=val_transform)
test_dataset = AlzheimerDataset(test_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# --- 4. MODEL INITIALIZATION ---
print("Initializing EfficientNet-B0 for Alzheimer's Classification...")
weights = EfficientNet_B0_Weights.DEFAULT
model = efficientnet_b0(weights=weights)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, NUM_CLASSES)
model = model.to(DEVICE)

# Freeze backbone initially
for param in model.features.parameters():
    param.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

# --- 5. TRAINING FUNCTIONS ---
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return running_loss / len(loader.dataset), acc, f1, all_labels, all_preds

# --- 6. TRAINING LOOP ---
HEAD_EPOCHS = 4
FINETUNE_EPOCHS = 3
TOTAL_EPOCHS = HEAD_EPOCHS + FINETUNE_EPOCHS

history = {
    "train_loss": [], "train_acc": [],
    "val_loss": [], "val_acc": [], "val_f1": []
}

best_val_f1 = 0.0
print("\n--- Phase 1: Training Classifier Head ---")
for epoch in range(HEAD_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion)
    
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)
    
    print(f"Epoch {epoch+1}/{TOTAL_EPOCHS} | Train Acc: {train_acc:.4f} Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} Loss: {val_loss:.4f} F1: {val_f1:.4f}")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  --> Saved new best model to {MODEL_SAVE_PATH}")

print("\n--- Phase 2: Fine-Tuning Top Blocks ---")
# Unfreeze top feature layers
for param in model.features[-2:].parameters():
    param.requires_grad = True

ft_optimizer = torch.optim.AdamW([
    {"params": model.features[-2:].parameters(), "lr": 1e-4},
    {"params": model.classifier.parameters(), "lr": 5e-4}
], weight_decay=1e-4)

for epoch in range(FINETUNE_EPOCHS):
    curr_epoch = HEAD_EPOCHS + epoch + 1
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, ft_optimizer)
    val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion)
    
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)
    
    print(f"Epoch {curr_epoch}/{TOTAL_EPOCHS} | Train Acc: {train_acc:.4f} Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} Loss: {val_loss:.4f} F1: {val_f1:.4f}")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  --> Saved new best model to {MODEL_SAVE_PATH}")

# --- 7. TEST EVALUATION ---
print("\nLoading best checkpoint for evaluation...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
test_loss, test_acc, test_f1, y_true, y_pred = evaluate(model, test_loader, criterion)
kappa = cohen_kappa_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)

print("\n" + "="*60)
print("FINAL ALZHEIMER'S MODEL TEST EVALUATION")
print("="*60)
print(f"Test Loss                : {test_loss:.4f}")
print(f"Test Accuracy            : {test_acc*100:.2f}%")
print(f"Test Macro F1            : {test_f1:.4f}")
print(f"Cohen's Kappa Score      : {kappa:.4f}")
print("\nClassification Report:\n", report)

# --- 8. PLOT & SAVE VISUALIZATIONS ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Alzheimer's Disease Classification Confusion Matrix", fontsize=14, pad=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH, dpi=300)
plt.close()
print(f"Saved confusion matrix plot to {CONF_MATRIX_PATH}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, TOTAL_EPOCHS + 1), history["train_loss"], label="Train Loss", marker='o')
plt.plot(range(1, TOTAL_EPOCHS + 1), history["val_loss"], label="Val Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(range(1, TOTAL_EPOCHS + 1), history["train_acc"], label="Train Accuracy", marker='o')
plt.plot(range(1, TOTAL_EPOCHS + 1), history["val_acc"], label="Val Accuracy", marker='o')
plt.plot(range(1, TOTAL_EPOCHS + 1), history["val_f1"], label="Val Macro F1", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Accuracy & Macro F1 Evolution")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(TRAIN_PLOT_PATH, dpi=300)
plt.close()
print(f"Saved training plot to {TRAIN_PLOT_PATH}")
