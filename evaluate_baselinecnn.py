import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score
import numpy as np

# =====================
# CONFIG
# =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
MODEL_PATH = "bestbaseline_cnn_model.pth"
DATA_DIR = "data"

print(f"Using device: {DEVICE}")


# =====================
# MODEL DEFINITION
# =====================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, padding=2)

    def forward(self, x):
        return torch.cat([
            self.conv1(x),
            self.conv3(x),
            self.conv5(x)
        ], dim=1)


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64, stride=2)

        self.multi = MultiScaleBlock(64, 32)  # output = 96 channels

        self.layer3 = ResidualBlock(96, 128, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.multi(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# =====================
# DATA LOADING
# =====================
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Test dataset size: {len(test_dataset)}")


# =====================
# MODEL LOADING
# =====================
model = CustomCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"Model loaded from: {MODEL_PATH}")


# =====================
# EVALUATION
# =====================
def evaluate_model(model, loader, device):
    """Evaluate model and collect predictions and labels"""
    all_probs = []
    all_preds = []
    all_labels = []
    all_raw_outputs = []
    all_images = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            probs = torch.sigmoid(out).cpu().numpy()
            preds = (probs > 0.5).astype(int).squeeze()
            labels = y.cpu().numpy()

            all_probs.extend(probs.flatten())
            all_preds.extend(preds if preds.ndim > 0 else [preds])
            all_labels.extend(labels)
            all_raw_outputs.extend(out.cpu().numpy().flatten())
            all_images.extend(x.cpu().numpy())

    return (np.array(all_raw_outputs), np.array(all_probs),
            np.array(all_preds), np.array(all_labels), np.array(all_images))


# Run evaluation
raw_outputs, probs, preds, labels, images = evaluate_model(model, test_loader, DEVICE)

# =====================
# METRICS COMPUTATION
# =====================
accuracy = accuracy_score(labels, preds)
conf_matrix = confusion_matrix(labels, preds)
roc_auc = roc_auc_score(labels, probs.flatten())

print("\n" + "="*50)
print("EVALUATION RESULTS (Model Epoch 5)")
print("="*50)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)
print("(True Negatives, False Positives)")
print("(False Negatives, True Positives)")

print("\nClassification Report:")
print(classification_report(labels, preds, target_names=["Class 0", "Class 1"]))

# Breakdown of predictions
tn, fp, fn, tp = conf_matrix.ravel()
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

print("\nDetailed Metrics:")
print(f"Sensitivity (TPR): {sensitivity:.4f}")
print(f"Specificity (TNR): {specificity:.4f}")
print(f"Precision (PPV): {precision:.4f}")


# =====================
# VISUALIZATIONS
# =====================
# 1. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues', aspect='auto')
plt.colorbar()
plt.title('Confusion Matrix - Model Epoch 5')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks([0, 1])
plt.yticks([0, 1])

# Add text annotations
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix[i, j]),
                ha='center', va='center', color='white' if conf_matrix[i, j] > conf_matrix.max()/2 else 'black',
                fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
plt.show()


# 2. ROC Curve
fpr, tpr, thresholds = roc_curve(labels, probs.flatten())

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Model Epoch 5')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
print("✓ ROC curve saved as 'roc_curve.png'")
plt.show()


# 3. Prediction Distribution
plt.figure(figsize=(10, 5))
plt.hist(probs[labels == 0], bins=30, alpha=0.6, label='Class 0 (Negative)', color='blue')
plt.hist(probs[labels == 1], bins=30, alpha=0.6, label='Class 1 (Positive)', color='red')
plt.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Decision Threshold')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities - Model Epoch 5')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('prediction_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Prediction distribution saved as 'prediction_distribution.png'")
plt.show()


# 3b. False Positives (Predicted 1, Actual 0)
false_positives_mask = (preds == 1) & (labels == 0)
false_positive_images = images[false_positives_mask]
false_positive_probs = probs[false_positives_mask]

if len(false_positive_images) > 0:
    num_to_show = min(12, len(false_positive_images))
    plt.figure(figsize=(16, 4))
    for i in range(num_to_show):
        plt.subplot(3, 4, i + 1)
        img = false_positive_images[i].transpose(1, 2, 0)  # CHW -> HWC
        img = np.clip(img, 0, 1)  # Ensure valid range
        plt.imshow(img)
        plt.title(f"FP: Prob={false_positive_probs[i]:.3f}")
        plt.axis('off')
    plt.suptitle(f'False Positives (N={len(false_positive_images)}): Predicted 1 but Actually 0', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('false_positives.png', dpi=150, bbox_inches='tight')
    print(f"✓ False positives saved as 'false_positives.png' ({len(false_positive_images)} total)")
    plt.show()
else:
    print("✗ No false positives found")


# 3c. False Negatives (Predicted 0, Actual 1)
false_negatives_mask = (preds == 0) & (labels == 1)
false_negative_images = images[false_negatives_mask]
false_negative_probs = probs[false_negatives_mask]

if len(false_negative_images) > 0:
    num_to_show = min(12, len(false_negative_images))
    plt.figure(figsize=(16, 4))
    for i in range(num_to_show):
        plt.subplot(3, 4, i + 1)
        img = false_negative_images[i].transpose(1, 2, 0)  # CHW -> HWC
        img = np.clip(img, 0, 1)  # Ensure valid range
        plt.imshow(img)
        plt.title(f"FN: Prob={false_negative_probs[i]:.3f}")
        plt.axis('off')
    plt.suptitle(f'False Negatives (N={len(false_negative_images)}): Predicted 0 but Actually 1', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('false_negatives.png', dpi=150, bbox_inches='tight')
    print(f"✓ False negatives saved as 'false_negatives.png' ({len(false_negative_images)} total)")
    plt.show()
else:
    print("✗ No false negatives found")


# 4. Threshold Analysis
thresholds_to_test = np.arange(0.1, 0.9, 0.05)
accuracies = []
precisions = []
recalls = []

for threshold in thresholds_to_test:
    preds_threshold = (probs.flatten() > threshold).astype(int)
    acc = accuracy_score(labels, preds_threshold)
    cm_tmp = confusion_matrix(labels, preds_threshold)
    tn_tmp, fp_tmp, fn_tmp, tp_tmp = cm_tmp.ravel()

    prec = tp_tmp / (tp_tmp + fp_tmp) if (tp_tmp + fp_tmp) > 0 else 0
    rec = tp_tmp / (tp_tmp + fn_tmp) if (tp_tmp + fn_tmp) > 0 else 0

    accuracies.append(acc)
    precisions.append(prec)
    recalls.append(rec)

plt.figure(figsize=(10, 5))
plt.plot(thresholds_to_test, accuracies, marker='o', label='Accuracy')
plt.plot(thresholds_to_test, precisions, marker='s', label='Precision')
plt.plot(thresholds_to_test, recalls, marker='^', label='Recall')
plt.xlabel('Decision Threshold')
plt.ylabel('Score')
plt.title('Metrics vs Decision Threshold - Model Epoch 5')
plt.legend()
plt.grid(alpha=0.3)
plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Default Threshold')
plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Threshold analysis saved as 'threshold_analysis.png'")
plt.show()


# =====================
# SAVE RESULTS SUMMARY
# =====================
with open('evaluation_results.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("MODEL EVALUATION RESULTS - EPOCH 5\n")
    f.write("="*60 + "\n\n")

    f.write(f"Model Path: {MODEL_PATH}\n")
    f.write(f"Test Dataset Size: {len(test_dataset)}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Device: {DEVICE}\n\n")

    f.write("PERFORMANCE METRICS:\n")
    f.write("-"*60 + "\n")
    f.write(f"Accuracy:        {accuracy:.4f}\n")
    f.write(f"ROC AUC Score:   {roc_auc:.4f}\n")
    f.write(f"Sensitivity:     {sensitivity:.4f}\n")
    f.write(f"Specificity:     {specificity:.4f}\n")
    f.write(f"Precision:       {precision:.4f}\n\n")

    f.write("CONFUSION MATRIX:\n")
    f.write("-"*60 + "\n")
    f.write(f"True Negatives:  {tn}\n")
    f.write(f"False Positives: {fp}\n")
    f.write(f"False Negatives: {fn}\n")
    f.write(f"True Positives:  {tp}\n\n")

    f.write("DETAILED CLASSIFICATION REPORT:\n")
    f.write("-"*60 + "\n")
    f.write(classification_report(labels, preds, target_names=["Class 0", "Class 1"]))

print("✓ Results summary saved as 'evaluation_results.txt'")
print("\n" + "="*50)
print("Evaluation complete!")
print("="*50)
