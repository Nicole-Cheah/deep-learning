import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from models.cnn import get_resnet50_model

# ===== SETTINGS =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_resnet50_ai_face_detector.pth"
DATA_PATH = "data/test"
BATCH_SIZE = 32
THRESHOLD = 0.80  # tuned threshold

# ===== TRANSFORMS =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== LOAD DATA =====
test_dataset = datasets.ImageFolder(DATA_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===== LOAD MODEL =====
model = get_resnet50_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ===== EVALUATION =====
all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = (probs >= THRESHOLD).long()

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())

# ===== RESULTS =====
print("===== TEST RESULTS =====")
print(classification_report(all_labels, all_preds, target_names=["Real", "AI-Generated"]))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))