Violence Detection using Multi-Frame CNN (Without LSTM)
# Uses multiple frames + powerful CNN architecture

# ============================================================
# STEP 1: Import Libraries
# ============================================================
import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using: {device}")

# ============================================================
# STEP 2: Settings
# ============================================================
DATA_PATH = '/kaggle/input/real-life-violence-situations-dataset/Real Life Violence Dataset'
VIOLENCE_PATH = os.path.join(DATA_PATH, 'Violence')
NON_VIOLENCE_PATH = os.path.join(DATA_PATH, 'NonViolence')

IMG_SIZE = 224
NUM_FRAMES = 16        # Extract 16 frames per video
BATCH_SIZE = 8         # Smaller batch for multiple frames
EPOCHS = 15
LR = 0.0001

# ============================================================
# STEP 3: Extract Multiple Frames from Videos
# ============================================================
def extract_frames(video_path, num_frames=NUM_FRAMES):
    """Extract evenly spaced frames from video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return None
    
    # Get evenly spaced frame indices
    if total_frames < num_frames:
        frame_indices = list(range(total_frames)) + [total_frames-1] * (num_frames - total_frames)
    else:
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frames.append(frame)
    
    cap.release()
    
    if len(frames) == num_frames:
        return np.array(frames)
    return None

# ============================================================
# STEP 4: Load Dataset
# ============================================================
print("Loading videos and extracting frames...")
data = []

# Violence videos
if os.path.exists(VIOLENCE_PATH):
    violence_videos = [f for f in os.listdir(VIOLENCE_PATH) if f.endswith(('.mp4', '.avi'))]
    for video in tqdm(violence_videos, desc="Violence"):
        video_path = os.path.join(VIOLENCE_PATH, video)
        frames = extract_frames(video_path)
        if frames is not None:
            data.append({'frames': frames, 'label': 1})

# Non-Violence videos
if os.path.exists(NON_VIOLENCE_PATH):
    non_violence_videos = [f for f in os.listdir(NON_VIOLENCE_PATH) if f.endswith(('.mp4', '.avi'))]
    for video in tqdm(non_violence_videos, desc="Non-Violence"):
        video_path = os.path.join(NON_VIOLENCE_PATH, video)
        frames = extract_frames(video_path)
        if frames is not None:
            data.append({'frames': frames, 'label': 0})

print(f"\nTotal videos: {len(data)}")
print(f"Violence: {sum(1 for d in data if d['label']==1)}")
print(f"Non-Violence: {sum(1 for d in data if d['label']==0)}")

# Split data
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"\nTrain: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# ============================================================
# STEP 5: Dataset Class
# ============================================================
class VideoDataset(Dataset):
    def init(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def len(self):
        return len(self.data)
    
    def getitem(self, idx):
        frames = self.data[idx]['frames']
        label = self.data[idx]['label']
        
        if self.transform:
            transformed_frames = []
            for frame in frames:
                transformed_frames.append(self.transform(frame))
            frames = torch.stack(transformed_frames)
        
        return frames, label

# ============================================================
# STEP 6: Data Transforms
# ============================================================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# STEP 7: Data Loaders
# ============================================================
train_dataset = VideoDataset(train_data, train_transform)
val_dataset = VideoDataset(val_data, test_transform)
test_dataset = VideoDataset(test_data, test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ============================================================
# STEP 8: Multi-Frame CNN Model (No LSTM)
# ============================================================
class MultiFrameCNN(nn.Module):
    def init(self, num_frames=NUM_FRAMES, num_classes=2):
        super(MultiFrameCNN, self).init()
        
        # Use ResNet50 as backbone (more powerful than ResNet18)
        resnet = models.resnet50(pretrained=True)
        
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze early layers for faster training
        for param in list(self.feature_extractor.parameters())[:-30]:
            param.requires_grad = False
        
        # Feature dimension from ResNet50
        self.feature_dim = 2048
        
        # Process multiple frames - each frame gives 2048 features
        # Total: num_frames * 2048 features
        
        # Temporal pooling options: max, avg, or attention
        # Here we use both max and avg pooling
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim * 2, 1024),  # *2 for max+avg pooling
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, num_frames, channels, height, width)
        batch_size, num_frames, c, h, w = x.size()
        
        # Reshape to process all frames together
        x = x.view(batch_size * num_frames, c, h, w)
        
        # Extract features for all frames
        features = self.feature_extractor(x)
        features = features.view(batch_size, num_frames, -1)
        
        # Temporal pooling across frames
        # Max pooling: captures most prominent features across time
        max_pooled = torch.max(features, dim=1)[0]
        
        # Average pooling: captures average activity across time
        avg_pooled = torch.mean(features, dim=1)
        
        # Concatenate both pooling strategies
        combined = torch.cat([max_pooled, avg_pooled], dim=1)
        
        # Final classification
        output = self.classifier(combined)
        
        return output

model = MultiFrameCNN(num_frames=NUM_FRAMES).to(device)
print(f"\nModel created with ResNet50 backbone!")
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================
# STEP 9: Loss and Optimizer
# ============================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# ============================================================
# STEP 10: Training Function
# ============================================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    
    for frames, labels in tqdm(loader, desc="Training"):
        frames, labels = frames.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * frames.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Validation"):
            frames, labels = frames.to(device), labels.to(device)
            
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * frames.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ============================================================
# STEP 11: Training Loop
# ============================================================
print("\n" + "="*60)
print("Starting Training...")
print("="*60)

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print("-" * 60)
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion)
    
    scheduler.step(val_loss)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_multiframe_model.pth')
        print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)

# ============================================================


# STEP 12: Plot Results
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history['train_loss'], 'o-', label='Train Loss')
ax1.plot(history['val_loss'], 'o-', label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(history['train_acc'], 'o-', label='Train Acc')
ax2.plot(history['val_acc'], 'o-', label='Val Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_results.png', dpi=150)
plt.show()


# ============================================================
# STEP 13: Test the Model
# ============================================================
print("\n" + "="*60)
print("Testing on Test Set...")
print("="*60)

model.load_state_dict(torch.load('best_multiframe_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for frames, labels in tqdm(test_loader, desc="Testing"):
        frames = frames.to(device)
        outputs = model(frames)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Calculate metrics
test_acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['Non-Violence', 'Violence']))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Non-Violence', 'Violence'],
            yticklabels=['Non-Violence', 'Violence'])
plt.title('Confusion Matrix - Multi-Frame CNN')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

print("\n✓ Done! Model saved as 'best_multiframe_model.pth'")
