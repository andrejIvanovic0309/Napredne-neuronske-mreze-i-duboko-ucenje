import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from google.colab import drive

# Priprema
data_dir = '/content/train'
test_data_dir = '/content/test'

label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class Fer2013Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_map = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
        self.load_data()

    def load_data(self):
        for emotion in os.listdir(self.root_dir):
            emotion_path = os.path.join(self.root_dir, emotion)
            if os.path.isdir(emotion_path):
                for img_name in os.listdir(emotion_path):
                    img_path = os.path.join(emotion_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.label_map[emotion])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, axis=2)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

train_dataset = Fer2013Dataset(root_dir=data_dir, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

test_dataset = Fer2013Dataset(root_dir=test_data_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Podaci su učitani i spremni za treniranje.")

# KReiranje modela
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv_layers = nn.Sequential(
        
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Koristim uređaj: {device}")

model = EmotionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
start_time = time.time()
total_batches = len(train_loader)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / total_batches
    print(f"--- Epoha {epoch+1}/{num_epochs} završena ---")
    print(f"Gubitak na treningu: {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Tačnost na validacijskom setu: {accuracy:.2f}%')
        model.train()

    elapsed_time = time.time() - start_time
    avg_epoch_time = elapsed_time / (epoch + 1)
    remaining_time = avg_epoch_time * (num_epochs - (epoch + 1))
    minutes = int(remaining_time / 60)
    seconds = int(remaining_time % 60)
    print(f"Procenjeno preostalo vreme: {minutes} minuta i {seconds} sekundi")

# Cuvanje modela
try:
    drive.mount('/content/drive')
    torch.save(model.state_dict(), '/content/drive/My Drive/emotion_model5.pth')
    print("Model je sačuvan na Google Drive-u.")
except Exception as e:
    print(f"Greška prilikom čuvanja modela na Google Drive-u: {e}")