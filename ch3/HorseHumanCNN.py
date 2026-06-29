import urllib.request, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class HorsesHumansCNN(nn.Module):
    def __init__(self):
        super(HorsesHumansCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 18 * 18, 512)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 18 * 18)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


def load_data():
    url = "https://storage.googleapis.com/learning-datasets/horse-or-human.zip"
    file_name = "horse-or-human.zip"
    training_dir = "horse-or-human/training/"
    urllib.request.urlretrieve(url, file_name)

    zip_ref = zipfile.ZipFile(file_name, "r")
    zip_ref.extractall(training_dir)
    zip_ref.close()

    url = (
        "https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip"
    )
    file_name = "validation-horse-or-human.zip"
    validation_dir = "horse-or-human/validation/"
    urllib.request.urlretrieve(url, file_name)

    zip_ref = zipfile.ZipFile(file_name, "r")
    zip_ref.extractall(validation_dir)
    zip_ref.close()

    train_transform = transforms.Compose(
        [
            transforms.Resize((150, 150)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(
                degrees=0,  # No rotation
                translate=(0.2, 0.2),  # Translate up to 20% vertically and horizontally
                scale=(0.8, 1.2),  # Zoom in or out by 20%
                shear=20,  # Shear by up to 20 degrees
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Load the datasets
    train_dataset = datasets.ImageFolder(root=training_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=validation_dir, transform=train_transform)

    return train_dataset, val_dataset


class HorseHumanNetwork:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train_model(self, num_epochs, train_loader, val_loader):
        for epoch in range(num_epochs):
            # 모델 학습
            self.model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(images).view(-1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

            # 훈련 세트에 대한 평가
            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in train_loader:
                    outputs = self.model(images).view(-1)
                    predicted = outputs > 0.5  # Threshold predictions
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print(f"Training Set Accuracy: {100 * correct / total}%")

            # 검증 세트에 대한 평가
            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in val_loader:
                    outputs = self.model(images).view(-1)
                    predicted = outputs > 0.5  # Threshold predictions
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print(f"Validation Set Accuracy: {100 * correct / total}%")


# Data loaders
train_dataset, val_dataset = load_data()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# Make network
model = HorsesHumansCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
net = HorseHumanNetwork(model, criterion, optimizer)

# Train
net.train_model(15, train_loader, val_loader)
