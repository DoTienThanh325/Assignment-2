import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Thiết lập seed ngẫu nhiên để đảm bảo kết quả tái lập
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Tải và tiền xử lý tập dữ liệu CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 2. Xây dựng MLP (3 tầng)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# 3. Xây dựng CNN (3 tầng chập)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.fc2(x)
        return x

# 4. Hàm huấn luyện mô hình
def train_model(model, trainloader, criterion, optimizer, num_epochs=10):
    train_losses, train_accuracies = [], []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(trainloader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    return train_losses, train_accuracies

# 5. Hàm đánh giá mô hình
def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels

# 6. Vẽ đường cong học tập
def plot_learning_curves(mlp_losses, mlp_accuracies, cnn_losses, cnn_accuracies):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(mlp_losses, label='MLP Loss')
    plt.plot(cnn_losses, label='CNN Loss')
    plt.title('Độ Mất Mát Huấn Luyện')
    plt.xlabel('Vòng Lặp')
    plt.ylabel('Mất Mát')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(mlp_accuracies, label='MLP Accuracy')
    plt.plot(cnn_accuracies, label='CNN Accuracy')
    plt.title('Độ Chính Xác Huấn Luyện')
    plt.xlabel('Vòng Lặp')
    plt.ylabel('Độ Chính Xác (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('duong_cong_hoc_tap.png')
    plt.show()

# 7. Vẽ ma trận nhầm lẫn
def plot_confusion_matrix(labels, preds, title):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Dự Đoán')
    plt.ylabel('Thực Tế')
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()

# 8. Thực thi chính
# Huấn luyện và đánh giá MLP
mlp = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp.parameters(), lr=0.001)
mlp_losses, mlp_accuracies = train_model(mlp, trainloader, criterion, optimizer)
mlp_test_acc, mlp_preds, mlp_labels = evaluate_model(mlp, testloader)
print(f"Độ chính xác kiểm tra MLP: {mlp_test_acc:.2f}%")

# Huấn luyện và đánh giá CNN
cnn = CNN().to(device)
optimizer = optim.Adam(cnn.parameters(), lr=0.001)
cnn_losses, cnn_accuracies = train_model(cnn, trainloader, criterion, optimizer)
cnn_test_acc, cnn_preds, cnn_labels = evaluate_model(cnn, testloader)
print(f"Độ chính xác kiểm tra CNN: {cnn_test_acc:.2f}%")

# Vẽ kết quả
plot_learning_curves(mlp_losses, mlp_accuracies, cnn_losses, cnn_accuracies)
plot_confusion_matrix(mlp_labels, mlp_preds, "Ma Trận Nhầm Lẫn MLP")
plot_confusion_matrix(cnn_labels, cnn_preds, "Ma Trận Nhầm Lẫn CNN")