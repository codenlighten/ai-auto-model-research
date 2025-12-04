import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import json
from torchvision import datasets, transforms
from pathlib import Path

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def distillation_loss(y_student, y_teacher, T, alpha):
    soft_loss = nn.KLDivLoss()(F.log_softmax(y_student / T, dim=1), F.softmax(y_teacher / T, dim=1)) * (T * T)
    return alpha * soft_loss

# Data generation
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Initialize models, optimizers, and hyperparameters
teacher_model = TeacherModel().to(device)
student_model = StudentModel().to(device)
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

# Training parameters
num_epochs = 5
alpha = 0.5
T_values = [1, 2, 3, 5]

# Metrics storage
metrics = {'train_time': [], 'final_loss': [], 'final_accuracy': [], 'model_size_MB': 0, 'inference_time': 0}

# Training loop
for T in T_values:
    print(f'Running distillation with temperature: {T}')
    start_time = time.time()
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.view(images.size(0), -1).to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            student_outputs = student_model(images)
            loss = distillation_loss(student_outputs, teacher_outputs, T, alpha) + nn.CrossEntropyLoss()(student_outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(student_outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    end_time = time.time()
    metrics['train_time'].append(end_time - start_time)
    metrics['final_loss'].append(epoch_loss)
    metrics['final_accuracy'].append(epoch_accuracy)

# Model size in MB
metrics['model_size_MB'] = sum(p.numel() for p in student_model.parameters()) * 4 / (1024 ** 2)

# Inference time measurement
start_inference = time.time()
with torch.no_grad():
    student_model.eval()
    for images, _ in train_loader:
        images = images.view(images.size(0), -1).to(device)
        _ = student_model(images)
end_inference = time.time()
metrics['inference_time'] = end_inference - start_inference

# Save metrics to JSON
results_path = Path('./results')
results_path.mkdir(exist_ok=True)
results_file = results_path / f'results_{int(time.time())}.json'
with open(results_file, 'w') as f:
    json.dump(metrics, f, indent=4)

# Print final results
print(f'Final metrics saved to {results_file}')
print(json.dumps(metrics, indent=4))