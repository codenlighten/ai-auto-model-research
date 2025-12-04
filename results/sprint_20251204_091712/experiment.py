import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from pathlib import Path

# Define the Teacher and Student Models
class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(0.5))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Function to create synthetic MNIST-like data
def create_synthetic_data(num_samples=1000, img_size=(1, 28, 28), num_classes=10):
    X = np.random.rand(num_samples, *img_size).astype(np.float32)
    y = np.random.randint(0, num_classes, num_samples)
    return torch.tensor(X), torch.tensor(y)

# Training routine
def train_model(model, data_loader, criterion, optimizer, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(images.view(images.size(0), -1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}')  

# Main experiment function
def run_experiment():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create synthetic data
    X, y = create_synthetic_data()
    dataset = torch.utils.data.TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize models
    teacher_model = MLP([784, 128, 64, 32, 10])
    student_model = MLP([784, 64, 10])

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)

    # Train Teacher Model
    print('Training Teacher Model...')
    train_model(teacher_model, data_loader, criterion, optimizer, num_epochs=5)

    # Generate soft labels from Teacher Model
    teacher_model.eval()
    soft_labels = []
    with torch.no_grad():
        for images, _ in data_loader:
            outputs = teacher_model(images.view(images.size(0), -1))
            soft_labels.append(torch.softmax(outputs / 3, dim=1))
    soft_labels = torch.cat(soft_labels)

    # Train Student Model with combined loss
    print('Training Student Model...')
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    student_model.train()
    total_loss = 0.0
    for epoch in range(5):
        for i, (images, labels) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = student_model(images.view(images.size(0), -1))
            hard_loss = criterion(outputs, labels)
            soft_loss = criterion(outputs, soft_labels[i * 32:(i + 1) * 32])
            loss = 0.5 * hard_loss + 0.5 * soft_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f'Step [{i + 1}/{len(data_loader)}], Loss: {loss.item():.4f}')  

    # Save metrics
    metrics = {
        'final_loss': total_loss / len(data_loader),
        'model_size': sum(p.numel() for p in student_model.parameters()),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    results_path = Path('results.json')
    with results_path.open('w') as f:
        json.dump(metrics, f, indent=4)
    print(f'Results saved to {results_path}')

if __name__ == '__main__':
    run_experiment()