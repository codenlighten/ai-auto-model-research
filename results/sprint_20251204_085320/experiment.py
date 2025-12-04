import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from pathlib import Path

# Define the Teacher Model (4-layer MLP)
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Define the Student Model (2-layer MLP)
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to generate synthetic MNIST-like data
def generate_data(num_samples=1000):
    X = np.random.rand(num_samples, 784).astype(np.float32)  # Random data
    y = np.random.randint(0, 10, size=(num_samples,)).astype(np.int64)  # Random labels
    return torch.tensor(X), torch.tensor(y)

# Training function
def train_model(model, criterion, optimizer, train_loader, num_epochs=5):
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}], Loss: {loss.item():.4f}')  

    training_time = time.time() - start_time
    return training_time

# Function to evaluate the model
def evaluate_model(model, test_data):
    model.eval()
    with torch.no_grad():
        inputs, labels = test_data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        return accuracy

# Main execution
if __name__ == '__main__':
    # Create synthetic data
    X_train, y_train = generate_data(1000)
    X_val, y_val = generate_data(200)
    train_loader = [(X_train, y_train)]
    test_data = (X_val, y_val)

    # Initialize models, criterion and optimizer
    teacher_model = TeacherModel()
    student_model = StudentModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)

    # Train the student model
    training_time = train_model(student_model, criterion, optimizer, train_loader)

    # Evaluate the student model
    accuracy = evaluate_model(student_model, test_data)
    model_size = sum(p.numel() for p in student_model.parameters()) * 4 / (1024 ** 2)  # Size in MB

    # Prepare results
    results = {
        'training_time': training_time,
        'final_accuracy': accuracy,
        'model_size_mb': model_size,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save results to JSON file
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / 'experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Print final results
    print('Experiment completed. Results saved to experiment_results.json')
    print(f'Training Time: {training_time:.2f} seconds')
    print(f'Final Accuracy: {accuracy:.4f}')
    print(f'Model Size: {model_size:.4f} MB')