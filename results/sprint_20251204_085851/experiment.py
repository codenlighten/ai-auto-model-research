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
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

# Define the Student Model (2-layer MLP)
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Generate synthetic MNIST-like data
def generate_data(num_samples=1000):
    X = np.random.rand(num_samples, 28 * 28).astype(np.float32)
    y = np.random.randint(0, 10, num_samples)
    return torch.tensor(X), torch.tensor(y)

# Training function
def train_model(teacher_model, student_model, train_loader, epochs=5, temperature=2.0, alpha=0.5):
    criterion = nn.CrossEntropyLoss()
    optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=0.001)
    optimizer_student = optim.Adam(student_model.parameters(), lr=0.001)

    for epoch in range(epochs):
        teacher_model.train()
        for data, target in train_loader:
            optimizer_teacher.zero_grad()
            output = teacher_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer_teacher.step()

        # Generate soft labels from teacher model
        with torch.no_grad():
            soft_labels = nn.functional.softmax(output / temperature, dim=1)

        student_model.train()
        for data, target in train_loader:
            optimizer_student.zero_grad()
            student_output = student_model(data)
            student_loss = alpha * criterion(student_output, target) + (1 - alpha) * nn.KLDivLoss()(nn.functional.log_softmax(student_output / temperature, dim=1), soft_labels)
            student_loss.backward()
            optimizer_student.step()

# Main execution
if __name__ == '__main__':
    start_time = time.time()
    teacher_model = TeacherModel()
    student_model = StudentModel()

    # Create synthetic dataset
    X, y = generate_data(1000)
    train_loader = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=32, shuffle=True)

    # Train the models
    train_model(teacher_model, student_model, train_loader)

    # Evaluate models
    # (This part is simplified for brevity)
    print('Training completed.')
    print(f'Total training time: {time.time() - start_time:.2f} seconds')

    # Save metrics to JSON
    metrics = {
        'training_time': time.time() - start_time,
        'teacher_model_parameters': sum(p.numel() for p in teacher_model.parameters()),
        'student_model_parameters': sum(p.numel() for p in student_model.parameters()),
    }
    results_path = Path('results.json')
    with results_path.open('w') as f:
        json.dump(metrics, f)
    print(f'Metrics saved to {results_path}' )