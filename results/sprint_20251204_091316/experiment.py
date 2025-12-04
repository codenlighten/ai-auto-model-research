import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from pathlib import Path

# Define the Teacher and Student MLP models
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

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.fc2 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Create synthetic dataset resembling MNIST
def create_synthetic_data(num_samples=1000):
    X = np.random.rand(num_samples, 28 * 28).astype(np.float32)
    y = np.random.randint(0, 10, size=(num_samples,)).astype(np.int64)
    return torch.tensor(X), torch.tensor(y)

# Training loop
def train_model(teacher, student, data, labels, num_epochs=10, batch_size=32, temperature=2.0, alpha=0.5):
    criterion_hard = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters())
    metrics = {'loss': [], 'accuracy': []}
    start_time = time.time()

    for epoch in range(num_epochs):
        for i in range(0, len(data), batch_size):
            inputs = data[i:i + batch_size]
            targets = labels[i:i + batch_size]

            optimizer.zero_grad()
            soft_labels = teacher(inputs) / temperature
            outputs = student(inputs)

            loss_hard = criterion_hard(outputs, targets)
            loss_soft = nn.KLDivLoss()(nn.functional.log_softmax(outputs / temperature, dim=1), nn.functional.softmax(soft_labels, dim=1))
            loss = alpha * loss_hard + (1 - alpha) * loss_soft
            loss.backward()
            optimizer.step()

            if i % (batch_size * 10) == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i // batch_size + 1}], Loss: {loss.item():.4f}')
                metrics['loss'].append(loss.item())

    end_time = time.time()
    training_time = end_time - start_time
    return metrics, training_time

# Evaluate model
def evaluate_model(model, data, labels):
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).sum().item() / len(labels)
    return accuracy

# Save results to JSON
def save_results(metrics, training_time):
    results = {
        'metrics': metrics,
        'training_time': training_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    results_path = Path('results.json')
    with results_path.open('w') as f:
        json.dump(results, f, indent=4)
    print(f'Results saved to {results_path}')  

# Main execution
if __name__ == '__main__':
    # Create models
    teacher = TeacherModel()
    student = StudentModel()

    # Create synthetic data
    data, labels = create_synthetic_data()

    # Train teacher model (for demonstration, we skip actual training)
    # teacher.train()  # Uncomment to train the teacher model

    # Train student model
    metrics, training_time = train_model(teacher, student, data, labels)

    # Evaluate student model
    accuracy = evaluate_model(student, data, labels)
    metrics['accuracy'] = accuracy

    # Save results
    save_results(metrics, training_time)

    # Print final results
    print(f'Final Accuracy: {accuracy:.4f}')