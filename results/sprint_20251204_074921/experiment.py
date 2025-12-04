import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from pathlib import Path

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to create synthetic training data
def create_synthetic_data(num_samples: int):
    X = np.random.rand(num_samples, 10).astype(np.float32)
    y = (np.sum(X, axis=1) > 5).astype(np.float32).reshape(-1, 1)  # Binary classification
    return torch.from_numpy(X), torch.from_numpy(y)

# Function to train the model
def train_model(batch_size: int, num_epochs: int = 5):
    model = SimpleNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create synthetic data
    X, y = create_synthetic_data(1000)
    dataset = torch.utils.data.TensorDataset(X, y)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    metrics = {
        'batch_size': batch_size,
        'training_time': 0,
        'final_loss': None,
        'model_size_MB': 0,
        'inference_time': 0,
        'memory_usage_MB': 0
    }

    # Training loop
    start_time = time.time()
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(data_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i}/{len(data_loader)}], Loss: {loss.item():.4f}')  

    end_time = time.time()
    metrics['training_time'] = end_time - start_time
    metrics['final_loss'] = loss.item()
    metrics['model_size_MB'] = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)  # Approximate size in MB

    # Inference time measurement
    model.eval()
    with torch.no_grad():
        inference_start = time.time()
        _ = model(X)
        inference_end = time.time()
        metrics['inference_time'] = inference_end - inference_start

    # Memory usage measurement (approximation)
    metrics['memory_usage_MB'] = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    return metrics

# Main function to run the experiment
if __name__ == '__main__':
    batch_sizes = [16, 32, 64, 128, 256]
    results = []

    for batch_size in batch_sizes:
        print(f'Running experiment with batch size: {batch_size}')
        metrics = train_model(batch_size)
        results.append(metrics)

    # Save results to JSON file
    results_path = Path('experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f'Results saved to {results_path}')
    print('Experiment completed successfully!')