import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a simple neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train the model with gradient clipping
def train_model(clip_threshold: float, num_epochs: int = 10, batch_size: int = 64):
    try:
        # Load MNIST dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model, loss function, and optimizer
        model = SimpleNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        # Training loop
        for epoch in range(num_epochs):
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_threshold)

                optimizer.step()

                # Log metrics
                logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        return model
    except Exception as e:
        logging.error(f'Error during training: {e}')
        return None

# Function to run experiments with different clipping thresholds
def run_experiments(thresholds: list, num_epochs: int = 10):
    results = {}
    for threshold in thresholds:
        logging.info(f'Starting training with gradient clipping threshold: {threshold}')
        model = train_model(threshold, num_epochs)
        if model:
            results[threshold] = 'Training completed successfully'
        else:
            results[threshold] = 'Training failed'
    return results

# Main function to execute the experiment
if __name__ == '__main__':
    clipping_thresholds = [0.1, 0.5, 1.0, 5.0]
    num_epochs = 10
    results = run_experiments(clipping_thresholds, num_epochs)

    # Save results to a JSON file
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f)
    logging.info('Experiment results saved to experiment_results.json')
