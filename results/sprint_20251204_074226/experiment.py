import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import json
import pathlib
import numpy as np

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to train the model
def train_model(optimizer_name: str, model: nn.Module, criterion: nn.Module, train_loader: torch.utils.data.DataLoader, num_epochs: int = 5) -> dict:
    metrics = {'loss': [], 'accuracy': [], 'training_time': 0}
    optimizer = get_optimizer(optimizer_name, model.parameters())
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Loss calculation
            loss.backward()  # Backward pass
            optimizer.step()  # Optimizer step

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        metrics['loss'].append(avg_loss)
        metrics['accuracy'].append(accuracy)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    metrics['training_time'] = time.time() - start_time
    return metrics

# Function to get the optimizer
def get_optimizer(name: str, parameters: torch.nn.Parameter):
    if name == 'adam':
        return optim.Adam(parameters)
    elif name == 'adamw':
        return optim.AdamW(parameters)
    elif name == 'sgd':
        return optim.SGD(parameters, momentum=0.9)
    else:
        raise ValueError(f'Unknown optimizer: {name}')

# Main function to run the experiment
def run_experiment():
    # Load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    # Initialize model, loss function
    model = MLP(input_size=28*28, hidden_size=128, output_size=10)
    criterion = nn.CrossEntropyLoss()

    # Track results for each optimizer
    results = {}
    for optimizer_name in ['adam', 'adamw', 'sgd']:
        print(f'Running experiment with optimizer: {optimizer_name}')
        metrics = train_model(optimizer_name, model, criterion, train_loader)
        results[optimizer_name] = metrics

    # Save results to JSON file
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    results_file = pathlib.Path(f'optimizer_results_{timestamp}.json')
    with results_file.open('w') as f:
        json.dump(results, f, indent=4)
    print(f'Results saved to {results_file}')

if __name__ == '__main__':
    run_experiment()