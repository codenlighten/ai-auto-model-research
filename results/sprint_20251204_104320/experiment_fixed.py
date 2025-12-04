import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the SimpleMLP model
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Assuming input size is 784 for MNIST
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)    # Output size for 10 classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Function to train the model
def train_model(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, epochs: int = 5) -> None:
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.view(-1, 784).float(), target.long()  # Reshape and typecast
            optimizer.zero_grad()  # Zero the gradients
            output = model(data)  # Forward pass
            loss = criterion(output, target)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')

# Main function to execute the training with different learning rates
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize model, loss function, and learning rates
    model = SimpleMLP()
    criterion = nn.CrossEntropyLoss()
    learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0]

    # Train the model with different learning rates
    for lr in learning_rates:
        print(f'Training with learning rate: {lr}')
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_model(model, train_loader, criterion, optimizer)

if __name__ == '__main__':
    main()