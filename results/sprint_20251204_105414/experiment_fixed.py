import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ml_utils import create_synthetic_mnist

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def prune_model(model, pruning_ratio):
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Calculate the number of weights to prune
            num_weights = param.numel()
            num_prune = int(num_weights * pruning_ratio)
            # Get the indices of the weights to prune
            prune_indices = torch.topk(param.abs().view(-1), num_prune, largest=False).indices
            # Set the pruned weights to zero
            param.view(-1)[prune_indices] = 0

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    # Hyperparameters
    num_samples = 10000
    batch_size = 64
    pruning_ratio = 0.2

    # Create synthetic MNIST data
    train_loader = create_synthetic_mnist(num_samples, batch_size)

    # Initialize model, loss function, and optimizer
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    train_model(model, train_loader, criterion, optimizer)

    # Prune the model
    prune_model(model, pruning_ratio)
    
    # Optionally: Evaluate the model after pruning (not implemented in this snippet)