import torch
import torch.nn as nn
import torch.optim as optim
from ml_utils import prune_model

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def run_experiment(pruning_ratios, train_loader, test_loader):
    for ratio in pruning_ratios:
        model = SimpleCNN()
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        # Train the model
        train_model(model, train_loader, criterion, optimizer)

        # Prune the model
        try:
            prune_model(model, pruning_ratio=ratio)
        except Exception as e:
            print(f'Error during pruning with ratio {ratio}: {e}')  
            continue

        # Evaluate the model
        accuracy = evaluate_model(model, test_loader)
        print(f'Accuracy after pruning with ratio {ratio}: {accuracy:.4f}')

# Example usage
if __name__ == '__main__':
    # Assuming train_loader and test_loader are defined
    pruning_ratios = [0.1, 0.2, 0.3]
    run_experiment(pruning_ratios, train_loader, test_loader)