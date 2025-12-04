import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ml_utils import create_synthetic_mnist

# Function to perform pruning on a model

def prune_model(model: nn.Module, pruning_ratio: float) -> nn.Module:
    """
    Prunes the model by the specified pruning ratio.
    Args:
        model (nn.Module): The model to prune.
        pruning_ratio (float): The ratio of weights to prune (0 to 1).
    Returns:
        nn.Module: The pruned model.
    Raises:
        ValueError: If pruning_ratio is not between 0 and 1.
    """
    if not (0 <= pruning_ratio <= 1):
        raise ValueError('Pruning ratio must be between 0 and 1.')

    # Implementing a simple weight pruning
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Calculate the number of weights to prune
            num_weights = param.numel()
            num_prune = int(num_weights * pruning_ratio)
            # Get the threshold value for pruning
            threshold = torch.topk(param.abs().view(-1), num_prune, largest=False).values[-1]
            # Prune the weights
            param.data[torch.abs(param.data) < threshold] = 0
    return model

# Function to run the experiment with different pruning ratios

def run_experiment(num_samples: int, batch_size: int, pruning_ratios: list) -> None:
    """
    Runs the pruning experiment with the specified parameters.
    Args:
        num_samples (int): Number of samples to generate.
        batch_size (int): Batch size for DataLoader.
        pruning_ratios (list): List of pruning ratios to test.
    """
    # Create synthetic MNIST dataset
    train_loader, test_loader, _, _ = create_synthetic_mnist(num_samples=num_samples, batch_size=batch_size)

    # Initialize a simple model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for ratio in pruning_ratios:
        print(f'Running experiment with pruning ratio: {ratio}')
        # Train the model
        for epoch in range(5):  # Small number of epochs for demonstration
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images.float())
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()
        # Prune the model
        pruned_model = prune_model(model, ratio)
        # Evaluate the pruned model
        # ... (Evaluation code would go here)

# Example usage
if __name__ == '__main__':
    run_experiment(num_samples=1000, batch_size=32, pruning_ratios=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75])
