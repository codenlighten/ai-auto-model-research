"""
ML Utilities for AI Research Team
Provides common patterns and helper functions to prevent errors
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
import json


def create_synthetic_mnist(num_samples=1000, batch_size=32):
    """
    Create synthetic MNIST-like data with proper types
    Returns: DataLoader with correct tensor types
    """
    # Generate random image data (flattened 28x28)
    X = torch.randn(num_samples, 784).float()
    # Generate random labels (0-9) with correct type
    y = torch.randint(0, 10, (num_samples,)).long()
    
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def get_model_size(model):
    """
    Calculate model size in parameters and MB
    """
    param_count = sum(p.numel() for p in model.parameters())
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    return {
        'parameters': param_count,
        'size_mb': round(param_size_mb, 4)
    }


def measure_inference_time(model, input_shape=(1, 784), num_runs=100):
    """
    Measure average inference time
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    end = time.time()
    
    avg_time_ms = ((end - start) / num_runs) * 1000
    return round(avg_time_ms, 4)


def train_simple_classifier(model, train_loader, num_epochs=5, lr=0.001, print_every=50):
    """
    Standard training loop with proper error handling
    Returns: metrics dict
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    start_time = time.time()
    metrics = {
        'epoch_losses': [],
        'final_loss': None,
        'training_time': None
    }
    
    model.train()
    step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Ensure correct types
            data = data.float()
            target = target.long()
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            
            # Verify shapes
            assert output.shape[0] == target.shape[0], f"Batch size mismatch: {output.shape[0]} vs {target.shape[0]}"
            assert len(target.shape) == 1, f"Target should be 1D, got shape {target.shape}"
            
            # Loss and backward
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            step += 1
            
            # Print progress
            if step % print_every == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Step {step} | Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / num_batches
        metrics['epoch_losses'].append(avg_epoch_loss)
        print(f"Epoch {epoch+1} completed | Avg Loss: {avg_epoch_loss:.4f}")
    
    training_time = time.time() - start_time
    metrics['final_loss'] = metrics['epoch_losses'][-1]
    metrics['training_time'] = round(training_time, 2)
    
    return metrics


def evaluate_accuracy(model, test_loader):
    """
    Evaluate model accuracy
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.float()
            target = target.long()
            
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    accuracy = (correct / total) * 100 if total > 0 else 0
    return round(accuracy, 2)


def save_experiment_results(metrics, filepath='results.json'):
    """
    Save metrics to JSON with timestamp
    """
    import datetime
    metrics['timestamp'] = datetime.datetime.now().isoformat()
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n💾 Results saved to {filepath}")


class SimpleMLP(nn.Module):
    """
    Simple MLP for quick experiments
    """
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DeepMLP(nn.Module):
    """
    Deeper MLP for comparison experiments
    """
    def __init__(self, input_dim=784, output_dim=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)


class SimpleCNN(nn.Module):
    """
    Simple CNN for image classification experiments
    """
    def __init__(self, input_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)  # 14x14
        x = self.relu(self.conv2(x))
        x = self.pool(x)  # 7x7
        x = self.relu(self.conv3(x))
        x = self.pool(x)  # 3x3
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def create_synthetic_mnist_images(num_samples=1000, batch_size=32):
    """
    Create synthetic MNIST-like image data (28x28 format)
    Returns: DataLoader with images in (B, 1, 28, 28) format
    """
    # Generate random image data
    X = torch.randn(num_samples, 1, 28, 28).float()
    # Generate random labels (0-9)
    y = torch.randint(0, 10, (num_samples,)).long()
    
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def prune_model(model, pruning_ratio=0.3, method='magnitude'):
    """
    Prune model weights by removing smallest magnitude weights
    
    Args:
        model: PyTorch model to prune
        pruning_ratio: Fraction of weights to remove (0.0 - 1.0)
        method: 'magnitude' (only supported method currently)
    
    Returns:
        Pruned model (in-place operation)
    """
    if method != 'magnitude':
        raise ValueError("Only 'magnitude' pruning is currently supported")
    
    if not 0.0 <= pruning_ratio < 1.0:
        raise ValueError(f"Pruning ratio must be in [0.0, 1.0), got {pruning_ratio}")
    
    # Collect all weights
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            all_weights.append(param.data.abs().view(-1))
    
    if len(all_weights) == 0:
        print("⚠️  No prunable weights found")
        return model
    
    # Concatenate all weights
    all_weights_tensor = torch.cat(all_weights)
    
    # Find threshold (k-th smallest value)
    k = int(pruning_ratio * all_weights_tensor.numel())
    if k == 0:
        print("⚠️  Pruning ratio too small, no weights pruned")
        return model
    
    threshold = torch.kthvalue(all_weights_tensor, k).values.item()
    
    # Apply pruning mask
    pruned_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            mask = param.data.abs() > threshold
            param.data *= mask.float()
            
            pruned_count += (~mask).sum().item()
            total_count += mask.numel()
    
    actual_ratio = pruned_count / total_count if total_count > 0 else 0
    print(f"✂️  Pruned {pruned_count:,}/{total_count:,} weights ({actual_ratio:.1%})")
    
    return model


def compare_models(models_dict, test_loader):
    """
    Compare multiple models on same test set
    
    Args:
        models_dict: Dict of {name: model}
        test_loader: DataLoader for evaluation
    
    Returns:
        Dict with comparison metrics
    """
    results = {}
    
    for name, model in models_dict.items():
        # Accuracy
        accuracy = evaluate_accuracy(model, test_loader)
        
        # Model size
        size_info = get_model_size(model)
        
        # Inference time (use appropriate input shape)
        sample_input = next(iter(test_loader))[0][:1]
        input_shape = sample_input.shape
        inference_time = measure_inference_time(model, input_shape=input_shape)
        
        results[name] = {
            'accuracy': accuracy,
            'parameters': size_info['parameters'],
            'size_mb': size_info['size_mb'],
            'inference_ms': inference_time
        }
    
    return results


def print_comparison_table(results):
    """
    Pretty print model comparison results
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<25} {'Accuracy':<12} {'Params':<15} {'Size (MB)':<12} {'Inference (ms)'}")
    print("-"*80)
    
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['accuracy']:>6.2f}%     {metrics['parameters']:>10,}   "
              f"{metrics['size_mb']:>8.4f}     {metrics['inference_ms']:>8.4f}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Test the utilities
    print("Testing ML Utilities...")
    
    # Create data
    train_loader = create_synthetic_mnist(num_samples=500, batch_size=32)
    print("✓ Data loader created")
    
    # Create model
    model = SimpleMLP()
    print(f"✓ Model created: {get_model_size(model)}")
    
    # Train
    metrics = train_simple_classifier(model, train_loader, num_epochs=2, print_every=5)
    print(f"✓ Training completed: {metrics['training_time']}s")
    
    # Inference
    inference_time = measure_inference_time(model)
    print(f"✓ Inference time: {inference_time}ms")
    
    # Accuracy
    test_loader = create_synthetic_mnist(num_samples=100, batch_size=32)
    acc = evaluate_accuracy(model, test_loader)
    print(f"✓ Accuracy: {acc}%")
    
    print("\n✅ All utilities working correctly!")
