import torch
import numpy as np
import time
import json
from ml_utils import (
    create_synthetic_mnist,      # Returns DataLoader with correct types
    get_model_size,               # Calculate model parameters and size
    measure_inference_time,       # Measure inference speed
    train_simple_classifier,      # Standard training loop with error handling
    evaluate_accuracy,            # Evaluate model accuracy
    save_experiment_results,      # Save metrics to JSON
    SimpleMLP,                    # Simple 2-layer MLP (784→128→10)
    DeepMLP                       # 4-layer MLP (784→512→256→128→10)
)

# Define pruning ratios to test
pruning_ratios = np.arange(0.05, 0.76, 0.1)

# Initialize results storage
results = []

# Create synthetic MNIST data
train_loader, test_loader = create_synthetic_mnist(num_samples=1000, batch_size=32)

# Define model
model = SimpleMLP().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Function to prune model weights

def prune_model(model, pruning_ratio):
    """Prune the model weights based on the specified pruning ratio."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Calculate the threshold for pruning
            threshold = torch.quantile(param.abs(), pruning_ratio)
            # Prune weights below the threshold
            param.data[torch.abs(param.data) < threshold] = 0

# Function to fine-tune the model

def fine_tune_model(model, train_loader, num_epochs=5):
    """Fine-tune the model on the training dataset."""
    return train_simple_classifier(model, train_loader, num_epochs=num_epochs)

# Function to evaluate the model

def evaluate_model(model, test_loader):
    """Evaluate the model on the test dataset and return accuracy and inference time."""
    model.eval()
    with torch.no_grad():
        accuracy = evaluate_accuracy(model, test_loader)
        inference_time = measure_inference_time(model, test_loader)
    return accuracy, inference_time

# Iterate over each pruning ratio
for ratio in pruning_ratios:
    # Prune the model
    prune_model(model, ratio)
    # Fine-tune the model
    metrics = fine_tune_model(model, train_loader)
    # Evaluate the model
    accuracy, inference_time = evaluate_model(model, test_loader)
    # Get model size
    model_size = get_model_size(model)
    # Store results
    results.append({
        'pruning_ratio': ratio,
        'accuracy': accuracy,
        'model_size': model_size,
        'inference_time': inference_time,
        'fine_tuning_time': metrics['training_time'],
    })
    print(f'Pruning Ratio: {ratio:.2f}, Accuracy: {accuracy:.4f}, Model Size: {model_size}, Inference Time: {inference_time:.4f}')

# Save results to JSON
save_experiment_results(results, 'pruning_experiment_results.json')

# Print final results
print('Experiment completed. Results saved to pruning_experiment_results.json.')