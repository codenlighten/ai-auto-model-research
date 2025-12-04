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

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create training data
train_loader = create_synthetic_mnist(num_samples=1000, batch_size=32)

# Create model and move to device
model = SimpleMLP().to(device)

# Initialize metrics dictionary
metrics = {
    'training_time': 0,
    'final_loss': None,
    'final_accuracy': None,
    'model_size': None,
    'inference_time': None
}

# Start training
start_time = time.time()
try:
    metrics = train_simple_classifier(model, train_loader, num_epochs=5)
except Exception as e:
    print(f'Error during training: {e}')  # Handle training errors
    metrics['final_loss'] = float('nan')
    metrics['final_accuracy'] = float('nan')

# Analyze model size
metrics['model_size'] = get_model_size(model)

# Measure inference time
with torch.no_grad():
    inference_start = time.time()
    # Run inference on a batch to measure time
    for inputs, labels in train_loader:
        inputs = inputs.float().to(device)  # Ensure inputs are float
        labels = labels.long().to(device)    # Ensure labels are long
        outputs = model(inputs)
        break  # Only need one batch for timing
    metrics['inference_time'] = measure_inference_time(model, inputs)

# Calculate total training time
metrics['training_time'] = time.time() - start_time

# Evaluate accuracy on validation set (not implemented in this snippet)
# metrics['final_accuracy'] = evaluate_accuracy(model, validation_loader)

# Save results to JSON
save_experiment_results('experiment_results.json', metrics)

# Print metrics
print(f'Model size: {metrics['model_size']} parameters')
print(f'Training time: {metrics['training_time']} seconds')
print(f'Final loss: {metrics['final_loss']}')
print(f'Final accuracy: {metrics['final_accuracy']}')
print(f'Inference time: {metrics['inference_time']} seconds')