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

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create data
train_loader = create_synthetic_mnist(num_samples=1000, batch_size=32)

# Create model
model = SimpleMLP().to(device)

# Initialize metrics dictionary
metrics = {'training_time': 0, 'final_loss': 0, 'final_accuracy': 0, 'model_size': 0, 'inference_time': 0}

# Train the model
start_time = time.time()
try:
    metrics = train_simple_classifier(model, train_loader, num_epochs=5)
except Exception as e:
    print(f"Error during training: {e}")
    exit(1)

# Analyze model size
metrics['model_size'] = get_model_size(model)

# Measure inference time
try:
    with torch.no_grad():
        inference_start = time.time()
        # Run inference on a batch
        for inputs, labels in train_loader:
            inputs = inputs.float().to(device)  # Ensure inputs are float
            labels = labels.long().to(device)    # Ensure labels are long
            outputs = model(inputs)
            break  # Only need to measure for one batch
        metrics['inference_time'] = measure_inference_time(model, inputs)
except Exception as e:
    print(f"Error during inference: {e}")
    exit(1)

# Calculate final accuracy
try:
    metrics['final_accuracy'] = evaluate_accuracy(model, train_loader)
except Exception as e:
    print(f"Error during accuracy evaluation: {e}")
    exit(1)

# Calculate training time
metrics['training_time'] = time.time() - start_time

# Save results to JSON
try:
    save_experiment_results(metrics, 'experiment_results.json')
except Exception as e:
    print(f"Error saving results: {e}")
    exit(1)

# Print metrics
print(f"Model size: {metrics['model_size']} parameters")
print(f"Training time: {metrics['training_time']} seconds")
print(f"Final accuracy: {metrics['final_accuracy']}%")
print(f"Inference time: {metrics['inference_time']} seconds")