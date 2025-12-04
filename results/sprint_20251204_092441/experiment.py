import torch
import time
import json
from ml_utils import create_synthetic_mnist, SimpleMLP, train_simple_classifier, get_model_size, measure_inference_time

# Set device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create data loader for synthetic MNIST dataset
train_loader = create_synthetic_mnist(num_samples=1000, batch_size=32)

# Create model and move it to the appropriate device
model = SimpleMLP().to(device)

# Initialize metrics dictionary to track performance
metrics = {
    'training_time': 0,
    'final_loss': None,
    'final_accuracy': None,
    'model_size': get_model_size(model),
    'inference_time': None
}

# Training function with error handling
try:
    # Start training and measure time
    start_time = time.time()
    metrics = train_simple_classifier(model, train_loader, num_epochs=5)
    metrics['training_time'] = time.time() - start_time

    # Measure inference time
    with torch.no_grad():
        inference_time = measure_inference_time(model, train_loader)
        metrics['inference_time'] = inference_time

    # Print metrics
    print(f"Model size: {metrics['model_size']} parameters")
    print(f"Training time: {metrics['training_time']:.2f}s")
    print(f"Final loss: {metrics['final_loss']}")
    print(f"Final accuracy: {metrics['final_accuracy']:.2f}%")
    print(f"Inference time: {metrics['inference_time']:.4f}s")

    # Save results to JSON
    with open('experiment_results.json', 'w') as f:
        json.dump(metrics, f)
        print('Results saved to experiment_results.json')

except Exception as e:
    print(f'An error occurred during training: {e}')