"""Test new pruning utilities"""
import sys
import torch
from ml_utils import SimpleCNN, create_synthetic_mnist_images, prune_model, compare_models, print_comparison_table, train_simple_classifier

print("Testing new pruning utilities...\n")

# Create image data
print("1. Creating synthetic image data...")
train_loader = create_synthetic_mnist_images(num_samples=500, batch_size=32)
test_loader = create_synthetic_mnist_images(num_samples=100, batch_size=32)
print("   ✓ Image data loaders created (28x28 format)\n")

# Create baseline model
print("2. Creating and training baseline CNN...")
baseline_model = SimpleCNN(input_channels=1, num_classes=10)
metrics = train_simple_classifier(baseline_model, train_loader, num_epochs=2, print_every=10)
print(f"   ✓ Baseline trained in {metrics['training_time']}s\n")

# Create pruned models
print("3. Creating pruned variants...")
models = {'Baseline (0%)': baseline_model}

for ratio in [0.3, 0.5, 0.7]:
    print(f"\n   Pruning at {ratio:.0%}...")
    # Clone the baseline
    pruned_model = SimpleCNN(input_channels=1, num_classes=10)
    pruned_model.load_state_dict(baseline_model.state_dict())
    
    # Apply pruning
    prune_model(pruned_model, pruning_ratio=ratio)
    models[f'Pruned ({ratio:.0%})'] = pruned_model

print("\n4. Comparing all models...")
results = compare_models(models, test_loader)
print_comparison_table(results)

print("✅ All pruning utilities working correctly!")
