"""
Real ML Experiment Runner with Energy Tracking
Executes actual training experiments with comprehensive metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime

try:
    from codecarbon import EmissionsTracker
    CARBON_AVAILABLE = True
except ImportError:
    CARBON_AVAILABLE = False
    print("⚠️  codecarbon not installed - energy tracking disabled")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not installed - memory tracking limited")


class ExperimentRunner:
    """Run real ML experiments with comprehensive tracking"""
    
    def __init__(self, experiment_name, output_dir):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = {}
        self.start_time = None
        self.end_time = None
        
    def track_energy(self):
        """Context manager for energy tracking"""
        if CARBON_AVAILABLE:
            return EmissionsTracker(
                project_name=self.experiment_name,
                output_dir=str(self.output_dir),
                log_level="error"
            )
        else:
            return DummyTracker()
    
    def get_model_size(self, model):
        """Calculate model size in MB"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def count_parameters(self, model):
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def measure_inference_time(self, model, sample_input, num_runs=100):
        """Measure average inference time in milliseconds"""
        model.eval()
        with torch.no_grad():
            # Warmup
            for _ in range(10):
                _ = model(sample_input)
            
            # Actual measurement
            start = time.perf_counter()
            for _ in range(num_runs):
                _ = model(sample_input)
            end = time.perf_counter()
        
        avg_time_ms = ((end - start) / num_runs) * 1000
        return avg_time_ms
    
    def get_memory_usage(self):
        """Get current memory usage in MB"""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            mem_info = process.memory_info()
            return mem_info.rss / 1024 / 1024
        return 0
    
    def save_results(self):
        """Save experiment results to JSON"""
        results_file = self.output_dir / f"{self.experiment_name}_results.json"
        
        self.metrics["experiment_name"] = self.experiment_name
        self.metrics["timestamp"] = datetime.now().isoformat()
        self.metrics["duration_seconds"] = (
            (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        )
        
        with open(results_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        return results_file


class DummyTracker:
    """Dummy tracker when codecarbon not available"""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def stop(self):
        pass


# Example experiment templates
def train_baseline_model(dataset_name="mnist", epochs=5):
    """Train a baseline model on standard dataset"""
    
    experiment = ExperimentRunner(
        experiment_name=f"baseline_{dataset_name}",
        output_dir="results/baselines"
    )
    
    print(f"\n🔬 Training Baseline Model on {dataset_name}")
    print("=" * 80)
    
    experiment.start_time = time.time()
    
    # Simple 2-layer MLP for MNIST-like data
    class SimpleMLP(nn.Module):
        def __init__(self, input_size=784, hidden_size=128, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x
    
    model = SimpleMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Track model stats
    experiment.metrics["parameter_count"] = experiment.count_parameters(model)
    experiment.metrics["model_size_mb"] = experiment.get_model_size(model)
    
    print(f"📊 Model: {experiment.metrics['parameter_count']:,} parameters")
    print(f"📦 Size: {experiment.metrics['model_size_mb']:.2f} MB")
    
    # Generate synthetic data for quick testing
    batch_size = 64
    num_batches = 100
    
    print(f"\n🏋️  Training for {epochs} epochs...")
    
    with experiment.track_energy() as tracker:
        training_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx in range(num_batches):
                # Synthetic data
                inputs = torch.randn(batch_size, 784)
                labels = torch.randint(0, 10, (batch_size,))
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / num_batches
            training_losses.append(avg_loss)
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    experiment.metrics["training_losses"] = training_losses
    experiment.metrics["final_loss"] = training_losses[-1]
    
    # Measure inference time
    sample_input = torch.randn(1, 784)
    inference_time = experiment.measure_inference_time(model, sample_input)
    experiment.metrics["inference_time_ms"] = inference_time
    
    print(f"\n⚡ Inference: {inference_time:.4f} ms")
    
    # Memory usage
    memory_mb = experiment.get_memory_usage()
    experiment.metrics["peak_memory_mb"] = memory_mb
    print(f"💾 Memory: {memory_mb:.2f} MB")
    
    experiment.end_time = time.time()
    
    # Save results
    results_file = experiment.save_results()
    
    print("\n✅ Baseline training complete!")
    print("=" * 80)
    
    return experiment.metrics


def test_quantization_experiment():
    """Test INT8 quantization vs FP32"""
    
    print("\n🔬 Quantization Experiment: INT8 vs FP32")
    print("=" * 80)
    
    experiment = ExperimentRunner(
        experiment_name="quantization_int8_vs_fp32",
        output_dir="results/quantization"
    )
    
    experiment.start_time = time.time()
    
    # Create simple model
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 50)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(50, 10)
        
        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))
    
    # FP32 model
    model_fp32 = TinyModel()
    
    # Quantized model (dynamic quantization)
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {nn.Linear}, dtype=torch.qint8
    )
    
    # Measure sizes
    fp32_size = experiment.get_model_size(model_fp32)
    int8_size = experiment.get_model_size(model_int8)
    
    print(f"\n📦 FP32 Model: {fp32_size:.4f} MB")
    print(f"📦 INT8 Model: {int8_size:.4f} MB")
    print(f"🗜️  Compression: {fp32_size/int8_size:.2f}x")
    
    # Measure inference times
    sample_input = torch.randn(1, 100)
    
    fp32_time = experiment.measure_inference_time(model_fp32, sample_input)
    int8_time = experiment.measure_inference_time(model_int8, sample_input)
    
    print(f"\n⚡ FP32 Inference: {fp32_time:.4f} ms")
    print(f"⚡ INT8 Inference: {int8_time:.4f} ms")
    print(f"🚀 Speedup: {fp32_time/int8_time:.2f}x")
    
    experiment.metrics.update({
        "fp32_size_mb": fp32_size,
        "int8_size_mb": int8_size,
        "compression_ratio": fp32_size / int8_size,
        "fp32_inference_ms": fp32_time,
        "int8_inference_ms": int8_time,
        "speedup": fp32_time / int8_time
    })
    
    experiment.end_time = time.time()
    results_file = experiment.save_results()
    
    print("\n✅ Quantization experiment complete!")
    print("=" * 80)
    
    return experiment.metrics


if __name__ == "__main__":
    print("🧬 Real ML Experiment Runner")
    print("=" * 80)
    
    # Run baseline
    baseline_results = train_baseline_model(epochs=3)
    
    print("\n\n")
    
    # Run quantization experiment
    quant_results = test_quantization_experiment()
    
    print("\n\n📊 SUMMARY")
    print("=" * 80)
    print(f"✅ Baseline training: {baseline_results['final_loss']:.4f} final loss")
    print(f"✅ Quantization: {quant_results['speedup']:.2f}x speedup, {quant_results['compression_ratio']:.2f}x compression")
