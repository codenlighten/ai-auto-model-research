"""
FOCUSED HIGH-QUALITY EXPERIMENTS
Simpler experiments designed for 7+/10 breakthrough potential
"""

EXPERIMENTS = {
    "pruning_analysis": {
        "name": "Pruning Analysis",
        "description": """Comprehensive pruning study with fine-grained analysis:

Test pruning ratios: 10%, 20%, 30%, 40%, 50%, 60%, 70%
Train baseline SimpleCNN for 10 epochs on 2000 samples
For each pruning ratio:
  - Apply magnitude pruning
  - Fine-tune for 5 epochs
  - Measure accuracy, size, inference time

Discover:
1. Optimal pruning ratio (best accuracy/size tradeoff)
2. Critical pruning threshold (where accuracy drops >10%)
3. Inference speedup curve
4. Model compression ratio

Output: Complete comparison table + visualization-ready data + insights on pruning sweet spot.
Code must be clean, well-documented, and produce publication-quality results."""
    },
    
    "learning_rate_discovery": {
        "name": "Learning Rate Discovery",
        "description": """Systematic learning rate optimization study:

Test learning rates: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
Train SimpleMLP on 2000 samples for 15 epochs each
Track for each LR:
  - Final accuracy
  - Training stability (loss variance)
  - Convergence speed (epochs to 90% of final performance)
  - Overfitting tendency

Discover:
1. Optimal LR for fast convergence
2. LR ranges that cause instability
3. Sweet spot for accuracy vs speed
4. Relationship between LR and generalization

Output: Detailed analysis with convergence curves, stability metrics, and actionable recommendations.
Code must include proper validation split and statistical analysis."""
    },
    
    "architecture_depth_study": {
        "name": "Architecture Depth Study", 
        "description": """Study impact of network depth on performance:

Create MLPs with depths: 2, 3, 4, 5, 6, 7, 8 layers
Keep total parameters ~constant (~130K) by adjusting width
Train each for 12 epochs on 2000 samples

Measure:
1. Training time per epoch
2. Final validation accuracy  
3. Overfitting gap (train vs val accuracy)
4. Inference latency
5. Gradient flow (early vs late layers)

Discover:
1. Optimal depth for this problem
2. Point of diminishing returns
3. Depth vs overfitting relationship
4. Depth vs training efficiency

Output: Complete analysis showing depth sweet spot with statistical significance testing.
Code must implement all depths cleanly using ml_utils patterns."""
    },
    
    "batch_size_efficiency": {
        "name": "Batch Size Efficiency",
        "description": """Optimize batch size for training efficiency:

Test batch sizes: [8, 16, 32, 64, 128, 256]
Train SimpleMLP (3 layers, 512 hidden) for 1000 total updates each
Adjust epochs to keep total updates constant

Track:
1. Wall-clock training time
2. Final accuracy
3. Memory usage (model size + batch)
4. Convergence stability
5. Throughput (samples/second)

Discover:
1. Optimal batch size for speed
2. Batch size impact on accuracy
3. Memory/performance tradeoff
4. Sweet spot for efficiency

Output: Efficiency curve showing optimal batch size, with time/accuracy tradeoff analysis.
Code must be properly profiled and include memory measurements."""
    },
    
    "regularization_comparison": {
        "name": "Regularization Comparison",
        "description": """Compare regularization techniques on small datasets:

Techniques to test:
1. No regularization (baseline)
2. Dropout (rates: 0.1, 0.2, 0.3, 0.5)
3. L2 weight decay (strengths: 1e-5, 1e-4, 1e-3, 1e-2)
4. Combined (Dropout 0.2 + L2 1e-4)

Train SimpleMLP on 1000 samples for 20 epochs
Hold out 500 samples for validation

Measure:
1. Train vs val accuracy gap
2. Final validation accuracy
3. Robustness to small dataset
4. Training stability

Discover:
1. Best regularization for small data
2. Optimal dropout rate
3. Optimal L2 strength
4. Synergy between techniques

Output: Detailed overfitting analysis with recommendations for small-data scenarios.
Code must properly implement each regularization type."""
    }
}


def print_experiments():
    """Print all experiments"""
    print("\n" + "="*80)
    print("🎯 FOCUSED HIGH-QUALITY EXPERIMENTS")
    print("="*80)
    print("Designed for 7+/10 breakthrough potential\n")
    
    for i, (key, exp) in enumerate(EXPERIMENTS.items(), 1):
        print(f"\n{i}. {exp['name'].upper()}")
        print(f"   Key: {key}")
        print("-" * 80)
        print(exp['description'])
    
    print("\n" + "="*80)
    print("💡 Run with: python premier_research_lab.py <experiment_key>")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_experiments()
