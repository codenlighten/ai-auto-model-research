import numpy as np
import torch
import json
import logging
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a simple transformer model
class SimpleTransformer(nn.Module):
    def __init__(self):
        super(SimpleTransformer, self).__init__()
        self.fc = nn.Linear(10, 1)  # Simple linear layer for demonstration

    def forward(self, x):
        return self.fc(x)

# Learning rate scheduler with warmup
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = (self.current_step / self.warmup_steps) * self.optimizer.param_groups[0]['lr']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        elif self.current_step < self.total_steps:
            lr = (1 - (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)) * self.optimizer.param_groups[0]['lr']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

# Function to run the experiment
def run_experiment(warmup_steps: int) -> dict:
    try:
        # Create synthetic data
        x = np.random.rand(1000, 10).astype(np.float32)
        y = np.random.rand(1000, 1).astype(np.float32)
        dataset = TensorDataset(torch.tensor(x), torch.tensor(y))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Initialize model, loss function, and optimizer
        model = SimpleTransformer()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = WarmupScheduler(optimizer, warmup_steps, total_steps=100)

        # Training loop
        for epoch in range(10):  # Run for a few epochs
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                scheduler.step()

        # Log final loss
        logging.info(f'Warmup Steps: {warmup_steps}, Final Loss: {loss.item()}')
        return {'warmup_steps': warmup_steps, 'final_loss': loss.item()}
    except Exception as e:
        logging.error(f'Error during experiment: {e}')
        return {'warmup_steps': warmup_steps, 'error': str(e)}

# Main function to run experiments with different warmup periods
if __name__ == '__main__':
    results = []
    warmup_variants = [500, 1000, 2000]
    for warmup in warmup_variants:
        result = run_experiment(warmup)
        results.append(result)

    # Save results to JSON file
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    logging.info('Experiment results saved to experiment_results.json')
