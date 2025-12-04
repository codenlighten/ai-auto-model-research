import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a simple neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Function to train the model
def train_model(clip_value: float, steps: int = 10) -> dict:
    model = SimpleNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Dummy dataset
    x_data = torch.randn(100, 10)
    y_data = torch.randn(100, 1)

    losses = []
    gradient_norms_before = []
    gradient_norms_after = []

    for step in range(steps):
        optimizer.zero_grad()
        output = model(x_data)
        loss = loss_fn(output, y_data)
        losses.append(loss.item())

        # Backward pass
        loss.backward()
        # Log gradient norms before clipping
        for param in model.parameters():
            if param.grad is not None:
                gradient_norm = param.grad.data.norm(2)
                gradient_norms_before.append(gradient_norm.item())
                logging.info(f'Gradient norm before clipping: {gradient_norm.item()}')

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        # Log gradient norms after clipping
        for param in model.parameters():
            if param.grad is not None:
                gradient_norm = param.grad.data.norm(2)
                gradient_norms_after.append(gradient_norm.item())
                logging.info(f'Gradient norm after clipping: {gradient_norm.item()}')

        optimizer.step()

    return {
        'losses': losses,
        'gradient_norms_before': gradient_norms_before,
        'gradient_norms_after': gradient_norms_after
    }

# Main function to run the experiment
def main():
    results = {}
    clip_values = [0.0, 0.1, 0.5, 1.0]  # Control group (0.0) and experimental groups

    for clip_value in clip_values:
        logging.info(f'Training with gradient clipping value: {clip_value}')
        try:
            results[clip_value] = train_model(clip_value)
        except Exception as e:
            logging.error(f'Error during training with clip value {clip_value}: {str(e)}')

    # Save results to JSON file
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    logging.info('Results saved to experiment_results.json')

if __name__ == '__main__':
    main()