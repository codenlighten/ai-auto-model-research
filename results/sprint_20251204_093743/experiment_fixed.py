import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the Teacher model
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))

# Define the Student model
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.layers(x.view(x.size(0), -1))

# Knowledge Distillation Loss Function
class DistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits):
        student_probs = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = nn.functional.softmax(teacher_logits / self.temperature, dim=1)
        return self.criterion(student_probs, teacher_probs) * (self.temperature ** 2)

# Training function
def train(teacher_model, student_model, train_loader, optimizer, loss_function, device):
    teacher_model.train()
    student_model.train()
    for data, _ in train_loader:
        data = data.to(device)

        # Forward pass through teacher and student models
        teacher_logits = teacher_model(data).to(device).float()
        student_logits = student_model(data).to(device).float()

        # Check for shape and type compatibility
        assert teacher_logits.shape == student_logits.shape, "Logits shape mismatch"
        assert teacher_logits.dtype == student_logits.dtype, "Logits dtype mismatch"

        # Calculate loss
        loss = loss_function(student_logits, teacher_logits)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Main function to set up the experiment
def main():
    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 10
    temperature = 2.0

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize models, optimizer, and loss function
    teacher_model = TeacherModel().to(device)
    student_model = StudentModel().to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    loss_function = DistillationLoss(temperature=temperature)

    # Train the models
    for epoch in range(num_epochs):
        train(teacher_model, student_model, train_loader, optimizer, loss_function, device)
        print(f'Epoch [{epoch + 1}/{num_epochs}] completed.')  

if __name__ == '__main__':
    main()