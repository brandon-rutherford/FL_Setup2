import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Define a simple CNN model for MNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 14 * 14)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load dataset (used for both FL clients and central training)
def load_data(batch_size=32, train_split=0.8, client_id=0, num_clients=2):
    """Loads a unique partition of MNIST data for each client."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Split dataset into non-overlapping partitions
    num_samples_per_client = len(dataset) // num_clients
    start_idx = client_id * num_samples_per_client
    end_idx = (client_id + 1) * num_samples_per_client
    client_dataset = torch.utils.data.Subset(dataset, list(range(start_idx, end_idx)))

    train_size = int(train_split * len(client_dataset))
    test_size = len(client_dataset) - train_size
    train_dataset, test_dataset = random_split(client_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



# Train one epoch (useful for FL clients)
def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# Evaluate model (used for FL clients or centralized testing)
def evaluate(model, test_loader, criterion, device="cpu"):
    model.to(device)
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()

    accuracy = correct / len(test_loader.dataset)
    avg_loss = total_loss / len(test_loader)

    return avg_loss, accuracy  # Return loss and accuracy


# Run training for multiple epochs (external FL manager can call this)
import os

def run_training(model, train_loader, test_loader, num_epochs=10, lr=0.001, device="cpu", save_path="mnist_cnn.pth"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Accuracy={accuracy:.2%}")

    # Save trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")



# Main execution (No loops, just function calls)
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SimpleCNN()
    train_loader, test_loader = load_data()
    run_training(model, train_loader, test_loader, num_epochs=10, device=device)
