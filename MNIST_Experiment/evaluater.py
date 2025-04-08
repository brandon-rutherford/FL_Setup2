import torch
import matplotlib.pyplot as plt
import glob
import re
from centralized import SimpleCNN, load_data, evaluate

# Load test data
_, test_loader = load_data()


# Function to extract round number from filename
def extract_round_number(filename):
    match = re.search(r"global_model_round_(\d+)\.pth", filename)
    return int(match.group(1)) if match else None


# Get all model checkpoint files
model_files = sorted(glob.glob("global_model_round_*.pth"), key=extract_round_number)

# Prepare lists to store results
rounds = []
accuracies = []

# Load and evaluate each model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()

for model_file in model_files:
    round_num = extract_round_number(model_file)
    if round_num is None:
        continue

    # Load model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # Evaluate model
    test_loss, accuracy = evaluate(model, test_loader, criterion, device)

    # Store results
    rounds.append(round_num)
    accuracies.append(accuracy)

    print(f"Round {round_num}: Accuracy = {accuracy:.2%}")

# Plot accuracy vs. round number
plt.figure(figsize=(8, 5))
plt.plot(rounds, accuracies, marker="o", linestyle="-", label="Test Accuracy")
plt.xlabel("Federated Learning Round")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Over Rounds")
plt.legend()
plt.grid()
plt.show()
