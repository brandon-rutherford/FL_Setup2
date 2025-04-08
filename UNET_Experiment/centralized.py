import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from SegmentationDataset import SegmentationDataset
from model import UNet


def dice_score(preds, targets, epsilon=1e-8):
    # Flatten the tensors to compute the Dice coefficient per sample
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    dice = (2.0 * intersection + epsilon) / (preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + epsilon)
    return dice.mean().item()


# Load dataset (used for both FL clients and central training)
def load_data(batch_size=4, train_split=0.8, client_id=0, num_clients=1):
    """Loads a unique partition of data for each client."""
    image_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # Optional: transforms.Normalize([0.5], [0.5]) etc.
    ])

    # For the masks (already single-channel 0/255 or 0/1)
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # If needed, threshold to ensure 0/1
        # transforms.Lambda(lambda x: (x > 0.5).float())
    ])

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Build the path to the root of your repository (one level up from "Experiment")
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))
    # Build the full paths to the image and mask directories within your data folder
    img_path = os.path.join(repo_root, "data", "COVIDQU", "Infection Segmentation Data", "Train", "images")
    mask_path = os.path.join(repo_root, "data", "COVIDQU", "Infection Segmentation Data", "Train", "infection masks")

    dataset = SegmentationDataset(
        images_dir=img_path,
        masks_dir=mask_path,
        transform_img=image_transform,
        transform_mask=mask_transform
    )

    image, mask = dataset[0]
    print("Image tensor range:", image.min().item(), image.max().item())
    print("Mask tensor unique values:", torch.unique(mask))

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
def evaluate(model, test_loader, criterion, device="cpu", threshold=0.5):
    model.to(device)
    model.eval()
    total_loss = 0
    dice_total = 0

    total_correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Compute predictions using thresholding
            preds = torch.sigmoid(outputs)
            # Debug prints to inspect values before thresholding
            # print("Raw preds mean:", preds.mean().item(), "Targets mean:", targets.float().mean().item())
            preds = (preds > threshold).float()
            # print("Binarized preds mean:", preds.mean().item())

            # Compute dice score for this batch
            dice_total += dice_score(preds, targets)

            # Compute pixel-wise accuracy for this batch
            correct_pixels = (preds == targets).sum().item()
            batch_total_pixels = targets.numel()
            total_correct_pixels += correct_pixels
            total_pixels += batch_total_pixels

    avg_loss = total_loss / len(test_loader)
    avg_dice = dice_total / len(test_loader)
    pixel_accuracy = total_correct_pixels / total_pixels if total_pixels > 0 else 0
    return avg_loss, avg_dice, pixel_accuracy


# Run training for multiple epochs (external FL manager can call this)
def run_training(model, train_loader, test_loader, num_epochs=10, lr=0.001, device="cpu", save_path="Covid_UNet.pth"):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, dice, pixel_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Dice={dice:.4f}, Pixel Accuracy={pixel_acc:.2%}")

    # Save trained model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


# Main execution (No loops, just function calls)
from torchsummary import summary
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = UNet(in_channels=1, out_channels=1).to(device)
    # Print the summary of the model
    summary(model, input_size=(1, 256, 256))

    train_loader, test_loader = load_data()
    run_training(model, train_loader, test_loader, num_epochs=15, device=device, save_path="Centralized_UNet.pth")
