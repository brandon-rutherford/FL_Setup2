import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt

from SegmentationDataset import SegmentationDataset
from model import UNet


def dice_score(preds, targets, epsilon=1e-8):
    # Flatten the tensors to compute the Dice coefficient per sample
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    dice = (2.0 * intersection + epsilon) / (preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + epsilon)
    return dice.mean().item()


def load_data(batch_size=4, client_id=0, num_clients=1):
    image_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # Optionally add: transforms.Lambda(lambda x: (x > 0.5).float())
    ])

    # Get the directory of the current script and then the repo root
    #script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(("/content")))

    # Paths for training data
    train_img_path = os.path.join(repo_root, "data", "COVIDQU", "Infection Segmentation Data", "Train", "images")
    train_mask_path = os.path.join(repo_root, "data", "COVIDQU", "Infection Segmentation Data", "Train", "infection masks")

    # Paths for test data (note "Test" instead of "Train")
    test_img_path = os.path.join(repo_root, "data", "COVIDQU", "Infection Segmentation Data", "Test", "images")
    test_mask_path = os.path.join(repo_root, "data", "COVIDQU", "Infection Segmentation Data", "Test", "infection masks")

    # Create the full training dataset
    train_dataset_full = SegmentationDataset(
        images_dir=train_img_path,
        masks_dir=train_mask_path,
        transform_img=image_transform,
        transform_mask=mask_transform
    )

    # Create the test dataset
    test_dataset = SegmentationDataset(
        images_dir=test_img_path,
        masks_dir=test_mask_path,
        transform_img=image_transform,
        transform_mask=mask_transform
    )

    # Partition training dataset among clients if needed.
    num_samples_per_client = len(train_dataset_full) // num_clients
    start_idx = client_id * num_samples_per_client
    end_idx = (client_id + 1) * num_samples_per_client
    train_dataset = torch.utils.data.Subset(train_dataset_full, list(range(start_idx, end_idx)))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



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


def evaluate(model, test_loader, criterion, device="cpu", threshold=0.5):
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

            preds = torch.sigmoid(outputs)
            preds = (preds > threshold).float()
            dice_total += dice_score(preds, targets)

            correct_pixels = (preds == targets).sum().item()
            total_correct_pixels += correct_pixels
            total_pixels += targets.numel()

    avg_loss = total_loss / len(test_loader)
    avg_dice = dice_total / len(test_loader)
    pixel_accuracy = total_correct_pixels / total_pixels if total_pixels > 0 else 0
    return avg_loss, avg_dice, pixel_accuracy


def run_training(model, train_loader, test_loader, num_epochs=10, lr=0.001, device="cpu",
                 save_path="Centralized_UNet.pth"):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, dice, pixel_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}/{num_epochs}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Dice={dice:.4f}, Pixel Accuracy={pixel_acc:.2%}")

        # Visualize predictions non-blockingly at the end of each epoch.
        visualize_predictions(model, test_loader, device)

        # Save an intermediate checkpoint for this epoch.
        checkpoint_path = f"Centralized_UNet_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    torch.save(model.state_dict(), save_path)
    print(f"Final model saved to {save_path}")


def visualize_predictions(model, test_loader, device, threshold=0.5):
    model.eval()
    # Get one batch from the test loader.
    images, targets = next(iter(test_loader))
    images, targets = images.to(device), targets.to(device)
    with torch.no_grad():
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        preds = (preds > threshold).float()

    # Visualize the first image in the batch: image, ground truth, and prediction.
    img = images[0].cpu().squeeze().numpy()
    target = targets[0].cpu().squeeze().numpy()
    pred = preds[0].cpu().squeeze().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img, cmap="gray")
    axs[0].set_title("Image")
    axs[1].imshow(target, cmap="gray")
    axs[1].set_title("Ground Truth")
    axs[2].imshow(pred, cmap="gray")
    axs[2].set_title("Prediction")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)  # Display for 1 second without blocking.
    plt.close(fig)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = UNet(in_channels=1, out_channels=1).to(device)
    train_loader, test_loader = load_data()
    run_training(model, train_loader, test_loader, num_epochs=15, lr=0.001, device=device,
                 save_path="Centralized_UNet.pth")
