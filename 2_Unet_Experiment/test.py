import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

# Import your model and dataset (make sure these files are in your PYTHONPATH)
from model import UNet
from SegmentationDataset import SegmentationDataset
from StartConfig import BATCH_SIZE  # Assumes BATCH_SIZE is defined in StartConfig.py


def load_test_data():
    """
    Loads the test dataset with appropriate image and mask transforms.
    Adjust paths according to your repository structure.
    """
    image_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Get the current script directory and then the repository root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))

    dataset_type = "Test"
    print(f"Loading {dataset_type} dataset...")

    img_path = os.path.join(repo_root, "data", "COVIDQU", "Infection Segmentation Data", dataset_type, "images")
    mask_path = os.path.join(repo_root, "data", "COVIDQU", "Infection Segmentation Data", dataset_type,
                             "infection masks")

    dataset = SegmentationDataset(
        images_dir=img_path,
        masks_dir=mask_path,
        transform_img=image_transform,
        transform_mask=mask_transform
    )

    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader


def dice_score(preds, targets, epsilon=1e-8):
    """
    Computes the Dice coefficient between predictions and targets.
    """
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    dice = (2.0 * intersection + epsilon) / (preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + epsilon)
    return dice.mean().item()


def evaluate_model(model, test_loader, criterion, device="cpu", threshold=0.5):
    """
    Evaluates the model on the test set and returns the average loss, dice score, and pixel accuracy.
    """
    model.eval()
    total_loss = 0.0
    dice_total = 0.0
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

            total_correct_pixels += (preds == targets).sum().item()
            total_pixels += targets.numel()

    avg_loss = total_loss / len(test_loader)
    avg_dice = dice_total / len(test_loader)
    pixel_accuracy = total_correct_pixels / total_pixels if total_pixels > 0 else 0
    return avg_loss, avg_dice, pixel_accuracy


def visualize_predictions(model, test_loader, device, num_samples=3, threshold=0.5):
    """
    Visualizes a few predictions.
    For each sample, shows the input image, the ground truth mask, and the predicted mask.
    """
    model.eval()
    # Get one batch from the test loader
    images, targets = next(iter(test_loader))
    images, targets = images.to(device), targets.to(device)

    with torch.no_grad():
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        preds = (preds > threshold).float()

    # Determine how many samples to display (if the batch is smaller than requested)
    num_samples = min(num_samples, images.size(0))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    # If only one sample, make axes iterable
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        img = images[i].cpu().squeeze().numpy()
        target = targets[i].cpu().squeeze().numpy()
        pred = preds[i].cpu().squeeze().numpy()

        axes[i][0].imshow(img, cmap="gray")
        axes[i][0].set_title("Input Image")
        axes[i][0].axis("off")

        axes[i][1].imshow(target, cmap="gray")
        axes[i][1].set_title("Ground Truth")
        axes[i][1].axis("off")

        axes[i][2].imshow(pred, cmap="gray")
        axes[i][2].set_title("Prediction")
        axes[i][2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model and load saved weights
    model = UNet(in_channels=1, out_channels=1).to(device)
    model_path = "10,2,0,4,1,0.5,25,Global.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Model file {model_path} not found. Exiting.")
        exit(1)

    # Load the test data
    test_loader = load_test_data()

    # Evaluate the model
    criterion = nn.BCEWithLogitsLoss()
    avg_loss, avg_dice, pixel_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {avg_loss:.4f}, Dice Score: {avg_dice:.4f}, Pixel Accuracy: {pixel_acc:.2%}")

    # Visualize predictions for a few test samples
    visualize_predictions(model, test_loader, device, num_samples=3)
