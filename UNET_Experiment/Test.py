import os
import random
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from model import UNet
from SegmentationDataset import SegmentationDataset


# Helper function to compute the Dice score
def dice_score(preds, targets, epsilon=1e-8):
    # Flatten the tensors for each sample
    preds_flat = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    intersection = (preds_flat * targets_flat).sum(dim=1)
    dice = (2.0 * intersection + epsilon) / (preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + epsilon)
    return dice.mean().item()


# Evaluation function that computes average loss, dice score, and pixel-wise accuracy
def evaluate(model, data_loader, criterion, device):
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * images.size(0)

            # Compute predictions using sigmoid and thresholding
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            # Compute dice score for this batch
            batch_dice = dice_score(preds, masks)
            total_dice += batch_dice * images.size(0)

            # Compute pixel-wise accuracy
            correct = (preds == masks).sum().item()
            total_correct += correct
            total_pixels += masks.numel()

    avg_loss = total_loss / len(data_loader.dataset)
    avg_dice = total_dice / len(data_loader.dataset)
    pixel_accuracy = total_correct / total_pixels
    return avg_loss, avg_dice, pixel_accuracy


def visualize_samples(model, dataset, device, num_samples=5, threshold=0.5):
    model.to(device)
    model.eval()
    # Randomly sample indices from the dataset
    indices = random.sample(range(len(dataset)), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]  # Ensure axes is a list if only one sample is plotted

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            # Add a batch dimension and move to device
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            pred = torch.sigmoid(output)
            pred = (pred > threshold).float()
            # Remove batch dimension and move to CPU
            pred_img = pred.squeeze().cpu().numpy()
            image_np = image.squeeze().cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()

            # Plotting
            axes[i][0].imshow(image_np, cmap='gray')
            axes[i][0].set_title("Input Image")
            axes[i][0].axis("off")

            axes[i][1].imshow(mask_np, cmap='gray')
            axes[i][1].set_title("Ground Truth Mask")
            axes[i][1].axis("off")

            axes[i][2].imshow(pred_img, cmap='gray')
            axes[i][2].set_title("Predicted Mask")
            axes[i][2].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Get the directory of the current script and compute the repository root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))

    # Define the paths for test data (using "Test" instead of "Train")
    img_path = os.path.join(repo_root, "data", "COVIDQU", "Infection Segmentation Data", "Test", "images")
    mask_path = os.path.join(repo_root, "data", "COVIDQU", "Infection Segmentation Data", "Test", "infection masks")

    # Define image transformations
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Create the test dataset and DataLoader
    test_dataset = SegmentationDataset(transform=transform, images_dir=img_path, masks_dir=mask_path)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    print("Test data loaded.")

    # Instantiate the global model and load the checkpoint
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint_path = "Centralized_UNet.pth"  # Update path if needed
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Loaded global model checkpoint.")

    # Define the loss criterion (using BCEWithLogitsLoss for segmentation)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Evaluate the model on the test dataset
    test_loss, dice, pixel_acc = evaluate(model, test_loader, criterion, device=device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Dice Score: {dice:.4f}")
    print(f"Pixel Accuracy: {pixel_acc:.2%}")

    # Visualize 5 random samples from the test dataset along with predictions
    visualize_samples(model, test_dataset, device, num_samples=5, threshold=0.5)


if __name__ == '__main__':
    main()
#ChatGPT Test function