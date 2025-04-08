import os

from PIL import Image
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform_img=None, transform_mask=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        # Get a list of all image file names (e.g. .png, .jpg) in images_dir
        self.image_names = os.listdir(images_dir)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # Build the full image path
        image_path = os.path.join(self.images_dir, self.image_names[idx])
        # Build the corresponding mask path
        mask_path = os.path.join(self.masks_dir, self.image_names[idx])

        # Open the image
        image = Image.open(image_path)
        if self.transform_img:
            image = self.transform_img(image)

        # Open the mask
        mask = Image.open(mask_path)
        if self.transform_mask:
            mask = self.transform_mask(mask)

        return image, mask
