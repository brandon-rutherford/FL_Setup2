import torch.nn as nn
from segmentation_models_pytorch import Unet as SMPUnet

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # Use a ResNet34 encoder pre-trained on ImageNet.
        self.unet = SMPUnet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=out_channels
        )

    def forward(self, x):
        return self.unet(x)
