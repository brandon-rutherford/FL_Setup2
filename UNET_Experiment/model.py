import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Encoder path
        # Each block is two conv layers and two relus
        self.enc_conv1 = self.conv_block(in_channels, 64)
        self.enc_conv2 = self.conv_block(64, 128)
        self.enc_conv3 = self.conv_block(128, 256)
        self.enc_conv4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder path
        self.dec_conv4 = self.upconv_block(1024 + 512, 512)
        self.dec_conv3 = self.upconv_block(512 + 256, 256)
        self.dec_conv2 = self.upconv_block(256 + 128, 128)

        # Final output layer (directly after dec_conv2)
        self.final_conv = nn.Conv2d(128 + 64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc_conv1(x)  # 256x256
        e2 = self.enc_conv2(F.max_pool2d(e1, 2))  # 128x128
        e3 = self.enc_conv3(F.max_pool2d(e2, 2))  # 64x64
        e4 = self.enc_conv4(F.max_pool2d(e3, 2))  # 32x32

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))  # 16x16

        # Decoder
        d4 = self.dec_conv4(torch.cat([F.interpolate(b, size=e4.shape[2:], mode='bilinear', align_corners=True), e4], dim=1))  # 32x32
        d3 = self.dec_conv3(torch.cat([F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=True), e3], dim=1))  # 64x64
        d2 = self.dec_conv2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True), e2], dim=1))  # 128x128

        # Final output!!
        d1 = torch.cat([F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True), e1], dim=1)  # 256x256
        return self.final_conv(d1)  # 256x256 output
