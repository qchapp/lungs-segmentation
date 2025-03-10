import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        # x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        # x = self.bn(x)
        return self.relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block = Block(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        return self.block(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = Block(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.block(x)
    
class OutConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        # print(x.min(), x.max())
        return self.sigmoid(x)

class UNet(nn.Module):
    def __init__(self, BOTTLENECK, n_channels, n_class):
        super().__init__()

        self.block = Block(n_channels, BOTTLENECK//16)
        self.enc1 = Encoder(BOTTLENECK//16, BOTTLENECK//8)
        self.enc2 = Encoder(BOTTLENECK//8, BOTTLENECK//4)
        self.enc3 = Encoder(BOTTLENECK//4, BOTTLENECK//2)
        self.enc4 = Encoder(BOTTLENECK//2, BOTTLENECK)
        self.dec1 = Decoder(BOTTLENECK, BOTTLENECK//2)
        self.dec2 = Decoder(BOTTLENECK//2, BOTTLENECK//4)
        self.dec3 = Decoder(BOTTLENECK//4, BOTTLENECK//8)
        self.dec4 = Decoder(BOTTLENECK//8, BOTTLENECK//16)
        self.out = OutConvolution(BOTTLENECK//16, n_class)

    def forward(self, x):
        x1 = self.block(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        x = self.dec1(x5, x4)
        x = self.dec2(x, x3)
        x = self.dec3(x, x2)
        x = self.dec4(x, x1)
        return self.out(x)