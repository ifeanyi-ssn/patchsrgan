import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.instnorm1 = nn.InstanceNorm3d(32)  # Separate instance norms
        self.prelu = nn.PReLU()

        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.instnorm2 = nn.InstanceNorm3d(32)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.instnorm1(residual)
        residual = self.prelu(residual)

        residual = self.conv2(residual)
        residual = self.instnorm2(residual)

        return x + residual  # Skip connection

class UpsampleBlock(nn.Module):
    def __init__(self, n_features=32):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv3d(n_features, n_features, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        #self.upsample = nn.ConvTranspose3d(n_features, n_features, kernel_size=4, stride=2, padding=1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        x = self.prelu(x)
        return x

class srgenerator(nn.Module):
    def __init__(self, n_resblocks=16):
        super(srgenerator, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()

        self.resblocks = nn.Sequential(*[ResidualBlock() for _ in range(n_resblocks)])

        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.instnorm = nn.InstanceNorm3d(32)

        self.upsample1 = UpsampleBlock()
        self.upsample2 = UpsampleBlock()
        self.upsample3 = UpsampleBlock()

        self.conv3 = nn.Conv3d(32, 1, kernel_size=9, padding=4)  # Fixed padding
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.prelu(x1)

        x2 = self.resblocks(x1)

        x2 = self.conv2(x2)
        x2 = self.instnorm(x2)
        x3 = x1 + x2  

        x3 = self.upsample1(x3)
        x3 = self.upsample2(x3)
        x4 = self.upsample3(x3)  

        x4 = self.conv3(x4)  # Fixed final layer input
        gen_sr = self.activation(x4)

        return gen_sr