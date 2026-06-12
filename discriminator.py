import torch
import torch.nn as nn

class D32x3(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1)
        self.instnorm2 = nn.InstanceNorm3d(64, affine=True)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.instnorm3 = nn.InstanceNorm3d(128, affine=True)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        self.instnorm4 = nn.InstanceNorm3d(256, affine=True)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.instnorm5 = nn.InstanceNorm3d(256, affine=True)
        self.lrelu5 = nn.LeakyReLU(0.2, inplace=True)

        self.conv6 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.instnorm6 = nn.InstanceNorm3d(256, affine=True)
        self.lrelu6 = nn.LeakyReLU(0.2, inplace=True)

        self.conv7 = nn.Conv3d(256, 1, kernel_size=3, stride=1, padding=1)
        # self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.instnorm2(self.conv2(x)))
        x = self.lrelu3(self.instnorm3(self.conv3(x)))
        x = self.lrelu4(self.instnorm4(self.conv4(x)))
        x = self.lrelu5(self.instnorm5(self.conv5(x)))
        x = self.lrelu6(self.instnorm6(self.conv6(x)))
        x = self.conv7(x)
        # return self.sigmoid(x)
        return x