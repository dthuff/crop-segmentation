import torch
import torch.nn as nn
import math


class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class ResNetBlock(nn.Module):
    """
    ResNet block - two blocks of sequential conv, batchnorm, relu
    """

    def __init__(self, channels, kernel_size, stride=1, padding=1):
        super(ResNetBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out = self.conv(x) + x
        return out


class UpConvBlock(nn.Module):
    """
    UpConv block - conv with 1x1 kernel and upsample to recover spatial dims in decoder
    """

    def __init__(self, channels_in, channels_out, kernel_size=1, scale_factor=2, align_corners=False):
        super(UpConvBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=align_corners),
        )

    def forward(self, x):
        return self.up(x)


class Encoder(nn.Module):
    """
    Class for the Encoder (1st half of the VAE)
    4 blocks of conv, resnet, maxpool

    So, at bottleneck layer, input spatial dims are reduced by factor of 2^4
    """

    def __init__(self, channels):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(channels_in=channels, channels_out=32, kernel_size=3)
        self.res_block1 = ResNetBlock(channels=32, kernel_size=3)
        self.MaxPool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = ConvBlock(channels_in=32, channels_out=64, kernel_size=3)
        self.res_block2 = ResNetBlock(channels=64, kernel_size=3)
        self.MaxPool2 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv3 = ConvBlock(channels_in=64, channels_out=128, kernel_size=3)
        self.res_block3 = ResNetBlock(channels=128, kernel_size=3)
        self.MaxPool3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv4 = ConvBlock(channels_in=128, channels_out=256, kernel_size=3)
        self.res_block4 = ResNetBlock(channels=256, kernel_size=3)
        self.MaxPool4 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.res_block1(x1)
        x1 = self.MaxPool1(x1)

        x2 = self.conv2(x1)
        x2 = self.res_block2(x2)
        x2 = self.MaxPool2(x2)

        x3 = self.conv3(x2)
        x3 = self.res_block3(x3)
        x3 = self.MaxPool3(x3)

        x4 = self.conv4(x3)
        x4 = self.res_block4(x4)
        x4 = self.MaxPool4(x4)
        return x4  # shape 256, img_dim/16, img_dim/16


class Decoder(nn.Module):
    """
    Class for the decoder half of the VAE
    """
    def __init__(self, img_dim):
        super(Decoder, self).__init__()
        self.img_dim = img_dim

        self.upsize4 = UpConvBlock(channels_in=256, channels_out=128, kernel_size=1, scale_factor=2)
        self.res_block4 = ResNetBlock(channels=128, kernel_size=3)

        self.upsize3 = UpConvBlock(channels_in=128, channels_out=64, kernel_size=1, scale_factor=2)
        self.res_block3 = ResNetBlock(channels=64, kernel_size=3)

        self.upsize2 = UpConvBlock(channels_in=64, channels_out=32, kernel_size=1, scale_factor=2)
        self.res_block2 = ResNetBlock(channels=32, kernel_size=3)

        self.upsize1 = UpConvBlock(channels_in=32, channels_out=2, kernel_size=1, scale_factor=2)
        self.res_block1 = ResNetBlock(channels=2, kernel_size=3)  # We want 2 output channels

    def forward(self, x):
        # x4_ = x.view(-1, 256, int(self.img_dim / 16), int(self.img_dim / 16))
        x4_ = self.upsize4(x)
        x4_ = self.res_block4(x4_)

        x3_ = self.upsize3(x4_)
        x3_ = self.res_block3(x3_)

        x2_ = self.upsize2(x3_)
        x2_ = self.res_block2(x2_)

        x1_ = self.upsize1(x2_)
        x1_ = self.res_block1(x1_)

        return x1_


class UNet(nn.Module):
    """
    2D U-Net consists of encoder + decoder
    """
    def __init__(self, img_dim=128, channels=3):
        super(UNet, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.img_dim = img_dim
        self.encoder = Encoder(channels)
        self.decoder = Decoder(img_dim)
        self.xavier_init()

    def kaiming_init(self):
        for param in self.parameters():
            std = math.sqrt(2 / param.size(0))
            torch.nn.init.normal_(param, mean=0, std=std)

    def xavier_init(self):
        for param in self.parameters():
            std_dev = 1.0 / math.sqrt(param.size(0))
            torch.nn.init.uniform_(param, -std_dev, std_dev)

    def forward(self, x):
        x = self.encoder(x)
        y = self.decoder(x)
        return y
