# Adapted from milesial's Pytorch-UNet
# https://github.com/milesial/Pytorch-UNet/blob/2f62e6b1c8e98022a6418d31a76f6abd800e5ae7/unet/unet_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class UNetBlock(nn.Module):
    def __init__(self, shape, min_size=2):
        assert len(shape)==3
        self.shape = shape
        print(shape)
        super().__init__()

        assert shape[1]==shape[2], "Expected a square image"
        self.down = Down(shape[0], 2*shape[0])
        down_shape = (2*shape[0], shape[1]//2, shape[2]//2)
        if down_shape[1]//2 < min_size:
            self.sub_unet = nn.Identity()
        else:
            self.sub_unet = UNetBlock(down_shape, min_size=min_size)
        self.up = Up(2*shape[0], shape[0], bilinear=False)

    def forward(self, x):
        y = self.down(x)
        y = self.sub_unet(y)
        y = self.up(y, x)
        return y


class UNet(nn.Module):
    def __init__(self, shape, in_channels, out_channels, min_size=2):
        assert len(shape)==3
        assert shape[-1]==shape[-2], "Expected a square image"
        super().__init__()
        self.shape = shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.dconv = DoubleConv(shape[0], in_channels)
        shape = (in_channels, *shape[1:])
        if shape[1]//2 < min_size:
            self.sub_unet = nn.Identity()
        else:
            self.sub_unet = UNetBlock(shape, min_size=min_size)
        self.outconv = OutConv(in_channels, 1)

    def forward(self, x):
        y = self.dconv(x)
        y = self.sub_unet(y)
        y = self.outconv(y)
        return y


if __name__=="__main__":
    shape = (100,10,32,32)
    v = torch.randn(*shape)

    m = UNet(shape[1:], 32, 1)
    # m = Down(shape[1], 2*shape[1])
    # m2 = Up(2*shape[1], shape[1], bilinear=False)
    y = m(v)
    # y1 = m2(y,v)

    print(y.shape)
    print(y)