import torch
import torch.nn as nn
import torch.nn.functional as F


class InitialBlock(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 16) -> None:
        super().__init__()
        conv_channels = out_channels - in_channels
        self.conv = nn.Conv2d(in_channels, conv_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.conv(x)
        pool = self.pool(x)
        out = torch.cat([conv, pool], dim=1)
        out = self.bn(out)
        return self.prelu(out)


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = False,
        dilation: int = 1,
        asymmetric: bool = False,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        internal_channels = out_channels // 4
        stride = 2 if downsample else 1

        self.proj = nn.Conv2d(in_channels, internal_channels, kernel_size=1, stride=stride, bias=False)
        self.proj_bn = nn.BatchNorm2d(internal_channels)
        self.proj_act = nn.PReLU(internal_channels)

        if asymmetric:
            self.conv = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(5, 1), padding=(2, 0), bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(internal_channels),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, 5), padding=(0, 2), bias=False),
            )
        else:
            self.conv = nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )

        self.conv_bn = nn.BatchNorm2d(internal_channels)
        self.conv_act = nn.PReLU(internal_channels)
        self.expand = nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False)
        self.expand_bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.out_act = nn.PReLU(out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if downsample else None
        if in_channels != out_channels:
            self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.match_bn = nn.BatchNorm2d(out_channels)
        else:
            self.match_channels = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.proj(x)
        out = self.proj_bn(out)
        out = self.proj_act(out)
        out = self.conv(out)
        out = self.conv_bn(out)
        out = self.conv_act(out)
        out = self.expand(out)
        out = self.expand_bn(out)
        out = self.dropout(out)

        if self.downsample:
            identity = self.pool(identity)
        if self.match_channels is not None:
            identity = self.match_channels(identity)
            identity = self.match_bn(identity)

        out = out + identity
        return self.out_act(out)


class UpsampleBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        internal_channels = out_channels // 4

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels),
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels),
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip_act = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.main(x)
        skip = self.skip(x)
        skip = F.interpolate(skip, size=main.shape[-2:], mode="bilinear", align_corners=False)
        return self.skip_act(main + skip)


class ENet21(nn.Module):
    def __init__(self, num_classes: int = 1) -> None:
        super().__init__()
        self.initial = InitialBlock(3, 16)
        self.stage1 = nn.Sequential(
            Bottleneck(16, 64, downsample=True, dropout=0.01),
            Bottleneck(64, 64, dropout=0.01),
            Bottleneck(64, 64, dropout=0.01),
            Bottleneck(64, 64, dropout=0.01),
        )
        self.stage2 = nn.Sequential(
            Bottleneck(64, 128, downsample=True),
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilation=2),
            Bottleneck(128, 128, asymmetric=True),
            Bottleneck(128, 128, dilation=4),
            Bottleneck(128, 128),
        )
        self.stage3 = nn.Sequential(
            Bottleneck(128, 128),
            Bottleneck(128, 128, dilation=8),
            Bottleneck(128, 128, asymmetric=True),
            Bottleneck(128, 128, dilation=16),
            Bottleneck(128, 128),
        )
        self.decoder = nn.Sequential(
            UpsampleBottleneck(128, 64),
            Bottleneck(64, 64, dropout=0.01),
            UpsampleBottleneck(64, 16, dropout=0.01),
            Bottleneck(16, 16, dropout=0.01),
        )
        self.classifier = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.decoder(x)
        return self.classifier(x)
