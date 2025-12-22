from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNetAudio(nn.Module):
    def __init__(
        self,
        n_classes: int = 50,
        channels: tuple[int, int, int, int] = (16, 32, 64, 128),
        blocks: tuple[int, int, int, int] = (2, 2, 2, 2),
    ):
        super().__init__()
        self.in_channels = channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(channels[0], blocks[0], stride=1)
        self.layer2 = self._make_layer(channels[1], blocks[1], stride=2)
        self.layer3 = self._make_layer(channels[2], blocks[2], stride=2)
        self.layer4 = self._make_layer(channels[3], blocks[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.features = nn.Sequential(
            self.stem,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.pool,
        )
        self.classifier = nn.Linear(channels[3], n_classes)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
