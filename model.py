"""DenseNet-BC for five-class building image classification."""

import torch
from torch import nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int = 16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.SiLU(),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, self.layers(x)), dim=1)


class DenseNetBC(nn.Module):
    """Randomly initialized DenseNet-BC returning five unnormalized logits."""

    def __init__(self):
        super().__init__()
        growth_rate = 16
        channels = 32
        self.stem = nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1, bias=False)
        blocks = []
        block_sizes = (8, 12, 16, 12)
        for index, size in enumerate(block_sizes):
            layers = []
            for _ in range(size):
                layers.append(DenseLayer(channels, growth_rate))
                channels += growth_rate
            blocks.append(nn.Sequential(*layers))
            if index < len(block_sizes) - 1:
                compressed = channels // 2
                blocks.append(nn.Sequential(
                    nn.BatchNorm2d(channels),
                    nn.SiLU(),
                    nn.Conv2d(channels, compressed, kernel_size=1, bias=False),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                ))
                channels = compressed
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(channels, 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self.stem(x)))


if __name__ == '__main__':
    model = DenseNetBC().eval()
    parameters = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 5) and torch.isfinite(output).all()
    assert 1_000_000 <= parameters <= 3_000_000
    print(f'Parameters: {parameters:,}; output shape: {tuple(output.shape)}')
