from typing import Callable

import torch
import torch.nn as nn


class my_LeNet(nn.Module):
    """A small CNN: a few conv+pool blocks, then a few linear layers."""

    def __init__(
        self,
        input_size: tuple[int, int] = (224, 224),
        channels: tuple[int, ...] = (16, 32),
        pool_size: int = 2,
        dropout: float = 0.0,
        linears_layers: tuple[int, ...] = (100, 100),
        activation: Callable[[], nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()

        layers = []
        in_channels = 3

        for out_channels in channels:
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=5
                )
            )

            layers.append(activation())
            layers.append(nn.MaxPool2d(pool_size))

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            x = torch.zeros(1, 3, *input_size)
            x = self.features(x)
            in_features = x.flatten(1).shape[1]

        layers = [nn.Flatten()]

        for out_features in linears_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(activation())
            in_features = out_features

        layers.append(nn.Linear(in_features, 9))

        self.classifier = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.features(X)
        X = self.classifier(X)
        return X
