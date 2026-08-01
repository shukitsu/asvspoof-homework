import torch
import torch.nn as nn


class MFM(nn.Module):
    """
    Max Feature Map activation.
    Splits channels into two equal parts and keeps maximum.
    """

    def forward(self, x):
        assert x.shape[1] % 2 == 0, "Number of channels must be even."

        x1, x2 = torch.chunk(x, 2, dim=1)

        return torch.maximum(x1, x2)


class ConvBlock(nn.Module):
    """
    Conv + MFM (+ optional MaxPool)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        pool=False,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels * 2,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            MFM(),
        ]

        if pool:
            layers.append(
                nn.MaxPool2d(
                    kernel_size=2,
                    stride=2,
                )
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class LCNN(nn.Module):

    def __init__(
        self,
        num_classes=2,
        dropout_rate=0.5,
    ):
        super().__init__()

        self.features = nn.Sequential(

            ConvBlock(
                1,
                64,
                kernel_size=5,
                padding=2,
                pool=True,
            ),

            ConvBlock(
                64,
                64,
                kernel_size=1,
            ),

            ConvBlock(
                64,
                96,
                kernel_size=3,
                padding=1,
                pool=True,
            ),

            ConvBlock(
                96,
                96,
                kernel_size=1,
            ),

            ConvBlock(
                96,
                128,
                kernel_size=3,
                padding=1,
                pool=True,
            ),

            ConvBlock(
                128,
                128,
                kernel_size=1,
            ),

            ConvBlock(
                128,
                160,
                kernel_size=3,
                padding=1,
            ),

            ConvBlock(
                160,
                160,
                kernel_size=1,
            ),

            ConvBlock(
                160,
                192,
                kernel_size=3,
                padding=1,
                pool=True,
            ),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(

            nn.Linear(
                192,
                256,
            ),

            nn.Dropout(
                p=dropout_rate,
            ),

            nn.BatchNorm1d(
                256,
            ),

            nn.ReLU(inplace=True),

            nn.Linear(
                256,
                num_classes,
            ),
        )

        self._initialize_weights()

    def forward(
        self,
        data_object,
        **batch,
    ):

        x = self.features(data_object)

        x = self.global_pool(x)

        x = torch.flatten(x, 1)

        logits = self.classifier(x)

        return {
            "logits": logits,
        }

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Conv2d):

                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):

                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm1d):

                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def __str__(self):

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        result = super().__str__()

        result += f"\nAll parameters: {total}"
        result += f"\nTrainable parameters: {trainable}"

        return result