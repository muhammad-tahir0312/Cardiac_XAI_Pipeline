import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------------------------------------------------------------------------
# UNet building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Two conv layers with BatchNorm and LeakyReLU."""
    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock(nn.Module):
    """MaxPool followed by ConvBlock."""
    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout_p),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Bilinear upsample (or ConvTranspose) followed by ConvBlock."""
    def __init__(self, in_channels1, in_channels2, out_channels,
                 dropout_p, bilinear=True):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2,
                                          kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.in_chns = params['in_chns']
        self.ft_chns = params['feature_chns']
        self.dropout  = params['dropout']
        assert len(self.ft_chns) == 5

        self.in_conv = ConvBlock(self.in_chns,    self.ft_chns[0], self.dropout[0])
        self.down1   = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2   = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3   = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4   = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.ft_chns = params['feature_chns']
        self.n_class = params['class_num']
        assert len(self.ft_chns) == 5
        bilinear = params.get('bilinear', True)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], 0.0, bilinear)
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], 0.0, bilinear)
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], 0.0, bilinear)
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], 0.0, bilinear)

        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class, kernel_size=3, padding=1)

    def forward(self, feature):
        x0, x1, x2, x3, x4 = feature
        x = self.up1(x4, x3)
        x = self.up2(x,  x2)
        x = self.up3(x,  x1)
        x = self.up4(x,  x0)
        return self.out_conv(x)


class UNet(nn.Module):
    def __init__(self, in_chns=1, class_num=4):
        super().__init__()
        params = {
            'in_chns':      in_chns,
            'feature_chns': [32, 64, 128, 256, 512],
            'dropout':      [0.05, 0.1, 0.2, 0.3, 0.5],
            'class_num':    class_num,
            'bilinear':     False,
        }
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# DenseNet-121 diagnosis model
# ---------------------------------------------------------------------------

class DenseNetDiagnosis(nn.Module):
    """
    DenseNet-121 adapted for 5-class cardiac disease classification.

    Input : 2-channel tensor (ED slice, ES slice) — shape (B, 2, 256, 256).
    Output: logits  (B, class_num).

    Weight initialisation for the 2-channel first conv (pretrained=True):
      The 3-channel ImageNet conv has shape (64, 3, 7, 7).
      We average the weights of channels 0 and 1 from the pretrained conv
      → shape (64, 2, 7, 7) — a principled warm-start that preserves the
      low-level edge/texture detectors learned on ImageNet.
    """

    def __init__(self, class_num=5, pretrained=True):
        super().__init__()

        if pretrained:
            self.densenet = models.densenet121(
                weights=models.DenseNet121_Weights.IMAGENET1K_V1
            )
        else:
            self.densenet = models.densenet121(weights=None)

        # -------------------------------------------------------------- #
        # Dual-channel MRI (2-ch)   #
        # -------------------------------------------------------------- #
        original_conv = self.densenet.features.conv0   # (64, 3, 7, 7)

        new_conv = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if pretrained:
            # Average first two channel weight slices from the original conv.
            # Shape: (64, 3, 7, 7) → mean over channels 0-1 → (64, 1, 7, 7)
            # → repeat → (64, 2, 7, 7)
            with torch.no_grad():
                w_pretrained = original_conv.weight.data     # (64, 3, 7, 7)
                # Average the first 2 channels (both represent luminance-like info)
                w_avg = w_pretrained[:, :2, :, :].mean(dim=1, keepdim=True)  # (64,1,7,7)
                # Tile to 2 channels
                new_conv.weight.copy_(w_avg.repeat(1, 2, 1, 1))              # (64,2,7,7)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out',
                                    nonlinearity='relu')

        self.densenet.features.conv0 = new_conv

        # Replace classifier head for n-class output
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, class_num)

    def forward(self, x):
        return self.densenet(x)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== UNet smoke test ===")
    unet = UNet(in_chns=1, class_num=4)
    x = torch.randn(2, 1, 256, 256)
    y = unet(x)
    print(f"Input: {x.shape}  →  Output: {y.shape}")   # expect (2, 4, 256, 256)

    print("\n=== DenseNetDiagnosis smoke test (no pretrained) ===")
    dns = DenseNetDiagnosis(class_num=5, pretrained=False)
    x2  = torch.randn(2, 2, 256, 256)
    y2  = dns(x2)
    print(f"Input: {x2.shape}  →  Output: {y2.shape}")  # expect (2, 5)

    print("\n=== DenseNetDiagnosis weight init check (pretrained) ===")
    dns_pt = DenseNetDiagnosis(class_num=5, pretrained=True)
    w = dns_pt.densenet.features.conv0.weight.data
    print(f"First conv weight shape: {w.shape}")         # expect (64, 2, 7, 7)
    assert w.shape == (64, 2, 7, 7), "Weight shape mismatch!"
    print("All smoke tests passed.")
