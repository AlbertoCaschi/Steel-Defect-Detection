import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# 1. Standard Pure PyTorch U-Net Implementation (Standalone / No Dependencies)
# -------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool2D then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling followed by concatenation with skip connection and DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if necessary to match x2 spatial dimensions exactly
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class StandaloneUNet(nn.Module):
    """
    Standard U-Net architecture for multi-label semantic segmentation.
    Outputs logits of shape (Batch, num_classes, H, W).
    """
    def __init__(self, in_channels=3, num_classes=4, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# -------------------------------------------------------------------------
# 2. Pretrained Encoder Factory (Matching Paper: EfficientNet-B0 / ResNet34)
# -------------------------------------------------------------------------

def build_unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet", # imagenet normalization performed on the dataset
    in_channels=3,
    num_classes=4,
    use_smp=True
):
    """
    Factory function to build a U-Net model.
    
    If use_smp=True (recommended), uses `segmentation_models_pytorch` to leverage
    pretrained backbones (e.g. EfficientNet-B0 or ResNet34).
    If use_smp=False or smp is not installed, falls back to StandaloneUNet.
    """
    if use_smp:
        try:
            import segmentation_models_pytorch as smp
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=num_classes,
                activation=None  # Outputs raw logits for numerical stability with BCEWithLogitsLoss
            )
            return model
        except ImportError:
            print("[Warning] `segmentation_models_pytorch` not installed. Falling back to StandaloneUNet.")
            return StandaloneUNet(in_channels=in_channels, num_classes=num_classes)
    else:
        return StandaloneUNet(in_channels=in_channels, num_classes=num_classes)


if __name__ == "__main__":
    # Test tensor: (Batch=2, Channels=3, Height=256, Width=256)
    dummy_input = torch.randn(2, 3, 256, 256)
    
    # Instantiate model
    model = build_unet(encoder_name="efficientnet-b0", num_classes=4, use_smp=False)
    output = model(dummy_input)
    
    print("Model initialized successfully.")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Expected: [2, 4, 256, 256])")