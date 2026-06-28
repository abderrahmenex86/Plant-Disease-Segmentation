import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=116, base_channels=64):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_channels)
        self.d1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels, base_channels * 2))
        self.d2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_channels * 2, base_channels * 4))
        self.u1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.c1 = DoubleConv(base_channels * 4, base_channels * 2)
        self.u2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.c2 = DoubleConv(base_channels * 2, base_channels)
        self.outc = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x):
        x1, x2 = self.inc(x), self.d1(self.inc(x))
        x = self.c1(torch.cat([self.u1(self.d2(x2)), x2], dim=1))
        return self.outc(self.c2(torch.cat([self.u2(x), x1], dim=1)))


class DeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        from torchvision.models.segmentation import (
            DeepLabV3_MobileNet_V3_Large_Weights,
            deeplabv3_mobilenet_v3_large,
        )

        self.net = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT)
        self.net.classifier[4] = nn.Conv2d(256, num_classes, 1)
        self.net.aux_classifier = None

    def forward(self, x):
        return self.net(x)["out"]


class LinkNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.Linknet(encoder_name="mobilenet_v2", encoder_weights="imagenet", classes=num_classes)

    def forward(self, x):
        return self.net(x)


class SegFormer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.Unet(encoder_name="mit_b0", encoder_weights="imagenet", classes=num_classes)

    def forward(self, x):
        return self.net(x)


class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2.0):
        super().__init__()
        self.num_classes, self.alpha, self.gamma = num_classes, alpha, gamma

    def forward(self, p, t):
        t_long = t.squeeze(1).long()
        if self.num_classes == 1:
            ce = F.binary_cross_entropy_with_logits(p, t.float(), reduction="none")
            pt = torch.exp(-ce)
            focal = (self.alpha * (1 - pt) ** self.gamma * ce).mean()
            p_sig = torch.sigmoid(p)
            dice = 1 - (2.0 * (p_sig * t).sum((2, 3)) / (p_sig.sum((2, 3)) + t.sum((2, 3)) + 1e-6)).mean()
        else:
            ce = F.cross_entropy(p, t_long, reduction="none")
            pt = torch.exp(-ce)
            focal = ((1 - pt) ** self.gamma * ce).mean()
            p_s, t_oh = F.softmax(p, 1), F.one_hot(t_long, self.num_classes).permute(0, 3, 1, 2).float()
            dice = 1 - (2.0 * (p_s * t_oh).sum((2, 3)) / (p_s.sum((2, 3)) + t_oh.sum((2, 3)) + 1e-6)).mean()
        return focal + dice
