import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)


class UNetSegmenter(nn.Module):
    def __init__(self, num_classes=116, encoder_name="mobilenet_v2", encoder_weights="imagenet"):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)


class DeepLabV3Segmenter(nn.Module):
    def __init__(self, num_classes=116, pretrained=True):
        super().__init__()
        weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        weights_backbone = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None

        self.model = deeplabv3_mobilenet_v3_large(
            weights=weights,
            weights_backbone=weights_backbone,
            aux_loss=True,
        )

        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1))

        if self.model.aux_classifier:
            in_channels_aux = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=(1, 1))

    def forward(self, x):
        x = self.model(x)
        if self.training:
            return x
        return x["out"]


class MulticlassDiceLoss(nn.Module):
    def __init__(self, aux_weight=0.3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE, from_logits=True)
        self.aux_weight = aux_weight

    def _compute_loss(self, pred, target):
        return self.ce(pred, target) + self.dice(pred, target)

    def forward(self, outputs, masks):
        masks = masks.squeeze(1).long()

        if isinstance(outputs, dict):
            main_loss = self._compute_loss(outputs["out"], masks)
            aux_loss = self._compute_loss(outputs["aux"], masks)
            return (1 - self.aux_weight) * main_loss + self.aux_weight * aux_loss

        return self._compute_loss(outputs, masks)


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None, aux_weight=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
        self.aux_weight = aux_weight

    def _compute_loss(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)

    def forward(self, outputs, masks):
        masks = masks.to(dtype=torch.float32)

        if isinstance(outputs, dict):
            main_loss = self._compute_loss(outputs["out"], masks)
            aux_loss = self._compute_loss(outputs["aux"], masks)
            return (1 - self.aux_weight) * main_loss + self.aux_weight * aux_loss

        return self._compute_loss(outputs, masks)
