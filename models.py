import segmentation_models_pytorch as smp
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)


class DiseaseSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT, weights_backbone=MobileNet_V3_Large_Weights.DEFAULT
        )

        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels, 1, kernel_size=(1, 1))

        if self.model.aux_classifier:
            in_channels_aux = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Conv2d(in_channels_aux, 1, kernel_size=(1, 1))

    def forward(self, x):
        x = self.model(x)
        if self.training:
            return x
        return x["out"]


class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes=115):
        super().__init__()
        self.model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        in_features = self.model.classifier[3].in_features

        self.model.classifier[2] = nn.Dropout(p=0.5, inplace=True)
        self.model.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


# class DiseaseSegmenter(nn.Module):
#     def __init__(self, encoder_name="mobilenet_v2", encoder_weights="imagenet"):
#         super().__init__()
#         self.model = smp.Unet(
#             encoder_name=encoder_name,
#             encoder_weights=encoder_weights,
#             in_channels=3,
#             classes=1,
#         )
#
#     def forward(self, x):
#         return self.model(x)
