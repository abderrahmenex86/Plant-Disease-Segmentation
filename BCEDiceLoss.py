import segmentation_models_pytorch.losses as smp_losses
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight=None, aux_weight=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = smp_losses.DiceLoss(mode=smp_losses.BINARY_MODE, from_logits=True)
        self.aux_weight = aux_weight

    def forward(self, outputs, masks):
        if isinstance(outputs, dict):
            main_loss = self.bce(outputs["out"], masks) + self.dice(outputs["out"], masks)
            aux_loss = self.bce(outputs["aux"], masks) + self.dice(outputs["aux"], masks)
            return (1 - self.aux_weight) * main_loss + self.aux_weight * aux_loss

        return self.bce(outputs, masks) + self.dice(outputs, masks)
