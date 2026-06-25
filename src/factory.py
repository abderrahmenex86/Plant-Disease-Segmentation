import torch
import torch.optim as optim

from src.models import (
    BCEDiceLoss,
    DeepLabV3Segmenter,
    MulticlassDiceLoss,
    UNetSegmenter,
)


def build_model(model_type: str, num_classes: int, **kwargs):
    if model_type == "unet":
        encoder_name = kwargs.get("encoder_name", "mobilenet_v2")
        encoder_weights = kwargs.get("encoder_weights", "imagenet")
        return UNetSegmenter(num_classes=num_classes, encoder_name=encoder_name, encoder_weights=encoder_weights)
    elif model_type == "deeplabv3":
        pretrained = kwargs.get("pretrained", True)
        return DeepLabV3Segmenter(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model architecture selection: {model_type}")


def build_optimizer(model, lr: float, weight_decay: float, model_type: str, **kwargs):
    if model_type == "unet":
        encoder_params = list(model.model.encoder.parameters())
        decoder_params = list(model.model.decoder.parameters()) + list(model.model.segmentation_head.parameters())
        params = [{"params": encoder_params, "lr": lr * 0.1}, {"params": decoder_params, "lr": lr}]
    elif model_type == "deeplabv3":
        backbone_params = list(model.model.backbone.parameters())
        classifier_params = list(model.model.classifier.parameters())
        if model.model.aux_classifier:
            classifier_params += list(model.model.aux_classifier.parameters())

        params = [{"params": backbone_params, "lr": lr * 0.1}, {"params": classifier_params, "lr": lr}]
    else:
        params = model.parameters()

    optimizer_type = kwargs.get("optimizer", "adamw")
    if optimizer_type.lower() == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")


def build_scheduler(optimizer, scheduler_type: str = "plateau", **kwargs):
    if scheduler_type == "plateau":
        mode = kwargs.get("scheduler_mode", "max")
        factor = kwargs.get("scheduler_factor", 0.5)
        patience = kwargs.get("scheduler_patience", 4)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    elif scheduler_type == "cosine":
        t_max = kwargs.get("scheduler_t_max", 40)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    elif scheduler_type == "step":
        step_size = kwargs.get("scheduler_step_size", 10)
        gamma = kwargs.get("scheduler_gamma", 0.1)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        return None


def build_criterion(num_classes: int, **kwargs):
    if num_classes == 1:
        pos_weight_val = kwargs.get("pos_weight", 5.0)
        pos_weight = torch.tensor([pos_weight_val]) if pos_weight_val is not None else None
        return BCEDiceLoss(pos_weight=pos_weight)
    else:
        return MulticlassDiceLoss()
