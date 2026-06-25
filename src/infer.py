import json
import os

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.utils import draw_segmentation_masks

from src.factory import build_model


def run_smart_inference(image_path: str, run_dir: str, save_output: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hp_path = os.path.join(run_dir, "hyperparameters.json")
    if not os.path.exists(hp_path):
        raise FileNotFoundError(f"Missing run parameters at: {hp_path}")

    with open(hp_path, "r") as f:
        hp = json.load(f)

    model_type = hp.get("model_type", "unet")
    num_classes = hp.get("num_classes", 116)
    img_size = hp.get("img_size", 520)
    encoder_name = hp.get("encoder_name", "mobilenet_v2")
    encoder_weights = hp.get("encoder_weights", "imagenet")

    model = build_model(
        model_type=model_type, num_classes=num_classes, encoder_name=encoder_name, encoder_weights=encoder_weights
    ).to(device)

    best_weights_path = os.path.join(run_dir, "best_model.pth")
    if not os.path.exists(best_weights_path):
        checkpoint_path = os.path.join(run_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Warning: Loaded fallback checkpoint weights.")
        else:
            raise FileNotFoundError(f"No usable model states inside: {run_dir}")
    else:
        model.load_state_dict(torch.load(best_weights_path, map_location=device))
        print("Restored best model weight parameters.")

    model.eval()

    raw_image = Image.open(image_path).convert("RGB")
    image_v2 = tv_tensors.Image(raw_image)

    single_infer_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((img_size, img_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = single_infer_transform(image_v2)
    batch_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(batch_tensor)
        logits = outputs["out"] if isinstance(outputs, dict) else outputs

    if num_classes == 1:
        predicted_classes = (torch.sigmoid(logits) > 0.5).long().squeeze(0).cpu()
    else:
        predicted_classes = logits.argmax(dim=1).squeeze(0).cpu()

    present_elements = [p.item() for p in torch.unique(predicted_classes) if p.item() > 0]
    print(f"Detected plant disease classification index mapping: {present_elements}")

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    unnorm_image = (input_tensor.cpu() * std) + mean
    image_uint8 = (unnorm_image * 255).clamp(0, 255).to(torch.uint8)

    if len(present_elements) == 0:
        print("Foliage identified healthy (no masks found).")
        overlaid = image_uint8
    else:
        masks = [predicted_classes == cls for cls in present_elements]
        masks_tensor = torch.stack(masks)
        overlaid = draw_segmentation_masks(image_uint8, masks_tensor, alpha=0.6, colors="red")

    plt.figure(figsize=(10, 10))
    plt.imshow(overlaid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title(f"Segmenti Output Mask Overlay - Categories: {present_elements}")
    plt.tight_layout()

    if save_output:
        os.makedirs("docs/figs", exist_ok=True)
        out_fig_path = os.path.join("docs/figs", f"inference_output_{os.path.basename(image_path)}")
        plt.savefig(out_fig_path)
        print(f"Exported prediction visualization to: {out_fig_path}")
    else:
        plt.show()
