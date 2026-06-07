import argparse
import json

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, ToImage
from torchvision.utils import draw_segmentation_masks

from model import DiseaseSegmenter


def run_inference(image_path, model_path, class_mapping_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiseaseSegmenter().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    class_dict = None
    if class_mapping_path:
        with open(class_mapping_path, "r") as f:
            mapping = json.load(f)
            class_dict = {v + 1: k for k, v in mapping.items()}

    infer_transforms = Compose(
        [
            ToImage(),
            Resize((520, 520)),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    raw_image = Image.open(image_path).convert("RGB")
    input_tensor = infer_transforms(raw_image)
    batch_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        raw_logits = model(batch_tensor)
        if isinstance(raw_logits, dict):
            raw_logits = raw_logits["out"]

    predicted_classes = raw_logits.argmax(dim=1).squeeze(0).cpu()
    # print(class_dict)
    print("Predicted classes in the image:", class_dict[int(torch.max(predicted_classes))])

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    unnorm_image = (input_tensor.cpu() * std) + mean
    image_uint8 = (unnorm_image * 255).clamp(0, 255).to(torch.uint8)

    present_classes = torch.unique(predicted_classes)
    present_classes = present_classes[present_classes != 0]

    if len(present_classes) == 0:
        print("no detected disease.")
        overlaid = image_uint8
    else:
        masks = []
        labels = []

        for cls in present_classes:
            masks.append(predicted_classes == cls)
            if class_dict:
                labels.append(class_dict.get(cls.item(), f"Class {cls.item()}"))
            else:
                labels.append(f"Class {cls.item()}")

        masks = torch.stack(masks)

        overlaid = draw_segmentation_masks(image_uint8, masks, alpha=0.6, colors="red")

    plt.figure(figsize=(10, 10))
    plt.imshow(overlaid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title("Predicted Disease Mask")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--class_mapping", type=str, required=False)

    args = parser.parse_args()

    run_inference(image_path=args.image_path, model_path=args.model_path, class_mapping_path=args.class_mapping)
