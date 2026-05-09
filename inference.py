import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.utils import draw_segmentation_masks

from model import DiseaseSegmenter


def run_inference(image_path, model_path="best_disease_segmenter.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiseaseSegmenter().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    infer_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((256, 256), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    raw_image = Image.open(image_path).convert("RGB")
    input_tensor = infer_transforms(raw_image)

    batch_tensor = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        raw_logits = model(batch_tensor)

    probs = torch.sigmoid(raw_logits).squeeze(0)
    predicted_mask = (probs > 0.5).to(torch.bool).cpu()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    unnorm_image = (input_tensor.cpu() * std) + mean
    image_uint8 = (unnorm_image * 255).clamp(0, 255).to(torch.uint8)

    overlaid = draw_segmentation_masks(
        image_uint8, predicted_mask, alpha=0.6, colors="red"
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(overlaid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title("Predicted Disease Mask")
    plt.show()


if __name__ == "__main__":
    run_inference("test.jpg")
