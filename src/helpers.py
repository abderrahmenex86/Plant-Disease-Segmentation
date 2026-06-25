import json
import os

import matplotlib.pyplot as plt


def download_dataset(target_dir="dataset"):
    os.makedirs(target_dir, exist_ok=True)
    splits = ["train", "val", "test"]
    for split in splits:
        os.makedirs(os.path.join(target_dir, "plantsegv3", "images", split), exist_ok=True)
        os.makedirs(os.path.join(target_dir, "plantsegv3", "annotations", split), exist_ok=True)
    print(f"\nDirectories created in '{target_dir}'. To begin training:")
    print("1. Download the plantsegv3 disease dataset.")
    print("2. Place raw RGB images (.jpg) inside 'dataset/plantsegv3/images/{split}/'")
    print("3. Place matching mask targets (.png) inside 'dataset/plantsegv3/annotations/{split}/'")


def verify_dataset_structure(root_dir="dataset/plantsegv3"):
    print(f"Inspecting dataset alignment: {root_dir}")
    splits = ["train", "val"]
    for split in splits:
        img_dir = os.path.join(root_dir, "images", split)
        mask_dir = os.path.join(root_dir, "annotations", split)

        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f"  [WARNING] Split folder missing: {split}")
            continue

        imgs = sorted(os.listdir(img_dir))
        masks = sorted(os.listdir(mask_dir))
        print(f"  Split '{split}': Found {len(imgs)} source files and {len(masks)} target masks.")

        mismatches = 0
        for i in imgs[:20]:
            base = os.path.splitext(i)[0]
            if f"{base}.png" not in masks:
                mismatches += 1
        if mismatches > 0:
            print(f"  [ERROR] Alignment mismatch in {mismatches} of the first 20 file check targets.")
        else:
            print("  --> Match validation checks complete.")


def generate_analytics_plots():
    os.makedirs("docs/figs", exist_ok=True)
    artifacts_dir = "artifacts"
    if not os.path.exists(artifacts_dir):
        print("No run records directory found.")
        return

    runs = sorted(
        [
            os.path.join(artifacts_dir, d)
            for d in os.listdir(artifacts_dir)
            if os.path.isdir(os.path.join(artifacts_dir, d)) and not d.startswith("default_")
        ]
    )
    if not runs:
        print("No runs found in artifacts.")
        return

    latest_run = runs[-1]
    history_file = os.path.join(latest_run, "model_history.json")
    if not os.path.exists(history_file):
        print(f"No metric history found at: {latest_run}")
        return

    with open(history_file, "r") as f:
        history = json.load(f)

    train_loss = history["train"]["loss"]
    val_loss = history["val"]["loss"]
    train_dice = history["train"]["dice"]
    val_dice = history["val"]["dice"]
    train_iou = history["train"]["iou"]
    val_iou = history["val"]["iou"]

    epochs = list(range(1, len(train_loss) + 1))

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(epochs, train_loss, "o-", color="royalblue", label="Train")
    axs[0, 0].plot(epochs, val_loss, "s--", color="darkorange", label="Val")
    axs[0, 0].set_title("Dice + CE Objective Curves")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()

    axs[0, 1].plot(epochs, train_dice, "o-", color="forestgreen", label="Train")
    axs[0, 1].plot(epochs, val_dice, "s--", color="crimson", label="Val")
    axs[0, 1].set_title("Dice Overlap Accuracy")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Dice Score (%)")
    axs[0, 1].legend()

    axs[1, 0].plot(epochs, train_iou, "o-", color="indigo", label="Train")
    axs[1, 0].plot(epochs, val_iou, "s--", color="darkorchid", label="Val")
    axs[1, 0].set_title("Mean Intersection over Union")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("IoU (%)")
    axs[1, 0].legend()

    loss_delta = [v - t for t, v in zip(train_loss, val_loss)]
    axs[1, 1].bar(epochs, loss_delta, color="teal", alpha=0.7, label="Val - Train Loss")
    axs[1, 1].set_title("Generalization Variance")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Loss Delta")
    axs[1, 1].legend()

    plt.tight_layout()
    out_path = "docs/figs/training_metrics_grid.png"
    plt.savefig(out_path, dpi=300)
    print(f"Diagnostic plots saved to: {out_path}")
