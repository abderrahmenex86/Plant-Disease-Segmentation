import torch

from src.dataset import get_dataloaders
from src.factory import build_pipeline


def run_sanity_checks(**kwargs):
    print("============================================================")
    print("RUNNING GRADIENT AND PIPELINE INTEGRITY CHECKS")
    print("============================================================")

    test_kwargs = kwargs.copy()
    test_kwargs["batch_size"] = 2
    test_kwargs["limit_dataset"] = 5

    train_loader, _ = get_dataloaders(**test_kwargs)
    images_batch, masks_batch = next(iter(train_loader))

    print(f"  --> Batch loading checks verified.")
    print(f"      Images Batch Shape: {list(images_batch.shape)}")
    print(f"      Masks Batch Shape : {list(masks_batch.shape)}")

    assert images_batch.shape == (2, 3, test_kwargs["img_size"], test_kwargs["img_size"]), "Images shape mismatch."
    assert masks_batch.shape == (2, 1, test_kwargs["img_size"], test_kwargs["img_size"]), "Masks shape mismatch."
    assert images_batch.min() >= -3.0 and images_batch.max() <= 3.0, "Normalization out of valid scaling bounds."

    if test_kwargs["num_classes"] == 1:
        assert masks_batch.min() >= 0.0 and masks_batch.max() <= 1.0, "Binary mask values exceed boundaries."
    else:
        assert (
            masks_batch.min() >= 0 and masks_batch.max() < test_kwargs["num_classes"]
        ), "Target class indices exceed num_classes range."

    print("[Test 2/4] Testing forward execution pass...")
    model, criterion, optimizer, _ = build_pipeline(**test_kwargs)
    model.train()

    device = test_kwargs["device"]
    test_images = images_batch.to(device)
    test_masks = masks_batch.to(device)

    outputs = model(test_images)
    assert outputs.shape == (
        2,
        test_kwargs["num_classes"],
        test_kwargs["img_size"],
        test_kwargs["img_size"],
    ), "Model output dimensions mismatch."
    print("  --> Forward pass verification successful.")

    print("[Test 3/4] Testing gradient backward pass...")
    loss_value = criterion(outputs, test_masks)
    loss_value.backward()

    has_gradients = False
    gradient_anomaly_detected = False
    for parameter_name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None:
            has_gradients = True
            if torch.isnan(parameter.grad).any() or torch.isinf(parameter.grad).any():
                gradient_anomaly_detected = True
                print(f"      [WARNING] Gradient anomaly (NaN/Inf) detected inside: {parameter_name}")

    assert has_gradients, "No parameter gradients were updated in backward pass."
    assert not gradient_anomaly_detected, "Gradients contain invalid computational states."
    print("  --> Backward pass and gradient flow successfully verified.")

    print("[Test 4/4] Verifying parameter optimization grouping...")
    assert len(optimizer.param_groups) > 0, "No parameter groups initialized inside optimizer."
    for group_index, parameter_group in enumerate(optimizer.param_groups):
        print(f"      Group {group_index} | Base Learning Rate: {parameter_group['lr']}")

    print("\n============================================================")
    print("SANITY CHECKS COMPLETED. PIPELINE VERIFIED.")
    print("============================================================")
    return True
