import torch

from src.dataset import get_dataloaders
from src.factory import build_criterion, build_model, build_optimizer


def run_sanity_checks(args):
    print("=" * 60)
    print("RUNNING GRADIENT AND PIPELINE INTEGRITY CHECKS")
    print("=" * 60)

    print("[Test 1/4] Checking DataLoader slicing and bounds...")
    try:
        train_loader, val_loader = get_dataloaders(
            root_dir=args.root_dir, batch_size=2, num_classes=args.num_classes, img_size=args.img_size, limit_dataset=10
        )
        images, masks = next(iter(train_loader))
        print(f"  --> Success. Batch loading specs:")
        print(f"      Image Shape: {images.shape} (Expected: [2, 3, {args.img_size}, {args.img_size}])")
        print(f"      Mask Shape : {masks.shape}")

        assert images.min() >= -3.0 and images.max() <= 3.0, "Normalization bounds missing."
        if args.num_classes == 1:
            assert masks.min() >= 0.0 and masks.max() <= 1.0, "Mask bounds out of range."
        else:
            assert masks.min() >= 0 and masks.max() < args.num_classes, "Index range overflow."
        print("  --> Tensor boundaries checked.")
    except Exception as e:
        print(f"  [FAIL] DataLoader check: {e}")
        return False

    print("[Test 2/4] Testing forward execution passes...")
    try:
        model = build_model(
            model_type=args.model_type,
            num_classes=args.num_classes,
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
        ).to(args.device)
        model.train()

        test_images = images.to(args.device)
        outputs = model(test_images)

        if isinstance(outputs, dict):
            print(f"  --> Found aux headers: {list(outputs.keys())}")
        else:
            print(f"  --> Output Shape: {outputs.shape}")
        print("  --> Forward pass calculated.")
    except Exception as e:
        print(f"  [FAIL] Forward pass check: {e}")
        return False

    print("[Test 3/4] Testing gradient backward pass...")
    try:
        criterion = build_criterion(args.num_classes, pos_weight=args.pos_weight)
        test_masks = masks.to(args.device)
        loss = criterion(outputs, test_masks)
        loss.backward()

        has_grads = False
        grad_nan_check = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                has_grads = True
                if torch.isnan(param.grad).any():
                    grad_nan_check = True
                    print(f"      [WARNING] Gradient NaN in layer: {name}")

        assert has_grads, "No gradients were computed."
        assert not grad_nan_check, "Gradients contained NaNs."
        print("  --> Backward pass verified.")
    except Exception as e:
        print(f"  [FAIL] Backward pass check: {e}")
        return False

    print("[Test 4/4] Verifying parameter groupings...")
    try:
        optimizer = build_optimizer(model, lr=1e-3, weight_decay=1e-3, model_type=args.model_type)
        print(f"  --> Param Groups Identified: {len(optimizer.param_groups)}")
        for idx, grp in enumerate(optimizer.param_groups):
            print(f"      Group {idx} Target LR: {grp['lr']}")
        print("  --> Parameter group mapping verified.")
    except Exception as e:
        print(f"  [FAIL] Parameter splitting check: {e}")
        return False

    print("\n" + "=" * 60)
    print("SANITY CHECKS COMPLETED. PIPELINE VERIFIED.")
    print("=" * 60 + "\n")
    return True
