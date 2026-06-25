import json
import os

import optuna
import torch

from src.dataset import get_dataloaders
from src.factory import build_criterion, build_model, build_optimizer, build_scheduler
from src.trainer import Trainer


def run_optuna_study(args):
    optuna.logging.set_verbosity(optuna.logging.INFO)
    os.makedirs(args.run_dir, exist_ok=True)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["adamw", "adam"])

        train_loader, val_loader = get_dataloaders(
            root_dir=args.root_dir,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            img_size=args.img_size,
            limit_dataset=200,
        )

        model = build_model(
            model_type=args.model_type,
            num_classes=args.num_classes,
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
        ).to(args.device)

        optimizer = build_optimizer(
            model=model, lr=lr, weight_decay=weight_decay, model_type=args.model_type, optimizer=optimizer_name
        )

        scheduler = build_scheduler(optimizer, scheduler_type=args.scheduler)
        criterion = build_criterion(args.num_classes, pos_weight=args.pos_weight)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=args.device,
            num_classes=args.num_classes,
            epochs=5,
            patience=3,
            run_dir=os.path.join(args.run_dir, f"trial_{trial.number}"),
        )

        history = trainer.fit(img_size=args.img_size)
        val_dice = history["val"]["dice"]
        return max(val_dice) if val_dice else 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.optuna_trials)

    summary = {
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_trial.value,
            "params": study.best_trial.params,
        },
        "all_trials": [
            {"number": t.number, "value": t.value, "params": t.params, "state": str(t.state)} for t in study.trials
        ],
    }

    with open(os.path.join(args.run_dir, "optuna_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n[Search Run Complete]")
    print(f"Optimal Value (Dice Score): {study.best_trial.value:.2f}%")
