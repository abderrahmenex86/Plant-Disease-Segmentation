import json
from pathlib import Path

import optuna

from src.dataset import get_dataloaders
from src.factory import build_pipeline
from src.trainer import Trainer


def run_optuna_study(**kwargs):
    optuna.logging.set_verbosity(optuna.logging.INFO)
    run_directory = Path(kwargs.get("run_dir", "artifacts/default_optuna"))
    run_directory.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        suggested_learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        suggested_weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        suggested_optimizer_type = trial.suggest_categorical("optimizer_type", ["AdamW", "Adam", "SGD"])

        trial_kwargs = kwargs.copy()
        trial_kwargs["learning_rate"] = suggested_learning_rate
        trial_kwargs["weight_decay"] = suggested_weight_decay
        trial_kwargs["optimizer_type"] = suggested_optimizer_type

        if kwargs.get("model_type") == "unet":
            suggested_base_channels = trial.suggest_categorical("base_channels", [32, 64])
            trial_kwargs["base_channels"] = suggested_base_channels

        trial_kwargs["epochs"] = kwargs.get("optuna_epochs", 5)
        trial_kwargs["patience"] = kwargs.get("optuna_patience", 2)
        trial_kwargs["run_dir"] = str(run_directory / f"trial_{trial.number}")

        train_loader, val_loader = get_dataloaders(limit_dataset=200, **trial_kwargs)
        model, criterion, optimizer, scheduler = build_pipeline(**trial_kwargs)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **trial_kwargs,
        )

        history = trainer.fit()
        validation_dice_history = history["val"]["dice"]
        return max(validation_dice_history) if validation_dice_history else 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=kwargs.get("optuna_trials", 15))

    summary_data = {
        "best_trial_value": study.best_value,
        "best_hyperparameters": study.best_params,
        "all_trials": [
            {
                "trial_number": trial.number,
                "trial_value": trial.value,
                "hyperparameters": trial.params,
                "trial_state": str(trial.state),
            }
            for trial in study.trials
        ],
    }

    (run_directory / "optuna_summary.json").write_text(json.dumps(summary_data, indent=4))
