import json
from pathlib import Path

import optuna
from tqdm.auto import tqdm

from src.dataset import get_dataloaders
from src.factory import build_pipeline
from src.trainer import Trainer


def run_optuna_study(**kwargs):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
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

        trial_kwargs["epochs"] = kwargs.get("optuna_epochs", 10)
        trial_kwargs["patience"] = kwargs.get("optuna_patience", 5)
        trial_kwargs["run_dir"] = str(run_directory / f"trial_{trial.number}")

        train_loader, val_loader = get_dataloaders(limit_dataset=512, **trial_kwargs)
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
    total_trials = kwargs.get("optuna_trials", 30)

    with tqdm(total=total_trials, desc="Optuna Trials", position=0) as progress_bar:

        def tqdm_callback(active_study, active_trial):
            progress_bar.update(1)

        study.optimize(objective, n_trials=total_trials, callbacks=[tqdm_callback])

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

    cli_arguments = []
    for parameter_key, parameter_value in study.best_params.items():
        if isinstance(parameter_value, float):
            cli_arguments.append(f"--{parameter_key} {parameter_value:.5f}")
        else:
            cli_arguments.append(f"--{parameter_key} {parameter_value}")

    command_string = " ".join(cli_arguments)

    print(f"\n============================================================")
    print(f"OPTIMIZATION COMPLETE | Best Validation Dice: {study.best_value:.2f}%")
    print(f"============================================================")
    print(f"Execute this command for full-scale training:\n")
    print(f"python main.py --mode train --model_type {kwargs.get('model_type')} {command_string}\n")
