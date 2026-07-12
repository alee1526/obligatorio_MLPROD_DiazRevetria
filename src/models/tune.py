import argparse
import json
import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import shutil
from dataclasses import replace
from datetime import datetime

import mlflow
import optuna

from src.models.config import Config
from src.models.train import MLFLOW_URI, MODELS_DIR, get_device, train_run

OPTUNA_DIR = MODELS_DIR / "optuna"


def suggest_config(trial, base):
    return replace(
        base,
        backbone=trial.suggest_categorical(
            "backbone", ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3"]),
        lr=trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        dropout=trial.suggest_float("dropout", 0.1, 0.6),
        unfreeze_blocks=trial.suggest_int("unfreeze_blocks", 0, 4),
        tab_hidden=trial.suggest_categorical("tab_hidden", [32, 64, 128]),
        tab_out=trial.suggest_categorical("tab_out", [16, 32, 64]),
        fusion_hidden=trial.suggest_categorical("fusion_hidden", [64, 128, 256]),
        batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]),
        aug_rotation=trial.suggest_int("aug_rotation", 0, 30),
        aug_color=trial.suggest_float("aug_color", 0.0, 0.3),
        aug_erase=trial.suggest_float("aug_erase", 0.0, 0.4),
        loss=trial.suggest_categorical("loss", ["weighted_ce", "focal"]),
        focal_gamma=trial.suggest_float("focal_gamma", 1.0, 3.0),
    )


def objective(trial, base, device, run_dir):
    cfg = suggest_config(trial, base)
    path = run_dir / f"trial_{trial.number + 1:03d}.pt"
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number + 1:03d}"):
        mlflow.log_params(cfg.as_dict())

        def on_epoch(epoch, m):
            trial.report(m["macro_f1"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        best_f1 = train_run(cfg, device, save_path=path, log=True, on_epoch=on_epoch)
        mlflow.log_metric("best_val_macro_f1", best_f1)
        if path.exists():
            scored = path.with_name(f"trial_{trial.number + 1:03d}_f1-{best_f1:.3f}.pt")
            path.rename(scored)
            mlflow.set_tag("model_file", scored.name)
            mlflow.log_artifact(str(scored))
    return best_f1


def main():
    base = Config.load()
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=base.n_trials)
    n_trials = parser.parse_args().trials

    device = get_device()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = OPTUNA_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print("device:", device, "| trials:", n_trials, "| corrida:", stamp)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("skin-lesion-tuning")
    storage = f"sqlite:///{run_dir / 'study.db'}"
    study = optuna.create_study(direction="maximize", study_name=stamp, storage=storage,
                                pruner=optuna.pruners.MedianPruner())

    with mlflow.start_run(run_name=f"optuna-{stamp}"):
        mlflow.set_tag("run_dir", str(run_dir))
        study.optimize(lambda t: objective(t, base, device, run_dir), n_trials=n_trials)
        mlflow.log_metric("best_value", study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})

        best_n = study.best_trial.number + 1
        matches = list(run_dir.glob(f"trial_{best_n:03d}_*.pt"))
        if matches:
            shutil.copy(matches[0], run_dir / "best.pt")
        study.trials_dataframe().to_csv(run_dir / "summary.csv", index=False)
        (run_dir / "best_params.json").write_text(
            json.dumps({"value": study.best_value, "params": study.best_params}, indent=2))
        for f in ("best.pt", "summary.csv", "best_params.json"):
            if (run_dir / f).exists():
                mlflow.log_artifact(str(run_dir / f))

    print(f"\ncorrida: {run_dir}")
    print(f"mejor macro-F1: {study.best_value:.3f} (trial {best_n:03d} -> best.pt)")
    print("mejores hiperparámetros:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("copiar best_params.json al config.yaml para reentrenar el modelo final")


if __name__ == "__main__":
    main()
