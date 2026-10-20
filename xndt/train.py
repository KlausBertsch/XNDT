"""Reproducible from-scratch train/validation training for XNDT RouteNet-Fermi."""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
from collections import defaultdict
from pathlib import Path

import keras
import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xndt.data import (
    DEMAND_ORDER,
    adapt_sample,
    compute_train_normalization,
    read_jsonl,
)
from xndt.model import RouteNet_Fermi

FORBIDDEN = {
    "loss_ratio",
    "tx_packets",
    "rx_packets",
    "loss_packets",
    "rho_wire",
    "alpha",
    "control_wire_rate_gbps",
    "forward_loss_ratio",
    "qfx_drops",
    "queue_occupancy",
    "measured_utilization",
    "telemetry",
}


def arguments():
    """Parse dataset, output, and optimization settings."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    return p.parse_args()


def seed_all(seed):
    """Seed all random sources and request deterministic TensorFlow operations."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def paths(root):
    """Resolve and validate the fixed training and validation files."""
    train, validation = root / "train.jsonl", root / "validation.jsonl"
    for path in (train, validation):
        if not path.is_file():
            raise FileNotFoundError(path)
    return train, validation


def rows(path):
    """Load a non-empty JSONL split into memory."""
    result = list(read_jsonl(path))
    if not result:
        raise ValueError(f"Empty split: {path}")
    return result


def sha256(path):
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    """Return the current Git commit when the repository is available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate_splits(train, val):
    """Check split sizes, scenario separation, and repetition counts."""
    if (len(train), len(val)) != (294, 42):
        raise ValueError("Expected 294 train and 42 validation runs")
    tk, vk = {x["scenario_key"] for x in train}, {x["scenario_key"] for x in val}
    if len(tk) != 98 or len(vk) != 14 or tk & vk:
        raise ValueError("Invalid scenario-level split")
    for data in (train, val):
        count = defaultdict(int)
        for x in data:
            count[x["scenario_key"]] += 1
        if set(count.values()) != {3}:
            raise ValueError("Expected three repetitions per scenario")


def samples(raw):
    """Adapt raw runs for training while rejecting label leakage."""
    result = []
    for row in raw:
        x, y, _ = adapt_sample(row, True)
        found = FORBIDDEN & set(x)
        if found:
            raise ValueError(f"Leakage inputs: {sorted(found)}")
        result.append((x, y))
    return result


def metric(y, p):
    """Return MAE, MSE, and RMSE for predictions and targets."""
    e = np.asarray(p, np.float64).reshape(-1) - np.asarray(y, np.float64).reshape(-1)
    mse = float(np.mean(e**2))
    return {"MAE": float(np.mean(abs(e))), "MSE": mse, "RMSE": float(np.sqrt(mse))}


class Trainer:
    """Own the optimizer, loss function, and graph-level training operations."""

    def __init__(self, model, lr):
        """Configure Adam and binary cross-entropy for the supplied model."""
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    @tf.function(reduce_retracing=True)
    def update(self, x, y):
        """Apply one gradient update for a single graph sample."""
        y = tf.reshape(tf.cast(y, tf.float32), (-1, 1))
        with tf.GradientTape() as tape:
            loss = self.loss_fn(y, self.model(x, training=True))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        if any(g is None for g in gradients):
            raise ValueError("None gradient")
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    @tf.function(reduce_retracing=True)
    def predict(self, x):
        """Run inference without enabling training behavior."""
        return self.model(x, training=False)

    def evaluate(self, data):
        """Evaluate BCE and regression metrics over an adapted split."""
        losses, ys, ps = [], [], []
        for x, y in data:
            p = self.predict(x)
            losses.append(float(self.loss_fn(tf.reshape(y, (-1, 1)), p)))
            ys.append(np.asarray(y))
            ps.append(np.asarray(p).reshape(-1))
        result = metric(np.concatenate(ys), np.concatenate(ps))
        result["BCE"] = float(np.mean(losses))
        for i, d in enumerate(DEMAND_ORDER):
            result.update(
                {
                    f"{d}_{k}": v
                    for k, v in metric([y[i] for y in ys], [p[i] for p in ps]).items()
                }
            )
        return result, ps


def train(trainer, train_data, val_data, out, seed, max_epochs, patience):
    """Train with deterministic shuffling and validation-based early stopping.

    The best validation checkpoint is restored before returning.
    """
    rng = np.random.default_rng(seed)
    history = []
    best = float("inf")
    best_epoch = stale = 0
    checkpoint = out / "best.weights.h5"
    for epoch in range(1, max_epochs + 1):
        for i in rng.permutation(len(train_data)):
            trainer.update(*train_data[i])
        tr, _ = trainer.evaluate(train_data)
        va, _ = trainer.evaluate(val_data)
        record = {
            "epoch": epoch,
            "train_bce": tr["BCE"],
            "validation_bce": va["BCE"],
            "train_mae": tr["MAE"],
            "validation_mae": va["MAE"],
            "train_rmse": tr["RMSE"],
            "validation_rmse": va["RMSE"],
            "learning_rate": float(trainer.optimizer.learning_rate.numpy()),
        }
        history.append(record)
        print(
            f"epoch={epoch:03d} train_BCE={tr['BCE']:.6f} validation_BCE={va['BCE']:.6f} train_MAE={tr['MAE']:.6f} validation_MAE={va['MAE']:.6f}",
            flush=True,
        )
        if va["BCE"] < best:
            best, best_epoch, stale = va["BCE"], epoch, 0
            trainer.model.save_weights(checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    trainer.model.load_weights(checkpoint)
    return history, best_epoch


def dump(path, value):
    """Write a JSON artifact with stable formatting."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_csv(path, data):
    """Write result dictionaries to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def history_outputs(out, history):
    """Persist the training history and its BCE and MAE curves."""
    write_csv(out / "training_history.csv", history)
    e = [x["epoch"] for x in history]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(e, [x["train_bce"] for x in history], label="train")
    ax[0].plot(e, [x["validation_bce"] for x in history], label="validation")
    ax[0].set(xlabel="Epoch", ylabel="BCE")
    ax[0].legend()
    ax[1].plot(e, [x["train_mae"] for x in history], label="train")
    ax[1].plot(e, [x["validation_mae"] for x in history], label="validation")
    ax[1].set(xlabel="Epoch", ylabel="MAE")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "training_history.png", dpi=150)
    plt.close(fig)


def capacity_prediction(x):
    """Estimate path loss with a simple link-capacity overload heuristic."""
    traffic = np.asarray(x["traffic"]).reshape(-1)
    capacity = np.asarray(x["capacity"]).reshape(-1)
    users = x["path_to_link"].to_list()
    offered = np.array([sum(traffic[p] for p, _ in u) for u in users])
    link_loss = np.maximum(
        0,
        1
        - np.divide(
            capacity, offered, out=np.full_like(capacity, np.inf), where=offered > 0
        ),
    )
    return np.array(
        [max(link_loss[l] for l in path) for path in np.asarray(x["link_to_path"])]
    )


def diagnostics(raw, data, predictions):
    """Combine labels, predictions, baselines, and scenario metadata per demand."""
    result = []
    for row, (x, y), p in zip(raw, data, predictions):
        baseline = capacity_prediction(x)
        meta = row["metadata"]
        for i, d in enumerate(DEMAND_ORDER):
            measured, predicted = float(y[i]), float(p[i])
            result.append(
                {
                    "run_id": row["run_id"],
                    "scenario_key": row["scenario_key"],
                    "repetition_index": row["repetition_index"],
                    "rho_wire": meta["rho_wire"],
                    "alpha": meta["alpha"],
                    "control_wire_rate_gbps": meta["control_wire_rate_gbps"],
                    "demand_id": d,
                    "measured_loss_ratio": measured,
                    "predicted_loss_ratio": predicted,
                    "absolute_error": abs(predicted - measured),
                    "always_zero_prediction": 0.0,
                    "capacity_baseline_prediction": float(baseline[i]),
                }
            )
    return result


def grouped(data, key):
    """Calculate metrics overall and separately for each demand."""
    categories = {
        "overall": {"all": data},
        "per_demand": {
            d: [x for x in data if x["demand_id"] == d] for d in DEMAND_ORDER
        },
    }
    result = {k: {} for k in categories}
    for kind, groups in categories.items():
        for name, selected in groups.items():
            result[kind][name] = {
                k: v
                for k, v in metric(
                    [x["measured_loss_ratio"] for x in selected],
                    [x[key] for x in selected],
                ).items()
                if k != "MSE"
            }
    return result


def subsets(data, key):
    """Calculate metrics for zero-loss and positive-loss subsets."""
    result = {}
    for name, selected in (
        ("zero", [x for x in data if x["measured_loss_ratio"] == 0]),
        ("positive", [x for x in data if x["measured_loss_ratio"] > 0]),
    ):
        value = metric(
            [x["measured_loss_ratio"] for x in selected], [x[key] for x in selected]
        )
        result[name] = {
            "count": len(selected),
            "MAE": value["MAE"],
            "RMSE": value["RMSE"],
            "mean_prediction": float(np.mean([x[key] for x in selected])),
        }
    return result


def scenario_rows(data):
    """Aggregate repeated validation runs by scenario and demand."""
    groups = defaultdict(list)
    for x in data:
        groups[(x["scenario_key"], x["demand_id"])].append(x)
    result = []
    for (scenario, demand), values in sorted(groups.items()):
        m = np.array([x["measured_loss_ratio"] for x in values])
        p = np.array([x["predicted_loss_ratio"] for x in values])
        x = values[0]
        result.append(
            {
                "scenario_key": scenario,
                "rho_wire": x["rho_wire"],
                "alpha": x["alpha"],
                "control_wire_rate_gbps": x["control_wire_rate_gbps"],
                "demand_id": demand,
                "measured_loss_mean": float(m.mean()),
                "measured_loss_std": float(m.std()),
                "predicted_loss_mean": float(p.mean()),
                "absolute_error": float(abs(m.mean() - p.mean())),
            }
        )
    return result


def rho_rows(data):
    """Aggregate scenario-level validation results by offered load."""
    result = []
    for rho in sorted({x["rho_wire"] for x in data}):
        selected = [x for x in data if x["rho_wire"] == rho]
        m = [x["measured_loss_mean"] for x in selected]
        p = [x["predicted_loss_mean"] for x in selected]
        value = metric(m, p)
        result.append(
            {
                "rho_wire": rho,
                "n_scenarios": len({x["scenario_key"] for x in selected}),
                "measured_loss_mean": float(np.mean(m)),
                "predicted_loss_mean": float(np.mean(p)),
                "MAE": value["MAE"],
                "RMSE": value["RMSE"],
            }
        )
    return result


def plots(out, data):
    """Create validation prediction, load, and error figures."""
    measured = np.array([x["measured_loss_ratio"] for x in data])
    predicted = np.array([x["predicted_loss_ratio"] for x in data])
    rho = np.array([x["rho_wire"] for x in data])
    baseline = np.array([x["capacity_baseline_prediction"] for x in data])
    fig, ax = plt.subplots()
    ax.scatter(measured, predicted, alpha=0.6)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="Measured loss ratio", ylabel="Predicted loss ratio")
    fig.tight_layout()
    fig.savefig(out / "validation_measured_vs_predicted.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.scatter(rho, measured, s=16, alpha=0.5, label="Measured")
    ax.scatter(rho, predicted, s=16, alpha=0.5, label="RouteNet-Fermi")
    ax.scatter(rho, baseline, s=16, alpha=0.5, label="Capacity baseline")
    ax.set(xlabel="rho_wire", ylabel="Loss ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "validation_loss_vs_rho.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots()
    for d in DEMAND_ORDER:
        selected = [x for x in data if x["demand_id"] == d]
        ax.scatter(
            [x["rho_wire"] for x in selected],
            [x["absolute_error"] for x in selected],
            s=16,
            alpha=0.6,
            label=d,
        )
    ax.set(xlabel="rho_wire", ylabel="Absolute error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "validation_error_vs_rho.png", dpi=150)
    plt.close(fig)


def make_model(normalization):
    """Build the XNDT model with training-split normalization statistics."""
    return RouteNet_Fermi(xndt_mode=True, normalization=normalization["features"])


def main():
    """Run deterministic training and write the validation audit artifacts."""
    args = arguments()
    if min(args.max_epochs, args.patience, args.learning_rate) <= 0:
        raise ValueError("Training parameters must be positive")
    seed_all(args.seed)
    train_path, val_path = paths(args.dataset_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalization_path = args.output_dir / "xndt_normalization.json"
    normalization = compute_train_normalization(args.dataset_dir, normalization_path)
    train_raw, val_raw = rows(train_path), rows(val_path)
    validate_splits(train_raw, val_raw)
    train_data, val_data = samples(train_raw), samples(val_raw)
    config = {
        "model": "RouteNet_Fermi",
        "xndt_mode": True,
        "training_from_scratch": True,
        "state_dimensions": {"path": 32, "link": 32, "queue": 32},
        "message_passing_iterations": 8,
        "output_activation": "sigmoid",
        "batching": "one graph per gradient update; no padding",
    }
    dump(args.output_dir / "model_config.json", config)
    trainer = Trainer(make_model(normalization), args.learning_rate)
    history, best_epoch = train(
        trainer,
        train_data,
        val_data,
        args.output_dir,
        args.seed,
        args.max_epochs,
        args.patience,
    )
    history_outputs(args.output_dir, history)
    train_result, _ = trainer.evaluate(train_data)
    val_result, predictions = trainer.evaluate(val_data)
    diag = diagnostics(val_raw, val_data, predictions)
    write_csv(args.output_dir / "validation_predictions.csv", diag)
    scenario = scenario_rows(diag)
    write_csv(args.output_dir / "validation_scenario_predictions.csv", scenario)
    rho = rho_rows(scenario)
    write_csv(args.output_dir / "validation_metrics_by_rho.csv", rho)
    plots(args.output_dir, diag)
    comparisons = {
        "RouteNet-Fermi": grouped(diag, "predicted_loss_ratio"),
        "Always-zero": grouped(diag, "always_zero_prediction"),
        "Capacity-overload heuristic": grouped(diag, "capacity_baseline_prediction"),
    }
    subset = {
        name: subsets(diag, key)
        for name, key in (
            ("RouteNet-Fermi", "predicted_loss_ratio"),
            ("Always-zero", "always_zero_prediction"),
            ("Capacity-overload heuristic", "capacity_baseline_prediction"),
        )
    }
    manifest = args.dataset_dir / "dataset_manifest.json"
    scenario_metric = metric(
        [x["measured_loss_mean"] for x in scenario],
        [x["predicted_loss_mean"] for x in scenario],
    )
    summary = {
        "seed": args.seed,
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "git_commit": git_commit(),
        "dataset_manifest": str(manifest),
        "dataset_manifest_sha256": sha256(manifest),
        "normalization_file": str(normalization_path),
        "normalization_sha256": sha256(normalization_path),
        "train_samples": 294,
        "train_scenarios": 98,
        "validation_samples": 42,
        "validation_scenarios": 14,
        "test_split_usage": "not opened or evaluated",
        "model_config": config,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "optimization_loss": "BinaryCrossentropy(from_logits=False)",
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "epochs_trained": len(history),
        "best_validation_epoch": best_epoch,
        "train_at_best_weights": train_result,
        "validation_at_best_weights": val_result,
        "validation_scenario_level": {
            "MAE": scenario_metric["MAE"],
            "RMSE": scenario_metric["RMSE"],
        },
        "validation_subsets": subset,
        "validation_comparisons": comparisons,
        "validation_by_rho": rho,
    }
    dump(args.output_dir / "training_summary.json", summary)
    print("training_summary:", json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
