"""One-time evaluation of the frozen XNDT checkpoint on test.jsonl."""

import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xndt.data import DEMAND_ORDER, adapt_sample, read_jsonl
from xndt.model import RouteNet_Fermi
from xndt.train import capacity_prediction, metric

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


# Parse the paths and seed used for frozen test evaluation.
def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--training-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# Return the SHA-256 digest of a file.
def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# Seed all random sources and request deterministic TensorFlow operations.
def seed_all(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


# Write result dictionaries to a CSV file.
def write_csv(path, data):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


# Verify the declared test split, repetitions, and scenario separation.
def validate_split_metadata(dataset_dir, test_rows):
    split = json.loads((dataset_dir / "dataset_split.json").read_text())["scenarios"]
    keys = defaultdict(set)
    for x in split:
        keys[x["split"]].add(x["scenario_key"])
    test_keys = {x["scenario_key"] for x in test_rows}
    if len(test_rows) != 42 or len(test_keys) != 14 or test_keys != keys["test"]:
        raise ValueError("Expected 42 test runs and 14 declared test scenarios")
    if keys["test"] & keys["train"] or keys["test"] & keys["validation"]:
        raise ValueError("Test scenario overlap")
    counts = defaultdict(list)
    for x in test_rows:
        counts[x["scenario_key"]].append(x["repetition_index"])
    if any(sorted(v) != [0, 1, 2] for v in counts.values()):
        raise ValueError("Expected repetitions 0,1,2 per test scenario")


# Serialize dense or ragged tensors for deterministic hashing.
def tensor_bytes(value):
    if isinstance(value, tf.RaggedTensor):
        return value.flat_values.numpy().tobytes() + value.row_splits.numpy().tobytes()
    return np.asarray(value).tobytes()


# Hash a complete model-input dictionary in stable key order.
def input_fingerprint(inputs):
    h = hashlib.sha256()
    for name in sorted(inputs):
        h.update(name.encode())
        h.update(tensor_bytes(inputs[name]))
    return h.hexdigest()


# Hash all model weights to detect changes during evaluation.
def weight_fingerprint(model):
    h = hashlib.sha256()
    for value in model.weights:
        h.update(value.path.encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


# Run the frozen model and collect one diagnostic row per demand.
def evaluate_model(model, raw):
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    result = []
    bces = []
    fingerprints = defaultdict(set)
    for row in raw:
        inputs, target, _ = adapt_sample(row, True)
        found = FORBIDDEN & set(inputs)
        if found:
            raise ValueError(f"Leakage inputs: {sorted(found)}")
        prediction = np.asarray(model(inputs, training=False)).reshape(-1)
        target = np.asarray(target).reshape(-1)
        bces.append(float(loss_fn(target.reshape(-1, 1), prediction.reshape(-1, 1))))
        fingerprints[row["scenario_key"]].add(input_fingerprint(inputs))
        baseline = capacity_prediction(inputs)
        meta = row["metadata"]
        for i, demand in enumerate(DEMAND_ORDER):
            result.append(
                {
                    "run_id": row["run_id"],
                    "scenario_key": row["scenario_key"],
                    "repetition_index": row["repetition_index"],
                    "rho_wire": meta["rho_wire"],
                    "alpha": meta["alpha"],
                    "control_wire_rate_gbps": meta["control_wire_rate_gbps"],
                    "demand_id": demand,
                    "measured_loss_ratio": float(target[i]),
                    "predicted_loss_ratio": float(prediction[i]),
                    "absolute_error": float(abs(target[i] - prediction[i])),
                    "always_zero_prediction": 0.0,
                    "capacity_baseline_prediction": float(baseline[i]),
                }
            )
    return result, float(np.mean(bces)), max(len(v) for v in fingerprints.values())


# Compute overall, per-demand, and loss-regime metrics for one predictor.
def metrics_for(data, key):
    overall = metric([x["measured_loss_ratio"] for x in data], [x[key] for x in data])
    per = {}
    for d in DEMAND_ORDER:
        s = [x for x in data if x["demand_id"] == d]
        per[d] = {
            k: v
            for k, v in metric(
                [x["measured_loss_ratio"] for x in s], [x[key] for x in s]
            ).items()
            if k != "MSE"
        }
    subsets = {}
    for name, s in (
        ("zero", [x for x in data if x["measured_loss_ratio"] == 0]),
        ("positive", [x for x in data if x["measured_loss_ratio"] > 0]),
    ):
        value = metric([x["measured_loss_ratio"] for x in s], [x[key] for x in s])
        subsets[name] = {
            "count": len(s),
            "MAE": value["MAE"],
            "RMSE": value["RMSE"],
            "mean_measured_loss": float(np.mean([x["measured_loss_ratio"] for x in s])),
            "mean_predicted_loss": float(np.mean([x[key] for x in s])),
        }
    return {
        "overall": {"MAE": overall["MAE"], "RMSE": overall["RMSE"]},
        "per_demand": per,
        "subsets": subsets,
    }


# Aggregate the three repetitions of each scenario and demand.
def scenario_rows(data):
    groups = defaultdict(list)
    for x in data:
        groups[(x["scenario_key"], x["demand_id"])].append(x)
    result = []
    for (scenario, demand), values in sorted(groups.items()):
        measured = np.array([x["measured_loss_ratio"] for x in values])
        predicted = np.array([x["predicted_loss_ratio"] for x in values])
        x = values[0]
        result.append(
            {
                "scenario_key": scenario,
                "rho_wire": x["rho_wire"],
                "alpha": x["alpha"],
                "control_wire_rate_gbps": x["control_wire_rate_gbps"],
                "demand_id": demand,
                "measured_loss_mean": float(measured.mean()),
                "measured_loss_std": float(measured.std()),
                "predicted_loss_mean": float(predicted.mean()),
                "prediction_std": float(predicted.std()),
                "absolute_error": float(abs(measured.mean() - predicted.mean())),
            }
        )
    return result


# Calculate scenario-level metrics overall and per demand.
def scenario_metrics(rows):
    # Calculate MAE and RMSE for a selected set of scenario rows.
    def one(selected):
        value = metric(
            [x["measured_loss_mean"] for x in selected],
            [x["predicted_loss_mean"] for x in selected],
        )
        return {"MAE": value["MAE"], "RMSE": value["RMSE"]}

    return {
        "overall": one(rows),
        "per_demand": {
            d: one([x for x in rows if x["demand_id"] == d]) for d in DEMAND_ORDER
        },
    }


# Aggregate scenario predictions and errors by offered load.
def rho_rows(rows):
    result = []
    for rho in sorted({x["rho_wire"] for x in rows}):
        selected = [x for x in rows if x["rho_wire"] == rho]
        m = [x["measured_loss_mean"] for x in selected]
        p = [x["predicted_loss_mean"] for x in selected]
        value = metric(m, p)
        row = {
            "rho_wire": rho,
            "measured_loss_mean": float(np.mean(m)),
            "predicted_loss_mean": float(np.mean(p)),
            "MAE": value["MAE"],
            "RMSE": value["RMSE"],
        }
        for d in DEMAND_ORDER:
            s = [x for x in selected if x["demand_id"] == d]
            row[f"{d}_measured_loss_mean"] = s[0]["measured_loss_mean"]
            row[f"{d}_predicted_loss_mean"] = s[0]["predicted_loss_mean"]
            row[f"{d}_absolute_error"] = s[0]["absolute_error"]
        result.append(row)
    return result


# Create the measured-versus-predicted and load-dependent result figures.
def plots(out, data):
    m = np.array([x["measured_loss_ratio"] for x in data])
    p = np.array([x["predicted_loss_ratio"] for x in data])
    rho = np.array([x["rho_wire"] for x in data])
    fig, ax = plt.subplots()
    ax.scatter(m, p, alpha=0.6)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="Measured loss ratio", ylabel="Predicted loss ratio")
    fig.tight_layout()
    fig.savefig(out / "test_measured_vs_predicted.png", dpi=150)
    plt.close(fig)

    model_metrics = metrics_for(data, "predicted_loss_ratio")
    capacity_metrics = metrics_for(data, "capacity_baseline_prediction")
    zero_metrics = metrics_for(data, "always_zero_prediction")
    displayed = [x for x in data if x["demand_id"] in ("d1", "d2")]
    display_m = np.array([x["measured_loss_ratio"] for x in displayed])
    display_p = np.array([x["predicted_loss_ratio"] for x in displayed])
    display_rho = np.array([x["rho_wire"] for x in displayed])
    fig, ax = plt.subplots(figsize=(10, 7.2))
    ax.axvspan(1.0, rho.max() + 0.06, color="#f59e0b", alpha=0.055, zorder=0)
    ax.axvline(1.0, color="#64748b", linestyle="--", linewidth=1.35, zorder=1)
    ax.scatter(
        display_rho,
        display_m,
        s=28,
        alpha=0.62,
        label="Measured",
        color="#1f77b4",
        edgecolors="none",
        zorder=3,
    )
    ax.scatter(
        display_rho,
        display_p,
        s=28,
        alpha=0.78,
        label="RouteNet-Fermi",
        color="#ff7f0e",
        edgecolors="none",
        zorder=4,
    )
    ax.annotate(
        "Bottleneck capacity",
        xy=(1.0, 1),
        xycoords=("data", "axes fraction"),
        xytext=(7, -7),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=9,
        color="#475569",
    )
    ax.annotate(
        "Per-demand points: d1 and d2 (d3 omitted); metrics: all test targets",
        xy=(1, 0),
        xycoords="axes fraction",
        xytext=(-5, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#64748b",
    )
    ax.set_title(
        "RouteNet-Fermi reproduces measured packet loss on unseen scenarios",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel(r"Offered bottleneck load $\rho$", fontsize=12)
    ax.set_ylabel("Packet-loss ratio", fontsize=12)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(axis="y", color="#cbd5e1", linewidth=0.7, alpha=0.45)
    ax.set_axisbelow(True)
    metrics_text = (
        f"Positive-loss MAE {model_metrics['subsets']['positive']['MAE']:.4f}  ·  "
        f"MAE {model_metrics['overall']['MAE']:.4f}  ·  RMSE {model_metrics['overall']['RMSE']:.4f}\n"
        f"Capacity baseline: {capacity_metrics['subsets']['positive']['MAE']:.4f}  ·  "
        f"Always-zero: {zero_metrics['subsets']['positive']['MAE']:.4f}"
    )
    fig.text(
        0.5,
        0.025,
        metrics_text,
        ha="center",
        va="bottom",
        fontsize=9.5,
        linespacing=1.55,
        bbox={
            "boxstyle": "round,pad=.55",
            "facecolor": "#f8fafc",
            "edgecolor": "#cbd5e1",
            "linewidth": 0.8,
        },
    )
    fig.subplots_adjust(left=0.105, right=0.975, top=0.88, bottom=0.19)
    fig.savefig(out / "test_loss_vs_rho.png", dpi=180, facecolor="white")
    plt.close(fig)


# Evaluate the published checkpoint and write frozen test artifacts.
def main():
    args = arguments()
    seed_all(args.seed)
    required = {
        "checkpoint": args.training_dir / "best.weights.h5",
        "model_config": args.training_dir / "model_config.json",
        "normalization": args.training_dir / "xndt_normalization.json",
        "test_jsonl": args.dataset_dir / "test.jsonl",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    hashes = {name: sha256(path) for name, path in required.items()}
    normalization = json.loads(required["normalization"].read_text())
    config = json.loads(required["model_config"].read_text())
    raw = list(read_jsonl(required["test_jsonl"]))
    validate_split_metadata(args.dataset_dir, raw)
    model = RouteNet_Fermi(xndt_mode=True, normalization=normalization["features"])
    first_inputs, _, _ = adapt_sample(raw[0], True)
    model(first_inputs, training=False)
    model.load_weights(required["checkpoint"])
    before = weight_fingerprint(model)
    predictions, bce, max_input_variants = evaluate_model(model, raw)
    after = weight_fingerprint(model)
    if before != after:
        raise AssertionError("Frozen model weights changed during test evaluation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "test_predictions.csv", predictions)
    scenarios = scenario_rows(predictions)
    write_csv(args.output_dir / "test_scenario_predictions.csv", scenarios)
    rho = rho_rows(scenarios)
    all_metrics = {
        "RouteNet-Fermi": metrics_for(predictions, "predicted_loss_ratio"),
        "Capacity-overload heuristic": metrics_for(
            predictions, "capacity_baseline_prediction"
        ),
        "Always-zero": metrics_for(predictions, "always_zero_prediction"),
    }
    plots(args.output_dir, predictions)
    d3 = [x for x in predictions if x["demand_id"] == "d3"]
    validation = json.loads((args.training_dir / "training_summary.json").read_text())
    stability = {
        "max_prediction_std_across_repetitions": max(
            x["prediction_std"] for x in scenarios
        ),
        "max_measured_label_std_across_repetitions": max(
            x["measured_loss_std"] for x in scenarios
        ),
        "max_distinct_input_fingerprints_per_scenario": max_input_variants,
    }
    summary = {
        "seed": args.seed,
        "hashes": hashes,
        "model_config": config,
        "test_runs": 42,
        "test_scenarios": 14,
        "test_targets": 126,
        "weights_unchanged": before == after,
        "run_level": {"BCE": bce, **all_metrics["RouteNet-Fermi"]["overall"]},
        "RouteNet-Fermi": all_metrics["RouteNet-Fermi"],
        "baselines": {
            "Capacity-overload heuristic": all_metrics["Capacity-overload heuristic"],
            "Always-zero": all_metrics["Always-zero"],
        },
        "scenario_level": scenario_metrics(scenarios),
        "by_rho": rho,
        "d3_negative_control": {
            "mean_measured_loss": float(
                np.mean([x["measured_loss_ratio"] for x in d3])
            ),
            "mean_predicted_loss": float(
                np.mean([x["predicted_loss_ratio"] for x in d3])
            ),
            "max_predicted_loss": float(
                np.max([x["predicted_loss_ratio"] for x in d3])
            ),
            **all_metrics["RouteNet-Fermi"]["per_demand"]["d3"],
        },
        "repetition_stability": stability,
        "validation_comparison": {
            "validation_MAE": validation["validation_at_best_weights"]["MAE"],
            "validation_RMSE": validation["validation_at_best_weights"]["RMSE"],
            "validation_positive_MAE": validation["validation_subsets"][
                "RouteNet-Fermi"
            ]["positive"]["MAE"],
            "validation_scenario_MAE": validation["validation_scenario_level"]["MAE"],
        },
    }
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
