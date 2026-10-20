"""Final, frozen 512-interval Integrated Gradients evaluation on all test scenarios."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from xndt.data import DEMAND_ORDER, adapt_sample, read_jsonl
from xndt.ig_convergence import with_composite_rates
from xndt.integrated_gradients import (
    COMPLETENESS_TOLERANCE,
    FINAL_IG_STEPS,
    occlusion,
    output,
    ranking,
    seed_all,
    select_low_load,
    traffic_and_ratio,
    validate_inputs,
    weight_fingerprint,
)
from xndt.model import RouteNet_Fermi


assert FINAL_IG_STEPS == 512


def arguments():
    """Parse paths and the seed for the frozen test attribution run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sha256(path):
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, data):
    """Write result dictionaries to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def select_test_representatives(dataset_dir):
    """Select one deterministic representative from each test scenario."""
    raw = list(read_jsonl(dataset_dir / "test.jsonl"))
    groups = defaultdict(list)
    for row in raw:
        groups[row["scenario_key"]].append(row)

    invalid_size = (
        len(raw) != 42
        or len(groups) != 14
        or any(len(values) != 3 for values in groups.values())
    )
    if invalid_size:
        raise ValueError("Expected 42 test runs, 14 scenarios and three repetitions")

    return [
        min(
            values,
            key=lambda row: (row["repetition_index"], row["run_id"]),
        )
        for _, values in sorted(groups.items())
    ]


@tf.function(reduce_retracing=True)
def gradient_grid_512(model, inputs, baseline, actual, ratios, target_index):
    """Evaluate all 513 gradients required by the frozen test protocol."""
    alphas = tf.linspace(0.0, 1.0, FINAL_IG_STEPS + 1)

    def one(alpha):
        """Evaluate one gradient point on the straight attribution path."""
        rates = baseline + alpha * (actual - baseline)
        with tf.GradientTape() as tape:
            tape.watch(rates)
            prediction = model(
                with_composite_rates(inputs, rates, ratios),
                training=False,
            )[target_index, 0]
        return tape.gradient(prediction, rates)

    return tf.map_fn(
        one,
        alphas,
        fn_output_signature=tf.TensorSpec((3,), tf.float32),
    )


def integrate(grid, actual, baseline):
    """Convert the fixed gradient grid into per-demand IG attributions."""
    average = (grid[0] + grid[-1] + 2 * tf.reduce_sum(grid[1:-1], axis=0)) / (
        2 * FINAL_IG_STEPS
    )
    return (actual - baseline) * average


def top_forward_set(values):
    """Check whether d1 and d2 are the two strongest attributions."""
    top_two = {DEMAND_ORDER[index] for index in np.argsort(-np.abs(values))[:2]}
    return top_two == {"d1", "d2"}


def plots(output_dir, rows):
    """Plot absolute IG curves and congested relative demand relevance."""
    colors = {"d1": "#1f77b4", "d2": "#ff7f0e", "d3": "#2ca02c"}
    baselines = ("zero_traffic", "low_load")
    targets = ("d1", "d2")

    figure, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey=True)
    for row_index, target in enumerate(targets):
        for column_index, baseline in enumerate(baselines):
            axis = axes[row_index, column_index]
            selected = [
                row
                for row in rows
                if row["target_demand"] == target and row["baseline"] == baseline
            ]
            for demand in DEMAND_ORDER:
                values = sorted(
                    [row for row in selected if row["attributed_demand"] == demand],
                    key=lambda row: row["rho_wire"],
                )
                axis.plot(
                    [row["rho_wire"] for row in values],
                    [row["absolute_ig"] for row in values],
                    marker="o",
                    linewidth=1.5,
                    label=demand,
                    color=colors[demand],
                )
            axis.axvline(1.0, color="#64748b", linestyle="--", linewidth=1)
            axis.set_title(f"target {target}, {baseline}")
            axis.set_xlabel(r"Offered bottleneck load $\rho$")
            axis.set_ylabel("Absolute IG")
            axis.grid(axis="y", color="#cbd5e1", linewidth=0.6, alpha=0.45)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=True,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_dir / "test_ig_absolute_vs_rho.png", dpi=180)
    plt.close(figure)

    congested = [row for row in rows if row["rho_wire"] > 1]
    relative_means = [
        float(
            np.mean(
                [
                    row["relative_importance"]
                    for row in congested
                    if row["attributed_demand"] == demand
                ]
            )
        )
        for demand in DEMAND_ORDER
    ]
    figure, axis = plt.subplots(figsize=(6.5, 4.2))
    bars = axis.bar(
        DEMAND_ORDER,
        relative_means,
        color=[colors[demand] for demand in DEMAND_ORDER],
    )
    axis.bar_label(bars, labels=[f"{value:.1%}" for value in relative_means], padding=3)
    axis.set(
        ylim=(0, 0.62),
        xlabel="Attributed demand",
        ylabel="Mean relative |IG|",
        title="Relative attribution in congested test scenarios",
    )
    axis.text(
        0.5,
        -0.18,
        "Mean over five scenarios, two prediction targets and two reference states",
        transform=axis.transAxes,
        ha="center",
        color="#64748b",
        fontsize=9,
    )
    axis.grid(axis="y", color="#cbd5e1", linewidth=0.6, alpha=0.45)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output_dir / "test_ig_relative_congested.png", dpi=180)
    plt.close(figure)


def main():
    """Run frozen test IG, occlusion, and baseline robustness checks."""
    args = arguments()
    seed_all(args.seed)

    checkpoint = args.training_dir / "best.weights.h5"
    normalization_path = args.training_dir / "xndt_normalization.json"
    validation = list(read_jsonl(args.dataset_dir / "validation.jsonl"))
    selected = select_test_representatives(args.dataset_dir)
    normalization = json.loads(normalization_path.read_text())

    model = RouteNet_Fermi(
        xndt_mode=True,
        normalization=normalization["features"],
    )
    first, _, _ = adapt_sample(selected[0], True)
    model(first, training=False)
    model.load_weights(checkpoint)

    before = weight_fingerprint(model)
    checkpoint_hash = sha256(checkpoint)
    rows = []
    completeness = []
    occlusion_rows = []
    explanation_values = {}

    for raw in selected:
        inputs, _, _ = adapt_sample(raw, True)
        validate_inputs(inputs)
        actual, ratios = traffic_and_ratio(inputs)

        low = select_low_load(validation, raw)
        low_inputs, _, _ = adapt_sample(low, True)
        low_rates, _ = traffic_and_ratio(low_inputs)

        for target_index, target in enumerate(("d1", "d2")):
            baselines = (
                ("zero_traffic", tf.zeros_like(actual)),
                ("low_load", low_rates),
            )
            for baseline_name, baseline in baselines:
                grid = gradient_grid_512(
                    model,
                    inputs,
                    tf.cast(baseline, tf.float32),
                    actual,
                    ratios,
                    target_index,
                )
                if not bool(tf.reduce_all(tf.math.is_finite(grid))):
                    raise ValueError("Non-finite gradient")

                attrs = np.asarray(integrate(grid, actual, baseline))
                p_actual = float(output(model, inputs, actual, ratios, target_index))
                p_baseline = float(
                    output(model, inputs, baseline, ratios, target_index)
                )
                difference = p_actual - p_baseline
                error = abs(float(attrs.sum()) - difference)
                denominator = float(np.abs(attrs).sum())
                relative = np.abs(attrs) / denominator if denominator else np.zeros(3)
                deltas = np.asarray(occlusion(model, inputs, baseline, target_index))
                key = (raw["scenario_key"], target, baseline_name)
                explanation_values[key] = (attrs, relative, deltas)
                metadata = raw["metadata"]

                completeness.append(
                    {
                        "scenario_key": raw["scenario_key"],
                        "rho_wire": metadata["rho_wire"],
                        "target_demand": target,
                        "baseline": baseline_name,
                        "prediction_actual": p_actual,
                        "prediction_baseline": p_baseline,
                        "sum_ig": float(attrs.sum()),
                        "prediction_difference": difference,
                        "completeness_error": error,
                        "passes_completeness": (error <= COMPLETENESS_TOLERANCE),
                    }
                )
                occlusion_rows.append(
                    {
                        "scenario_key": raw["scenario_key"],
                        "rho_wire": metadata["rho_wire"],
                        "target_demand": target,
                        "baseline": baseline_name,
                        "ig_ranking": ranking(attrs),
                        "occlusion_ranking": ranking(deltas),
                        "exact_ranking_agreement": (ranking(attrs) == ranking(deltas)),
                        "ig_d3_last": ranking(attrs).endswith("d3"),
                        "occlusion_d3_last": ranking(deltas).endswith("d3"),
                        "ig_forward_set": top_forward_set(attrs),
                        "occlusion_forward_set": top_forward_set(deltas),
                    }
                )
                for index, demand in enumerate(DEMAND_ORDER):
                    rows.append(
                        {
                            "scenario_key": raw["scenario_key"],
                            "rho_wire": metadata["rho_wire"],
                            "alpha": metadata["alpha"],
                            "control_wire_rate_gbps": metadata[
                                "control_wire_rate_gbps"
                            ],
                            "target_demand": target,
                            "baseline": baseline_name,
                            "attributed_demand": demand,
                            "prediction_actual": p_actual,
                            "prediction_baseline": p_baseline,
                            "signed_ig": float(attrs[index]),
                            "absolute_ig": float(abs(attrs[index])),
                            "relative_importance": float(relative[index]),
                            "completeness_error": error,
                            "passes_completeness": (error <= COMPLETENESS_TOLERANCE),
                        }
                    )

    baseline_pairs = []
    for raw in selected:
        for target in ("d1", "d2"):
            zero = explanation_values[(raw["scenario_key"], target, "zero_traffic")]
            low = explanation_values[(raw["scenario_key"], target, "low_load")]
            baseline_pairs.append(
                {
                    "scenario_key": raw["scenario_key"],
                    "rho_wire": raw["metadata"]["rho_wire"],
                    "target_demand": target,
                    "zero_ranking": ranking(zero[0]),
                    "low_load_ranking": ranking(low[0]),
                    "ranking_agreement": ranking(zero[0]) == ranking(low[0]),
                    "zero_d3_last": ranking(zero[0]).endswith("d3"),
                    "low_load_d3_last": ranking(low[0]).endswith("d3"),
                    "zero_forward_set": top_forward_set(zero[0]),
                    "low_load_forward_set": top_forward_set(low[0]),
                    "forward_set_agreement": (
                        top_forward_set(zero[0]) == top_forward_set(low[0])
                    ),
                    "relative_d1_difference": float(abs(zero[1][0] - low[1][0])),
                    "relative_d2_difference": float(abs(zero[1][1] - low[1][1])),
                    "signed_d1_difference": float(abs(zero[0][0] - low[0][0])),
                    "signed_d2_difference": float(abs(zero[0][1] - low[0][1])),
                    "signed_d3_difference": float(abs(zero[0][2] - low[0][2])),
                }
            )

    after = weight_fingerprint(model)
    if before != after:
        raise AssertionError("Weights changed during final test XAI")

    errors = np.array([row["completeness_error"] for row in completeness])
    congested = [row for row in rows if row["rho_wire"] > 1]
    attr_summary = {}
    for demand in DEMAND_ORDER:
        values = [row for row in congested if row["attributed_demand"] == demand]
        attr_summary[demand] = {
            "mean_absolute_ig": float(np.mean([row["absolute_ig"] for row in values])),
            "mean_relative_importance": float(
                np.mean([row["relative_importance"] for row in values])
            ),
        }

    explanations = defaultdict(list)
    for row in congested:
        key = (
            row["scenario_key"],
            row["target_demand"],
            row["baseline"],
        )
        explanations[key].append(row)

    forward = [
        sum(
            row["relative_importance"]
            for row in values
            if row["attributed_demand"] in ("d1", "d2")
        )
        for values in explanations.values()
    ]
    control = [
        sum(
            row["relative_importance"]
            for row in values
            if row["attributed_demand"] == "d3"
        )
        for values in explanations.values()
    ]
    d3 = [row["absolute_ig"] for row in rows if row["attributed_demand"] == "d3"]

    summary = {
        "protocol": {
            "method": "Integrated Gradients",
            "steps": 512,
            "points": 513,
            "integration": "uniform trapezoidal",
            "tolerance": COMPLETENESS_TOLERANCE,
        },
        "checkpoint_sha256": checkpoint_hash,
        "weight_fingerprint_before": before,
        "weight_fingerprint_after": after,
        "weights_unchanged": before == after,
        "test_scenarios": 14,
        "explanations": 56,
        "completeness": {
            "n_pass": int(np.sum(errors <= COMPLETENESS_TOLERANCE)),
            "n_fail": int(np.sum(errors > COMPLETENESS_TOLERANCE)),
            "mean_error": float(errors.mean()),
            "median_error": float(np.median(errors)),
            "max_error": float(errors.max()),
        },
        "congested_attribution": attr_summary,
        "mean_forward_relevance": float(np.mean(forward)),
        "max_control_relevance": float(np.max(control)),
        "d3_negative_control": {
            "mean_absolute_ig": float(np.mean(d3)),
            "max_absolute_ig": float(np.max(d3)),
        },
        "baseline_robustness": {
            "pairs": len(baseline_pairs),
            "ranking_agreement": sum(
                row["ranking_agreement"] for row in baseline_pairs
            ),
            "d3_last_both": sum(
                row["zero_d3_last"] and row["low_load_d3_last"]
                for row in baseline_pairs
            ),
            "forward_set_agreement": sum(
                row["forward_set_agreement"] for row in baseline_pairs
            ),
            "forward_set_both": sum(
                row["zero_forward_set"] and row["low_load_forward_set"]
                for row in baseline_pairs
            ),
        },
        "occlusion": {
            "comparisons": len(occlusion_rows),
            "exact_ranking_agreement": sum(
                row["exact_ranking_agreement"] for row in occlusion_rows
            ),
            "d3_last_both": sum(
                row["ig_d3_last"] and row["occlusion_d3_last"] for row in occlusion_rows
            ),
            "forward_demand_agreement": sum(
                row["ig_forward_set"] and row["occlusion_forward_set"]
                for row in occlusion_rows
            ),
        },
        "all_finite": bool(np.isfinite([row["signed_ig"] for row in rows]).all()),
        "repetition_rule": "minimum repetition_index, then run_id",
        "low_load_rule": (
            "validation rho=0.20 loss-free; minimum alpha distance, "
            "control-rate distance, scenario_key/run_id"
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ig_test.csv", rows)
    (args.output_dir / "ig_test_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    plots(args.output_dir, rows)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
