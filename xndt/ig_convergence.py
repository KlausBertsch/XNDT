"""Fixed validation-only convergence sweep for the composite XNDT IG function."""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from xndt.data import DEMAND_ORDER, adapt_sample
from xndt.integrated_gradients import (
    COMPLETENESS_TOLERANCE,
    output,
    seed_all,
    select_low_load,
    traffic_and_ratio,
    validate_inputs,
    validation_scenarios,
    weight_fingerprint,
)
from xndt.model import RouteNet_Fermi


STEPS = (64, 128, 256, 512, 1024)


def arguments():
    """Parse paths and the reproducibility seed for the convergence sweep."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def with_composite_rates(inputs, rates, ratios):
    """Replace bit and packet rates together along the attribution path."""
    result = dict(inputs)
    result["traffic"] = tf.reshape(rates, (-1, 1))
    result["packets"] = tf.reshape(rates * ratios, (-1, 1))
    return result


@tf.function(reduce_retracing=True)
def gradient_grid(model, inputs, baseline, actual, ratios, target_index):
    """Evaluate rate gradients on the shared 1024-interval integration grid."""
    alphas = tf.linspace(0.0, 1.0, STEPS[-1] + 1)

    def one(alpha):
        """Evaluate one gradient point on the straight baseline-to-input path."""
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


def integrate_grid(grid, actual, baseline, steps):
    """Integrate a nested subgrid with the uniform trapezoidal rule."""
    stride = STEPS[-1] // steps
    gradients = grid[::stride]
    average = (
        gradients[0] + gradients[-1] + 2 * tf.reduce_sum(gradients[1:-1], axis=0)
    ) / (2 * steps)
    return (actual - baseline) * average


def main():
    """Select the smallest step count that passes every validation check."""
    args = arguments()
    seed_all(args.seed)

    normalization_path = args.training_dir / "xndt_normalization.json"
    normalization = json.loads(normalization_path.read_text())
    all_validation, selected = validation_scenarios(args.dataset_dir)
    model = RouteNet_Fermi(
        xndt_mode=True,
        normalization=normalization["features"],
    )
    first, _, _ = adapt_sample(selected[0], True)
    model(first, training=False)
    model.load_weights(args.training_dir / "best.weights.h5")

    before = weight_fingerprint(model)
    completeness = []
    attributions = []
    ratios_seen = []

    for raw in selected:
        inputs, _, _ = adapt_sample(raw, True)
        validate_inputs(inputs)
        actual, ratios = traffic_and_ratio(inputs)
        ratios_seen.extend(np.asarray(ratios).tolist())

        low = select_low_load(all_validation, raw)
        low_inputs, _, _ = adapt_sample(low, True)
        low_rates, _ = traffic_and_ratio(low_inputs)

        for target_index, target in enumerate(("d1", "d2")):
            baselines = (
                ("zero_traffic", tf.zeros_like(actual)),
                ("low_load", low_rates),
            )
            for baseline_name, baseline in baselines:
                grid = gradient_grid(
                    model,
                    inputs,
                    tf.cast(baseline, tf.float32),
                    actual,
                    ratios,
                    target_index,
                )
                if not bool(tf.reduce_all(tf.math.is_finite(grid))):
                    raise ValueError("Non-finite gradient grid")

                p_actual = float(output(model, inputs, actual, ratios, target_index))
                p_baseline = float(
                    output(model, inputs, baseline, ratios, target_index)
                )
                difference = p_actual - p_baseline

                for steps in STEPS:
                    attrs = np.asarray(integrate_grid(grid, actual, baseline, steps))
                    error = abs(float(attrs.sum()) - difference)
                    denominator = float(np.abs(attrs).sum())
                    relative = (
                        np.abs(attrs) / denominator if denominator else np.zeros(3)
                    )
                    completeness.append(
                        {
                            "steps": steps,
                            "scenario_key": raw["scenario_key"],
                            "rho_wire": raw["metadata"]["rho_wire"],
                            "target_demand": target,
                            "baseline": baseline_name,
                            "prediction_actual": p_actual,
                            "prediction_baseline": p_baseline,
                            "sum_ig": float(attrs.sum()),
                            "prediction_difference": difference,
                            "completeness_error": error,
                            "passes_tolerance": (error <= COMPLETENESS_TOLERANCE),
                        }
                    )
                    for index, demand in enumerate(DEMAND_ORDER):
                        attributions.append(
                            {
                                "steps": steps,
                                "scenario_key": raw["scenario_key"],
                                "rho_wire": raw["metadata"]["rho_wire"],
                                "target_demand": target,
                                "baseline": baseline_name,
                                "attributed_demand": demand,
                                "signed_ig": float(attrs[index]),
                                "absolute_ig": float(abs(attrs[index])),
                                "relative_importance": float(relative[index]),
                            }
                        )

    after = weight_fingerprint(model)
    if before != after:
        raise AssertionError("Weights changed during convergence sweep")

    summaries = []
    selected_steps = None
    for steps in STEPS:
        errors = np.array(
            [row["completeness_error"] for row in completeness if row["steps"] == steps]
        )
        n_pass = int(np.sum(errors <= COMPLETENESS_TOLERANCE))
        summaries.append(
            {
                "steps": steps,
                "n_explanations": len(errors),
                "n_pass": n_pass,
                "mean_error": float(errors.mean()),
                "median_error": float(np.median(errors)),
                "max_error": float(errors.max()),
            }
        )
        if selected_steps is None and n_pass == len(errors):
            selected_steps = steps

    args.output_dir.mkdir(parents=True, exist_ok=True)
    congested = [row for row in attributions if row["rho_wire"] > 1]
    d3_by_steps = {
        str(steps): max(
            row["absolute_ig"]
            for row in congested
            if row["steps"] == steps and row["attributed_demand"] == "d3"
        )
        for steps in STEPS
    }
    summary = {
        "split": "validation only",
        "steps_tested": STEPS,
        "integration": (
            "uniform trapezoidal rule; nested subgrids from the fixed 1024 grid"
        ),
        "composite_audit": {
            "watched_tensor": "r=[r_d1,r_d2,r_d3]",
            "traffic": "r",
            "packets": "r * sample-specific pps_per_bps inside GradientTape",
            "total_derivative_includes_packets_path": True,
            "pps_per_bps_min": min(ratios_seen),
            "pps_per_bps_max": max(ratios_seen),
        },
        "tolerance": COMPLETENESS_TOLERANCE,
        "selection_rule": "smallest tested steps with all 24 errors <= tolerance",
        "selected_steps": selected_steps,
        "summaries": summaries,
        "max_congested_d3_absolute_ig_by_steps": d3_by_steps,
        "weight_fingerprint_before": before,
        "weight_fingerprint_after": after,
        "weights_unchanged": before == after,
        "all_finite": bool(
            np.isfinite([row["signed_ig"] for row in attributions]).all()
        ),
    }
    output_path = args.output_dir / "ig_convergence_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
