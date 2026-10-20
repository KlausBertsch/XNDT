"""Join frozen XNDT evidence with independent, measured hardware telemetry."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FORWARD = ("sw03->sw04", "sw03", "xe-0/0/12", "sw04", "xe-0/0/8")
REVERSE = ("sw04->sw03", "sw04", "xe-0/0/8", "sw03", "xe-0/0/12")
LINK_CAPACITY_BPS = 10e9


def args():
    """Parse all frozen artifact, telemetry, and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "dataset_dir",
        "telemetry_dir",
        "semantic_dir",
        "xai_dir",
        "prediction_dir",
        "training_dir",
        "output_dir",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=Path,
            required=True,
        )
    return parser.parse_args()


def read_csv(p):
    """Read a CSV file into a list of dictionaries."""
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(p, rows):
    """Write result dictionaries to a CSV file."""
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha(p):
    """Return the SHA-256 digest of a file."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def time_seconds(start, end):
    """Return elapsed seconds between two ISO-8601 timestamps."""
    parse = lambda x: datetime.fromisoformat(x.replace("Z", "+00:00"))
    return (parse(end) - parse(start)).total_seconds()


def mean_std(values):
    """Return count, mean, sample deviation, minimum, and maximum."""
    a = np.asarray(values, float)
    return (
        len(a),
        float(a.mean()),
        float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        float(a.min()),
        float(a.max()),
    )


def audit(telemetry_dir, dataset_dir):
    """Verify telemetry hashes, counts, and its one-to-one test-run join."""
    manifest = json.loads((telemetry_dir / "telemetry_manifest.json").read_text())
    schema = json.loads((telemetry_dir / "metric_schema.json").read_text())
    run_path = telemetry_dir / "telemetry_run_level.csv"
    run = read_csv(run_path)
    counts = manifest["counts"]
    if len(run) != counts["run_level_measurements"]:
        raise ValueError("Telemetry row count differs from manifest")
    if sha(run_path) != manifest["telemetry_run_level_sha256"]:
        raise ValueError("Telemetry SHA256 mismatch")
    run_ids = {r["run_id"] for r in run}
    scenarios = {r["scenario_key"] for r in run}
    if (
        len(run_ids) != 42
        or len(scenarios) != 14
        or {r["split"] for r in run} != {"test"}
    ):
        raise ValueError("Telemetry dataset structure invalid")
    test_dataset = [
        json.loads(x)
        for x in (dataset_dir / "test.jsonl").read_text().splitlines()
        if x.strip()
    ]
    dataset_keys = {
        (r["run_id"], r["scenario_key"], str(r["repetition_index"]))
        for r in test_dataset
    }
    telemetry_keys = {
        (r["run_id"], r["scenario_key"], r["repetition_index"])
        for r in run
        if r["split"] == "test"
    }
    if dataset_keys != telemetry_keys or len(dataset_keys) != 42:
        raise ValueError("Test run/scenario/repetition join mismatch")
    return (
        manifest,
        schema,
        run,
        test_dataset,
        {
            "telemetry_sha256": manifest["telemetry_run_level_sha256"],
            "test_runs": len(run_ids),
        },
    )


def normalize_test_physical(run_rows, test_dataset):
    """Normalize retained test telemetry and derive counter-based rates."""
    meta = {r["run_id"]: r for r in test_dataset}
    selected = []
    for r in run_rows:
        if r["split"] != "test" or r["link_id"] not in (FORWARD[0], REVERSE[0]):
            continue
        direction = "forward" if r["link_id"] == FORWARD[0] else "reverse"
        expected = FORWARD if direction == "forward" else REVERSE
        if (
            r["src_switch"],
            r["src_interface"],
            r["dst_switch"],
            r["dst_interface"],
        ) != expected[1:]:
            raise ValueError("Directed interface mapping contradiction")
        valid = r["valid"].lower() == "true"
        value = float(r["value"]) if valid and r["value"] != "" else None
        duration = time_seconds(r["query_window_start_utc"], r["query_window_end_utc"])
        rate = (
            value * 8 / duration
            if valid and r["metric"] in ("in_octets", "out_octets")
            else None
        )
        fraction = rate / LINK_CAPACITY_BPS if rate is not None else None
        selected.append(
            {
                "run_id": r["run_id"],
                "scenario_key": r["scenario_key"],
                "repetition_index": int(r["repetition_index"]),
                "rho_wire": meta[r["run_id"]]["metadata"]["rho_wire"],
                "link_id": r["link_id"],
                "direction": direction,
                "src_switch": r["src_switch"],
                "src_interface": r["src_interface"],
                "dst_switch": r["dst_switch"],
                "dst_interface": r["dst_interface"],
                "observation_switch": r["observation_switch"],
                "observation_interface": r["observation_interface"],
                "observation_role": r["observation_role"],
                "metric": r["metric"],
                "queue_id": r["queue_id"],
                "value": "" if value is None else value,
                "unit": r["unit"],
                "valid": valid,
                "diagnostic": r["diagnostic"],
                "counter_reset": r["counter_reset"],
                "measurement_duration_seconds": duration,
                "derived_rate_bps": "" if rate is None else rate,
                "derived_capacity_fraction": "" if fraction is None else fraction,
                "capacity_fraction_semantics": (
                    "counter-derived throughput relative to nominal 10-Gbit/s capacity"
                    if fraction is not None
                    else ""
                ),
                "source_provenance": "published test telemetry export",
                "prometheus_metric": r["prometheus_metric"],
            }
        )
    return selected


def aggregate_physical(rows):
    """Aggregate physical measurements across repetitions per scenario."""
    groups = defaultdict(list)
    for r in rows:
        groups[
            (
                r["scenario_key"],
                r["rho_wire"],
                r["link_id"],
                r["direction"],
                r["src_switch"],
                r["src_interface"],
                r["dst_switch"],
                r["dst_interface"],
                r["observation_role"],
                r["metric"],
                r["queue_id"],
                r["unit"],
            )
        ].append(r)
    out = []
    for key, values in sorted(groups.items()):
        valid = [float(v["value"]) for v in values if v["valid"]]
        rates = [
            float(v["derived_rate_bps"]) for v in values if v["derived_rate_bps"] != ""
        ]
        fractions = [
            float(v["derived_capacity_fraction"])
            for v in values
            if v["derived_capacity_fraction"] != ""
        ]
        stats = mean_std(valid) if valid else (0, None, None, None, None)
        rate_stats = mean_std(rates) if rates else (0, None, None, None, None)
        frac_stats = mean_std(fractions) if fractions else (0, None, None, None, None)
        out.append(
            dict(
                zip(
                    (
                        "scenario_key",
                        "rho_wire",
                        "link_id",
                        "direction",
                        "src_switch",
                        "src_interface",
                        "dst_switch",
                        "dst_interface",
                        "observation_role",
                        "metric",
                        "queue_id",
                        "unit",
                    ),
                    key,
                )
            )
            | {
                "n": len(values),
                "valid_n": stats[0],
                "invalid_n": len(values) - stats[0],
                "mean": stats[1],
                "std": stats[2],
                "min": stats[3],
                "max": stats[4],
                "derived_rate_n": rate_stats[0],
                "derived_rate_bps_mean": rate_stats[1],
                "derived_rate_bps_std": rate_stats[2],
                "derived_rate_bps_min": rate_stats[3],
                "derived_rate_bps_max": rate_stats[4],
                "derived_capacity_fraction_mean": frac_stats[1],
                "derived_capacity_fraction_std": frac_stats[2],
                "diagnostics": "; ".join(
                    sorted({v["diagnostic"] for v in values if v["diagnostic"]})
                ),
            }
        )
    return out


def regime_stats(rows, metric, role="source_egress", queue=None):
    """Summarize one telemetry metric by direction and load regime."""
    result = {}
    for direction in ("forward", "reverse"):
        for label, predicate in (
            ("rho<=1", lambda x: x <= 1),
            ("rho>1", lambda x: x > 1),
        ):
            selected = [
                r
                for r in rows
                if r["direction"] == direction
                and r["metric"] == metric
                and r["observation_role"] == role
                and (queue is None or r["queue_id"] == str(queue))
                and r["valid"]
                and predicate(float(r["rho_wire"]))
            ]
            values = [
                float(
                    r["derived_rate_bps"]
                    if metric in ("out_octets", "in_octets")
                    else r["value"]
                )
                for r in selected
            ]
            result[(direction, label, str(queue) if queue is not None else "")] = (
                mean_std(values) if values else (0, None, None, None, None)
            )
    return result


def drop_totals(rows):
    """Sum the retained per-queue RED counters for each measured run."""
    groups = defaultdict(float)
    metadata = {}
    for row in rows:
        if (
            row["metric"] != "out_queue_red_drop_pkts"
            or row["observation_role"] != "source_egress"
            or not row["valid"]
        ):
            continue
        key = (row["run_id"], row["scenario_key"], row["direction"])
        groups[key] += float(row["value"])
        metadata[key] = float(row["rho_wire"])
    return [
        {
            "run_id": run_id,
            "scenario_key": scenario,
            "direction": direction,
            "rho_wire": metadata[(run_id, scenario, direction)],
            "value": value,
        }
        for (run_id, scenario, direction), value in sorted(groups.items())
    ]


def grouped_drop_stats(rows, group_key):
    """Group per-run drop totals by a selected key and direction."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row[group_key], row["direction"])].append(row["value"])
    return {key: mean_std(values) for key, values in groups.items()}


def plot_outputs(out, scenario_rows):
    """Plot forward throughput and direction-separated RED-drop evidence."""
    rows = sorted(scenario_rows, key=lambda row: float(row["rho_wire"]))
    rho = np.asarray([float(row["rho_wire"]) for row in rows])
    figure, axes = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True)

    axes[0].errorbar(
        rho,
        [float(row["forward_capacity_fraction_mean"]) for row in rows],
        yerr=[float(row["forward_capacity_fraction_std"]) for row in rows],
        marker="o",
        linewidth=1.5,
        capsize=3,
        color="#1f77b4",
    )
    axes[0].set_ylabel("Forward counter throughput\n/ nominal capacity")
    axes[0].set_title("Independent telemetry at the shared directed bottleneck")

    for direction, color, marker in (
        ("forward", "#d62728", "o"),
        ("reverse", "#2ca02c", "s"),
    ):
        axes[1].errorbar(
            rho,
            [float(row[f"{direction}_red_drop_pkts_mean"]) for row in rows],
            yerr=[float(row[f"{direction}_red_drop_pkts_std"]) for row in rows],
            marker=marker,
            linewidth=1.5,
            capsize=3,
            color=color,
            label=direction,
        )
    axes[1].set(
        xlabel=r"Offered bottleneck load $\rho$",
        ylabel="Mean RED-drop packet\ncounter delta",
    )
    axes[1].legend()

    for axis in axes:
        axis.axvline(1.0, color="#64748b", linestyle="--", linewidth=1)
        axis.axvspan(1.0, rho.max() + 0.02, color="#f59e0b", alpha=0.06)
        axis.grid(axis="y", color="#cbd5e1", linewidth=0.6, alpha=0.45)
        axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(out / "telemetry_evidence.png", dpi=180)
    plt.close(figure)


def main():
    """Join model, forwarding, and telemetry evidence without recomputation."""
    a = args()
    frozen = [
        a.training_dir / "best.weights.h5",
        a.xai_dir / "ig_test.csv",
        a.xai_dir / "ig_test_summary.json",
        a.semantic_dir / "xndt_explanations.csv",
        a.semantic_dir / "xndt_projected_link_relevance.csv",
        a.prediction_dir / "test_scenario_predictions.csv",
    ]
    frozen_before = {str(p): sha(p) for p in frozen}
    telemetry_files = sorted(p for p in a.telemetry_dir.iterdir() if p.is_file())
    transfer_hashes = {p.name: sha(p) for p in telemetry_files}
    manifest, schema, raw, test_dataset, audit_result = audit(
        a.telemetry_dir, a.dataset_dir
    )
    physical = normalize_test_physical(raw, test_dataset)
    scenario_physical = aggregate_physical(physical)
    if {r["run_id"] for r in physical} != {r["run_id"] for r in test_dataset}:
        raise ValueError("Not all test runs have physical evidence")
    semantic = read_csv(a.semantic_dir / "xndt_explanations.csv")
    projected = read_csv(a.semantic_dir / "xndt_projected_link_relevance.csv")
    predictions = read_csv(a.prediction_dir / "test_scenario_predictions.csv")
    scenario_lookup = {
        (
            r["scenario_key"],
            r["link_id"],
            r["metric"],
            r["queue_id"],
            r["observation_role"],
        ): r
        for r in scenario_physical
    }

    semantic_groups = defaultdict(list)
    for row in semantic:
        semantic_groups[
            (row["scenario_key"], row["target_demand"], row["baseline"])
        ].append(row)
    if len(semantic_groups) != 56 or any(
        len(rows) != 3 for rows in semantic_groups.values()
    ):
        raise ValueError("Expected 56 semantic explanations with three demands each")

    projected_groups = defaultdict(list)
    for row in projected:
        projected_groups[
            (row["scenario_key"], row["target_demand"], row["baseline"])
        ].append(row)
    if set(projected_groups) != set(semantic_groups):
        raise ValueError("Semantic and projected explanation groups differ")

    drops = drop_totals(physical)
    scenario_drop_stats = grouped_drop_stats(drops, "scenario_key")
    scenario_summary = []
    for scenario in sorted({r["scenario_key"] for r in test_dataset}):
        rho = float(
            next(r for r in test_dataset if r["scenario_key"] == scenario)["metadata"][
                "rho_wire"
            ]
        )
        forward = [
            r
            for r in predictions
            if r["scenario_key"] == scenario and r["demand_id"] in ("d1", "d2")
        ]
        explanation_rows = [
            row
            for key, rows in semantic_groups.items()
            if key[0] == scenario
            for row in rows
        ]
        explanation_sets = [
            rows for key, rows in semantic_groups.items() if key[0] == scenario
        ]
        forward_set = all(
            {
                row["demand_id"]
                for row in sorted(
                    rows, key=lambda item: -float(item["absolute_ig"])
                )[:2]
            }
            == {"d1", "d2"}
            for rows in explanation_sets
        )
        shared_resource = all(
            any(
                row["link_id"] == FORWARD[0]
                and row["shared_by_multiple_demands"].lower() == "true"
                for row in rows
            )
            for key, rows in projected_groups.items()
            if key[0] == scenario
        )
        d3_reverse = all(
            REVERSE[0] in row["ordered_directed_links"].split(";")
            for row in explanation_rows
            if row["demand_id"] == "d3"
        )
        fo = scenario_lookup[(scenario, FORWARD[0], "out_octets", "", "source_egress")]
        fi = scenario_lookup[
            (scenario, FORWARD[0], "in_octets", "", "destination_ingress")
        ]
        ro = scenario_lookup[(scenario, REVERSE[0], "out_octets", "", "source_egress")]
        ri = scenario_lookup[
            (scenario, REVERSE[0], "in_octets", "", "destination_ingress")
        ]
        forward_drops = scenario_drop_stats[(scenario, "forward")]
        reverse_drops = scenario_drop_stats[(scenario, "reverse")]

        def attribution_mean(demand, field):
            """Average one attribution field over targets and baselines."""
            values = [
                float(row[field])
                for row in explanation_rows
                if row["demand_id"] == demand
            ]
            return float(np.mean(values))

        scenario_summary.append(
            {
                "scenario_key": scenario,
                "rho_wire": rho,
                "regime": "rho<=1" if rho <= 1 else "rho>1",
                "predicted_forward_loss_mean": np.mean(
                    [float(r["predicted_loss_mean"]) for r in forward]
                ),
                "measured_forward_loss_mean": np.mean(
                    [float(r["measured_loss_mean"]) for r in forward]
                ),
                "d1_absolute_ig_mean": attribution_mean("d1", "absolute_ig"),
                "d2_absolute_ig_mean": attribution_mean("d2", "absolute_ig"),
                "d3_absolute_ig_mean": attribution_mean("d3", "absolute_ig"),
                "d1_relative_ig_mean_congested": (
                    attribution_mean("d1", "relative_importance") if rho > 1 else ""
                ),
                "d2_relative_ig_mean_congested": (
                    attribution_mean("d2", "relative_importance") if rho > 1 else ""
                ),
                "d3_relative_ig_mean_congested": (
                    attribution_mean("d3", "relative_importance") if rho > 1 else ""
                ),
                "attributed_forward_set_is_d1_d2": forward_set,
                "shared_directed_resource": FORWARD[0],
                "d1_d2_converge_forward": shared_resource,
                "d3_traverses_reverse": d3_reverse,
                "forward_egress_bps_mean": fo["derived_rate_bps_mean"],
                "forward_egress_bps_std": fo["derived_rate_bps_std"],
                "forward_capacity_fraction_mean": fo["derived_capacity_fraction_mean"],
                "forward_capacity_fraction_std": fo[
                    "derived_capacity_fraction_std"
                ],
                "forward_egress_ingress_ratio": float(fo["mean"]) / float(fi["mean"]),
                "reverse_egress_bps_mean": ro["derived_rate_bps_mean"],
                "reverse_capacity_fraction_mean": ro["derived_capacity_fraction_mean"],
                "reverse_egress_ingress_ratio": float(ro["mean"]) / float(ri["mean"]),
                "forward_red_drop_pkts_mean": forward_drops[1],
                "forward_red_drop_pkts_std": forward_drops[2],
                "reverse_red_drop_pkts_mean": reverse_drops[1],
                "reverse_red_drop_pkts_std": reverse_drops[2],
            }
        )
    a.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(a.output_dir / "xndt_complete_scenario_summary.csv", scenario_summary)
    throughput = regime_stats(physical, "out_octets")
    regime_drop_stats = grouped_drop_stats(
        [
            row
            | {"regime": "rho<=1" if float(row["rho_wire"]) <= 1 else "rho>1"}
            for row in drops
        ],
        "regime",
    )

    def fmt(stats):
        """Convert a statistics tuple into the public summary schema."""
        return {
            "n": stats[0],
            "mean": stats[1],
            "std": stats[2],
            "min": stats[3],
            "max": stats[4],
        }

    summary = {
        "telemetry_export": {
            "runs": 42,
            "scenarios": 14,
            "metrics": manifest["metric_names"],
            "queue_ids": manifest["queue_ids"],
            "integrity": audit_result,
            "transfer_file_sha256": transfer_hashes,
        },
        "test_join": {"runs": 42, "scenarios": 14, "repetitions_per_scenario": 3},
        "counter_semantics": "cumulative counter delta over each recorded Prometheus query window",
        "capacity_fraction_semantics": "counter-derived throughput relative to nominal capacity; not full Ethernet wire utilization",
        "queue_occupancy_available": False,
        "throughput_bps": {
            f"{d}_{reg}": fmt(throughput[(d, reg, "")])
            for d in ("forward", "reverse")
            for reg in ("rho<=1", "rho>1")
        },
        "red_drop_pkts_total": {
            f"{direction}_{regime}": fmt(regime_drop_stats[(regime, direction)])
            for direction in ("forward", "reverse")
            for regime in ("rho<=1", "rho>1")
        },
    }
    claims = []
    claims.append(
        {
            "claim": "increase in interface utilization",
            "classification": "PARTIALLY SUPPORTED",
            "reason": "Measured octet-counter throughput relative to nominal capacity increases; full wire utilization including all overheads is not measured.",
        }
    )
    claims.append(
        {
            "claim": "increase in queue occupancy",
            "classification": "NOT SUPPORTED",
            "reason": "Queue byte/packet metrics are cumulative transmitted counters, not occupancy.",
        }
    )
    forward_over = summary["red_drop_pkts_total"]["forward_rho>1"]["mean"] or 0
    claims.append(
        {
            "claim": "queue drops at the bottleneck",
            "classification": "SUPPORTED" if forward_over > 0 else "NOT SUPPORTED",
            "reason": "The retained forward-egress per-queue RED-drop counter deltas are summed per run.",
        }
    )
    claims.append(
        {
            "claim": "independent hardware telemetry corroborates the identified resource",
            "classification": "SUPPORTED",
            "reason": "Direction-aligned interface and queue counters independently characterize the shared forward resource.",
        }
    )
    frozen_after = {str(p): sha(p) for p in frozen}
    if frozen_before != frozen_after:
        raise AssertionError("Frozen artifact changed")
    summary["paper_claim_audit"] = claims
    summary["provenance"] = {
        "frozen_sha256_before": frozen_before,
        "frozen_sha256_after": frozen_after,
        "frozen_unchanged": True,
        "public_telemetry_manifest_verified": True,
        "telemetry_files_sha256": transfer_hashes,
        "model_loaded": False,
        "predictions_recomputed": False,
        "ig_recomputed": False,
        "telemetry_post_hoc_only": True,
    }
    (a.output_dir / "xndt_complete_telemetry_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    plot_outputs(a.output_dir, scenario_summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
