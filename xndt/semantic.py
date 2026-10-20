"""Build the post-hoc semantic XNDT explanation layer from frozen artifacts.

This module never loads the model.  Demand-level IG is projected onto installed
paths for semantic diagnosis; the result is not an IG attribution to a link.
Telemetry remains a separate physical-evidence channel and is never combined
numerically with model evidence.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DEMAND_ORDER = ("d1", "d2", "d3")
LOGICAL_SWITCHES = {"sw03": "Switch A", "sw04": "Switch B"}


def arguments():
    """Parse the input and output paths for semantic alignment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--xai-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    """Read all non-empty records from a JSONL file."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path):
    """Read a CSV file into a list of dictionaries."""
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    """Write result dictionaries with an optional explicit column order."""
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ordered_demands(sample):
    """Return demands in the fixed d1, d2, d3 order."""
    if len(sample["demands"]) != len(DEMAND_ORDER):
        raise ValueError("Every XNDT sample must contain exactly d1, d2 and d3")
    by_id = {d["demand_id"]: d for d in sample["demands"]}
    if set(by_id) != set(DEMAND_ORDER):
        raise ValueError("Every XNDT sample must contain exactly d1, d2 and d3")
    return [by_id[d] for d in DEMAND_ORDER]


def ordered_path(demand):
    """Return and validate one demand's directed forwarding path."""
    path = sorted(demand["path"], key=lambda link: link["order"])
    if [link["order"] for link in path] != list(range(len(path))):
        raise ValueError(f"Non-contiguous path order for {demand['demand_id']}")
    return path


def logical_link(link):
    """Format a directed link with presentation-level switch names."""
    source = LOGICAL_SWITCHES.get(link["source"], link["source"])
    destination = LOGICAL_SWITCHES.get(link["destination"], link["destination"])
    return f"{source} -> {destination}"


def group_test_runs(rows):
    """Group test runs by scenario and validate their repetition structure."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario_key"]].append(row)
    if (
        len(rows) != 42
        or len(groups) != 14
        or any(len(v) != 3 for v in groups.values())
    ):
        raise ValueError("Expected 42 test runs in 14 groups of three repetitions")
    for values in groups.values():
        values.sort(key=lambda x: (x["repetition_index"], x["run_id"]))
    return dict(sorted(groups.items()))


def validate_paths(groups):
    """Verify stable paths and the expected forward and reverse resources."""
    reference = None
    for scenario, runs in groups.items():
        signatures = []
        for run in runs:
            signature = tuple(
                (d["demand_id"], tuple(link["link_id"] for link in ordered_path(d)))
                for d in ordered_demands(run)
            )
            signatures.append(signature)
        if len(set(signatures)) != 1:
            raise ValueError(f"Paths vary across repetitions for {scenario}")
        reference = reference or signatures[0]
        if signatures[0] != reference:
            raise ValueError("Installed paths vary across test scenarios")
    paths = dict(reference)
    if "sw03->sw04" not in paths["d1"] or "sw03->sw04" not in paths["d2"]:
        raise ValueError("d1 and d2 do not share the expected forward resource")
    if "sw03->sw04" in paths["d3"] or "sw04->sw03" not in paths["d3"]:
        raise ValueError("d3 does not use the expected reverse resource")
    return paths


def project_links(sample, attribution_rows):
    """Project absolute demand attributions onto installed directed links.

    The result is semantic relevance, not a gradient of a link feature.
    """
    paths = {d["demand_id"]: ordered_path(d) for d in ordered_demands(sample)}
    demand_abs = {
        r["attributed_demand"]: float(r["absolute_ig"]) for r in attribution_rows
    }
    active = {}
    users = defaultdict(list)
    for demand_id in DEMAND_ORDER:
        for link in paths[demand_id]:
            active.setdefault(link["link_id"], link)
            users[link["link_id"]].append(demand_id)
    projected = {
        link_id: sum(demand_abs[d] for d in demand_ids)
        for link_id, demand_ids in users.items()
    }
    denominator = sum(projected.values())
    result = []
    for link_id, link in active.items():
        result.append(
            {
                "link_id": link_id,
                "source": link["source"],
                "destination": link["destination"],
                "logical_link": logical_link(link),
                "source_interface": link["source_interface"],
                "destination_interface": link["destination_interface"],
                "capacity_bps": link["capacity_bps"],
                "traversing_demands": ";".join(users[link_id]),
                "shared_by_multiple_demands": len(users[link_id]) > 1,
                "projected_link_relevance": projected[link_id],
                "relative_projected_link_relevance": (
                    projected[link_id] / denominator if denominator else 0.0
                ),
                "relevance_type": "semantic/projected relevance (not Integrated-Gradients link attribution)",
            }
        )
    return result


def main():
    """Join frozen demand attributions with forwarding paths and summarize them."""
    args = arguments()
    sources = {
        "checkpoint": args.training_dir / "best.weights.h5",
        "ig_csv": args.xai_dir / "ig_test.csv",
        "ig_summary": args.xai_dir / "ig_test_summary.json",
        "test_jsonl": args.dataset_dir / "test.jsonl",
        "scenario_predictions": args.prediction_dir / "test_scenario_predictions.csv",
    }
    hashes_before = {name: sha256(path) for name, path in sources.items()}
    test_groups = group_test_runs(read_jsonl(sources["test_jsonl"]))
    path_map = validate_paths(test_groups)
    ig_rows = read_csv(sources["ig_csv"])
    if len(ig_rows) != 168:
        raise ValueError("Expected 168 frozen attribution rows")
    prediction_rows = read_csv(sources["scenario_predictions"])
    predictions = {(r["scenario_key"], r["demand_id"]): r for r in prediction_rows}
    ig_groups = defaultdict(list)
    for row in ig_rows:
        ig_groups[(row["scenario_key"], row["target_demand"], row["baseline"])].append(
            row
        )
    if len(ig_groups) != 56 or any(len(v) != 3 for v in ig_groups.values()):
        raise ValueError(
            "Expected 56 frozen explanations with three demand attributions each"
        )

    explanation_rows, projected_rows = [], []
    for key, attrs in sorted(ig_groups.items()):
        scenario, target, baseline = key
        sample = test_groups[scenario][0]
        attr_by_demand = {r["attributed_demand"]: r for r in attrs}
        if set(attr_by_demand) != set(DEMAND_ORDER):
            raise ValueError("Attribution demand order is incomplete")
        projected = project_links(sample, attrs)
        for row in projected:
            projected_rows.append(
                {
                    "scenario_key": scenario,
                    "rho_wire": sample["metadata"]["rho_wire"],
                    "target_demand": target,
                    "baseline": baseline,
                    **row,
                }
            )
        for demand in ordered_demands(sample):
            demand_id = demand["demand_id"]
            attr = attr_by_demand[demand_id]
            pred = predictions[(scenario, demand_id)]
            links = ordered_path(demand)
            record = {
                "scenario_key": scenario,
                "rho_wire": sample["metadata"]["rho_wire"],
                "alpha": sample["metadata"]["alpha"],
                "control_wire_rate_gbps": sample["metadata"]["control_wire_rate_gbps"],
                "target_demand": target,
                "baseline": baseline,
                "demand_id": demand_id,
                "offered_wire_rate_bps": demand["traffic"]["configured_wire_rate_bps"],
                "predicted_loss": float(pred["predicted_loss_mean"]),
                "measured_loss_mean": float(pred["measured_loss_mean"]),
                "measured_loss_std": float(pred["measured_loss_std"]),
                "signed_ig": float(attr["signed_ig"]),
                "absolute_ig": float(attr["absolute_ig"]),
                "relative_importance": float(attr["relative_importance"]),
                "ordered_directed_links": ";".join(link["link_id"] for link in links),
                "model_evidence": "frozen demand-level Integrated Gradients",
                "forwarding_evidence": "installed ordered directed path from neutral XNDT export",
            }
            explanation_rows.append(record)

    scenario_rows = []
    for scenario, runs in test_groups.items():
        rho = float(runs[0]["metadata"]["rho_wire"])
        zero_groups = [ig_groups[(scenario, t, "zero_traffic")] for t in ("d1", "d2")]
        d3_zero = all(
            float(
                next(r for r in group if r["attributed_demand"] == "d3")["absolute_ig"]
            )
            == 0
            for group in zero_groups
        )
        forward_set = all(
            set(
                r["attributed_demand"]
                for r in sorted(group, key=lambda x: -float(x["absolute_ig"]))[:2]
            )
            == {"d1", "d2"}
            for group in zero_groups
        )
        p = [
            float(predictions[(scenario, d)]["predicted_loss_mean"])
            for d in ("d1", "d2")
        ]
        m = [
            float(predictions[(scenario, d)]["measured_loss_mean"])
            for d in ("d1", "d2")
        ]
        scenario_rows.append(
            {
                "scenario_key": scenario,
                "rho_wire": rho,
                "load_regime": "rho<=1" if rho <= 1 else "rho>1",
                "predicted_forward_loss_mean": float(np.mean(p)),
                "measured_forward_loss_mean": float(np.mean(m)),
                "forward_relevant_set_is_d1_d2": forward_set,
                "d3_absolute_ig_is_zero": d3_zero,
                "d1_d2_shared_directed_link": "sw03->sw04",
                "shared_link_verified_from_paths": "sw03->sw04" in path_map["d1"]
                and "sw03->sw04" in path_map["d2"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "xndt_explanations.csv", explanation_rows)
    write_csv(args.output_dir / "xndt_projected_link_relevance.csv", projected_rows)

    hashes_after = {name: sha256(path) for name, path in sources.items()}
    if hashes_before != hashes_after:
        raise AssertionError(
            "A frozen model, XAI, dataset, or prediction artifact changed"
        )
    below = [r for r in scenario_rows if r["rho_wire"] <= 1]
    above = [r for r in scenario_rows if r["rho_wire"] > 1]
    summary = {
        "scenarios": len(scenario_rows),
        "congested_scenarios": len(above),
        "forward_set_d1_d2_count": sum(
            r["forward_relevant_set_is_d1_d2"] for r in scenario_rows
        ),
        "d3_zero_count": sum(r["d3_absolute_ig_is_zero"] for r in scenario_rows),
        "d1_d2_converge_count": sum(
            r["shared_link_verified_from_paths"] for r in scenario_rows
        ),
        "shared_directed_resource": "sw03->sw04",
        "reverse_control_resource": "sw04->sw03",
        "scope": "semantic alignment only; physical telemetry is joined by xndt.evidence",
        "rho_le_1": {
            "scenarios": len(below),
            "mean_predicted_forward_loss": float(
                np.mean([r["predicted_forward_loss_mean"] for r in below])
            ),
            "mean_measured_forward_loss": float(
                np.mean([r["measured_forward_loss_mean"] for r in below])
            ),
        },
        "rho_gt_1": {
            "scenarios": len(above),
            "mean_predicted_forward_loss": float(
                np.mean([r["predicted_forward_loss_mean"] for r in above])
            ),
            "mean_measured_forward_loss": float(
                np.mean([r["measured_forward_loss_mean"] for r in above])
            ),
        },
        "integrity_hashes_before": hashes_before,
        "integrity_hashes_after": hashes_after,
        "integrity": {
            "unchanged": hashes_before == hashes_after,
            "model_loaded": False,
            "training_performed": False,
            "xai_recomputed": False,
            "telemetry_used_as_model_or_attribution_input": False,
        },
    }
    (args.output_dir / "xndt_semantic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
