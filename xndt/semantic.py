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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEMAND_ORDER = ("d1", "d2", "d3")
LOGICAL_SWITCHES = {"sw03": "Switch A", "sw04": "Switch B"}
REPRESENTATIVE_RHOS = (0.95, 1.02, 1.20, 1.40)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--xai-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ordered_demands(sample):
    if len(sample["demands"]) != len(DEMAND_ORDER):
        raise ValueError("Every XNDT sample must contain exactly d1, d2 and d3")
    by_id = {d["demand_id"]: d for d in sample["demands"]}
    if set(by_id) != set(DEMAND_ORDER):
        raise ValueError("Every XNDT sample must contain exactly d1, d2 and d3")
    return [by_id[d] for d in DEMAND_ORDER]


def ordered_path(demand):
    path = sorted(demand["path"], key=lambda link: link["order"])
    if [link["order"] for link in path] != list(range(len(path))):
        raise ValueError(f"Non-contiguous path order for {demand['demand_id']}")
    return path


def logical_link(link):
    source = LOGICAL_SWITCHES.get(link["source"], link["source"])
    destination = LOGICAL_SWITCHES.get(link["destination"], link["destination"])
    return f"{source} -> {destination}"


def group_test_runs(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["scenario_key"]].append(row)
    if len(rows) != 42 or len(groups) != 14 or any(len(v) != 3 for v in groups.values()):
        raise ValueError("Expected 42 test runs in 14 groups of three repetitions")
    for values in groups.values():
        values.sort(key=lambda x: (x["repetition_index"], x["run_id"]))
    return dict(sorted(groups.items()))


def validate_paths(groups):
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


def telemetry_rows(groups):
    """Describe aligned physical resources; raw telemetry is absent here."""
    rows = []
    for scenario, runs in groups.items():
        representative = runs[0]
        seen = set()
        for demand in ordered_demands(representative):
            for link in ordered_path(demand):
                if link["link_id"] in seen:
                    continue
                seen.add(link["link_id"])
                rows.append({
                    "scenario_key": scenario,
                    "rho_wire": representative["metadata"]["rho_wire"],
                    "link_id": link["link_id"],
                    "source_switch": link["source"],
                    "source_egress_interface": link["source_interface"],
                    "destination_switch": link["destination"],
                    "destination_ingress_interface": link["destination_interface"],
                    "direction": link["link_id"],
                    "metric": "Prometheus/QFX link/queue telemetry",
                    "status": "unavailable",
                    "source": "telemetry artifacts referenced by schema.json; not present in neutral export/workspace",
                    "measurement_window": "unavailable",
                    "unit": "unavailable",
                    "aggregation_rule": "mean/std over three repetitions if raw aligned measurements become available",
                    "repetitions_expected": 3,
                    "repetitions_available": 0,
                    "mean": "",
                    "std": "",
                    "minimum": "",
                    "maximum": "",
                })
    return rows


def project_links(sample, attribution_rows):
    paths = {d["demand_id"]: ordered_path(d) for d in ordered_demands(sample)}
    demand_abs = {r["attributed_demand"]: float(r["absolute_ig"]) for r in attribution_rows}
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
        result.append({
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
            "relative_projected_link_relevance": projected[link_id] / denominator if denominator else 0.0,
            "relevance_type": "semantic/projected relevance (not Integrated-Gradients link attribution)",
        })
    return result


def representative_scenarios(groups):
    by_rho = {float(v[0]["metadata"]["rho_wire"]): key for key, v in groups.items()}
    missing = [rho for rho in REPRESENTATIVE_RHOS if rho not in by_rho]
    if missing:
        raise ValueError(f"Missing deterministic representative rho levels: {missing}")
    return [by_rho[rho] for rho in REPRESENTATIVE_RHOS]


def draw_visualization(path_stem, scenario, rho, attrs, prediction):
    values = {r["attributed_demand"]: float(r["absolute_ig"]) for r in attrs}
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    colors = {"d1": "#0072B2", "d2": "#D55E00", "d3": "#777777"}
    for demand, y in (("d1", 5.8), ("d2", 4.2)):
        width = 1.5 + 18 * values[demand]
        ax.annotate("", xy=(4.2, 5), xytext=(0.9, y), arrowprops=dict(arrowstyle="->", lw=width, color=colors[demand]))
        ax.text(0.1, y, f"{demand}  |IG|={values[demand]:.4g}", va="center", color=colors[demand])
    ax.annotate("", xy=(8.8, 5), xytext=(4.2, 5), arrowprops=dict(arrowstyle="->", lw=5, color="#009E73"))
    ax.text(4.45, 5.35, "Switch A → Switch B\nshared directed link", color="#006B4F")
    ax.annotate("", xy=(1.0, 1.7), xytext=(8.8, 1.7), arrowprops=dict(arrowstyle="->", lw=1.5, color=colors["d3"]))
    ax.text(3.25, 1.25, f"d3 |IG|={values['d3']:.4g}   Switch B → Switch A", color=colors["d3"])
    ax.text(0.1, 6.65, f"XNDT semantic explanation, rho={rho:g}, target d1, zero_traffic baseline")
    ax.text(0.1, 0.25, f"Prediction={prediction:.5g} | link relevance is semantic projection | hardware telemetry: unavailable", fontsize=9)
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), dpi=180)
    fig.savefig(path_stem.with_suffix(".svg"))
    plt.close(fig)


def main():
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
        ig_groups[(row["scenario_key"], row["target_demand"], row["baseline"])].append(row)
    if len(ig_groups) != 56 or any(len(v) != 3 for v in ig_groups.values()):
        raise ValueError("Expected 56 frozen explanations with three demand attributions each")

    explanation_rows, projected_rows, documents = [], [], []
    for key, attrs in sorted(ig_groups.items()):
        scenario, target, baseline = key
        sample = test_groups[scenario][0]
        attr_by_demand = {r["attributed_demand"]: r for r in attrs}
        if set(attr_by_demand) != set(DEMAND_ORDER):
            raise ValueError("Attribution demand order is incomplete")
        projected = project_links(sample, attrs)
        shared = [r["link_id"] for r in projected if r["shared_by_multiple_demands"]]
        for row in projected:
            projected_rows.append({"scenario_key": scenario, "rho_wire": sample["metadata"]["rho_wire"], "target_demand": target, "baseline": baseline, **row})
        demands_doc = []
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
                "physical_evidence": "per-demand measured loss aggregated over three runs; aligned link telemetry unavailable",
            }
            explanation_rows.append(record)
            demands_doc.append({**record, "path": [{**link, "logical_link": logical_link(link)} for link in links]})
        documents.append({
            "scenario_key": scenario,
            "rho_wire": sample["metadata"]["rho_wire"],
            "target_demand": target,
            "baseline": baseline,
            "model_evidence": {"type": "demand-level IG", "demands": demands_doc},
            "forwarding_evidence": {"shared_directed_links": shared, "projected_links": projected},
            "physical_evidence": {"status": "link telemetry unavailable", "measured_loss_repetitions": 3},
        })

    telemetry = telemetry_rows(test_groups)
    representatives = representative_scenarios(test_groups)

    scenario_rows = []
    for scenario, runs in test_groups.items():
        rho = float(runs[0]["metadata"]["rho_wire"])
        zero_groups = [ig_groups[(scenario, t, "zero_traffic")] for t in ("d1", "d2")]
        d3_zero = all(float(next(r for r in group if r["attributed_demand"] == "d3")["absolute_ig"]) == 0 for group in zero_groups)
        forward_set = all(set(r["attributed_demand"] for r in sorted(group, key=lambda x: -float(x["absolute_ig"]))[:2]) == {"d1", "d2"} for group in zero_groups)
        p = [float(predictions[(scenario, d)]["predicted_loss_mean"]) for d in ("d1", "d2")]
        m = [float(predictions[(scenario, d)]["measured_loss_mean"]) for d in ("d1", "d2")]
        scenario_rows.append({
            "scenario_key": scenario, "rho_wire": rho, "load_regime": "rho<=1" if rho <= 1 else "rho>1",
            "predicted_forward_loss_mean": float(np.mean(p)), "measured_forward_loss_mean": float(np.mean(m)),
            "forward_relevant_set_is_d1_d2": forward_set, "d3_absolute_ig_is_zero": d3_zero,
            "d1_d2_shared_directed_link": "sw03->sw04", "shared_link_verified_from_paths": "sw03->sw04" in path_map["d1"] and "sw03->sw04" in path_map["d2"],
            "shared_link_telemetry_status": "unavailable",
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "xndt_explanations.csv", explanation_rows)
    write_csv(args.output_dir / "xndt_projected_link_relevance.csv", projected_rows)
    write_csv(args.output_dir / "xndt_telemetry_evidence.csv", telemetry)
    write_csv(args.output_dir / "xndt_scenario_summary.csv", scenario_rows)
    (args.output_dir / "xndt_explanations.json").write_text(json.dumps({"logical_names": LOGICAL_SWITCHES, "explanations": documents}, indent=2, sort_keys=True) + "\n")
    visual_scenario = representatives[-1]
    visual_attrs = ig_groups[(visual_scenario, "d1", "zero_traffic")]
    draw_visualization(args.output_dir / "xndt_semantic_congested", visual_scenario, 1.4, visual_attrs, float(visual_attrs[0]["prediction_actual"]))

    hashes_after = {name: sha256(path) for name, path in sources.items()}
    if hashes_before != hashes_after:
        raise AssertionError("A frozen model, XAI, dataset, or prediction artifact changed")
    below = [r for r in scenario_rows if r["rho_wire"] <= 1]
    above = [r for r in scenario_rows if r["rho_wire"] > 1]
    summary = {
        "scenarios": len(scenario_rows), "congested_scenarios": len(above),
        "forward_set_d1_d2_count": sum(r["forward_relevant_set_is_d1_d2"] for r in scenario_rows),
        "d3_zero_count": sum(r["d3_absolute_ig_is_zero"] for r in scenario_rows),
        "d1_d2_converge_count": sum(r["shared_link_verified_from_paths"] for r in scenario_rows),
        "shared_directed_resource": "sw03->sw04", "reverse_control_resource": "sw04->sw03",
        "telemetry": {"status": "unavailable", "aligned_scenarios": 0, "reason": "referenced raw telemetry artifacts are absent from neutral export/workspace", "rho_le_1_behavior": "unavailable", "rho_gt_1_behavior": "unavailable"},
        "rho_le_1": {"scenarios": len(below), "mean_predicted_forward_loss": float(np.mean([r["predicted_forward_loss_mean"] for r in below])), "mean_measured_forward_loss": float(np.mean([r["measured_forward_loss_mean"] for r in below]))},
        "rho_gt_1": {"scenarios": len(above), "mean_predicted_forward_loss": float(np.mean([r["predicted_forward_loss_mean"] for r in above])), "mean_measured_forward_loss": float(np.mean([r["measured_forward_loss_mean"] for r in above]))},
        "representative_rhos": list(REPRESENTATIVE_RHOS), "integrity_hashes_before": hashes_before, "integrity_hashes_after": hashes_after,
        "integrity": {"unchanged": hashes_before == hashes_after, "model_loaded": False, "training_performed": False, "xai_recomputed": False, "telemetry_used_as_model_or_attribution_input": False},
    }
    (args.output_dir / "xndt_semantic_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
