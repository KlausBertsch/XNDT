"""Adapt XNDT JSONL records directly to RouteNet-Fermi inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEMAND_ORDER = ("d1", "d2", "d3")
DETERMINISTIC_TIME_DIST_ID = 1  # TimeDist.DETERMINISTIC_T
FIFO_POLICY_ID = 3  # POLICIES = ['WFQ', 'SP', 'DRR', 'FIFO']
CONTINUOUS_FEATURES = (
    "traffic",
    "packets",
    "length",
    "capacity",
    "eq_lambda",
    "avg_pkts_lambda",
    "exp_max_factor",
    "pkts_lambda_on",
    "avg_t_off",
    "avg_t_on",
    "ar_a",
    "sigma",
)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _ordered_demands(sample):
    if len(sample["demands"]) != len(DEMAND_ORDER):
        raise ValueError(
            f"Expected exactly {len(DEMAND_ORDER)} demands, "
            f"got {len(sample['demands'])}"
        )
    by_id = {d["demand_id"]: d for d in sample["demands"]}
    if set(by_id) != set(DEMAND_ORDER):
        raise ValueError(f"Expected exactly {DEMAND_ORDER}, got {tuple(by_id)}")
    return [by_id[name] for name in DEMAND_ORDER]


def adapt_sample(sample, as_tensor=True):
    demands = _ordered_demands(sample)
    link_ids, capacities, paths = [], {}, []
    for demand in demands:
        ordered = sorted(demand["path"], key=lambda x: x["order"])
        if [x["order"] for x in ordered] != list(range(len(ordered))):
            raise ValueError(f"Invalid path order for {demand['demand_id']}")
        path = []
        for item in ordered:
            lid, cap = item["link_id"], float(item["capacity_bps"])
            if lid not in capacities:
                link_ids.append(lid)
                capacities[lid] = cap
            elif capacities[lid] != cap:
                raise ValueError(f"Inconsistent capacity for {lid}")
            path.append(link_ids.index(lid))
        paths.append(path)
    users = [[] for _ in link_ids]
    for p, path in enumerate(paths):
        for pos, link in enumerate(path): users[link].append([p, pos])

    def column(values):
        return np.asarray(values, np.float32).reshape(-1, 1)

    zeros = column([0.0] * len(DEMAND_ORDER))
    inputs = {
        "traffic": column(
            [d["traffic"]["configured_wire_rate_bps"] for d in demands]
        ),
        "packets": column(
            [d["traffic"]["configured_packet_rate_pps"] for d in demands]
        ),
        "length": np.asarray([len(x) for x in paths], np.int32),
        "model": np.asarray(
            [DETERMINISTIC_TIME_DIST_ID] * len(DEMAND_ORDER), np.int32
        ),
        "eq_lambda": zeros.copy(),
        "avg_pkts_lambda": zeros.copy(),
        "exp_max_factor": zeros.copy(),
        "pkts_lambda_on": zeros.copy(),
        "avg_t_off": zeros.copy(),
        "avg_t_on": zeros.copy(),
        "ar_a": zeros.copy(),
        "sigma": zeros.copy(),
        "capacity": column([capacities[x] for x in link_ids]),
        "policy": np.asarray([FIFO_POLICY_ID] * len(link_ids), np.int32),
        "priority": np.zeros(len(link_ids), np.int32),
        "weight": column([0.0] * len(link_ids)),
        "link_to_path": paths,
        "queue_to_path": [list(x) for x in paths],
        "queue_to_link": [[i] for i in range(len(link_ids))],
        "path_to_link": users,
        "path_to_queue": [[list(pair) for pair in x] for x in users],
    }
    target = np.asarray([d["label"]["loss_ratio"] for d in demands], np.float32)
    meta = {
        "demand_ids": list(DEMAND_ORDER),
        "link_ids": link_ids,
        "run_id": sample["run_id"],
    }
    if as_tensor:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required for as_tensor=True") from exc
        ragged = {"path_to_link": 1, "path_to_queue": 1}
        for name in set(inputs) - set(ragged):
            inputs[name] = tf.convert_to_tensor(inputs[name])
        for name, rank in ragged.items():
            kw = {} if rank is None else {"ragged_rank": rank}
            inputs[name] = tf.ragged.constant(inputs[name], dtype=tf.int32, **kw)
        target = tf.convert_to_tensor(target)
    return inputs, target, meta


def iter_split(dataset_dir, split="train", as_tensor=True):
    for sample in read_jsonl(Path(dataset_dir) / f"{split}.jsonl"):
        yield adapt_sample(sample, as_tensor)


def compute_train_normalization(dataset_dir, output_path=None):
    values = {x: [] for x in CONTINUOUS_FEATURES}
    for sample in read_jsonl(Path(dataset_dir) / "train.jsonl"):
        inputs, _, _ = adapt_sample(sample, False)
        for name in values:
            values[name].extend(np.asarray(inputs[name]).reshape(-1))
    stats = {"source_split": "train", "features": {}}
    for name, vals in values.items():
        arr = np.asarray(vals, np.float64)
        raw_std = float(arr.std())
        stats["features"][name] = {
            "mean": float(arr.mean()),
            "std": raw_std or 1.0,
            "constant": raw_std == 0.0,
            "count": int(arr.size),
        }
    dest = Path(output_path or Path(dataset_dir) / "xndt_normalization.json")
    dest.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return stats


def calculated_link_loads(inputs, meta):
    traffic = np.asarray(inputs["traffic"]).reshape(-1)
    capacity = np.asarray(inputs["capacity"]).reshape(-1)
    return {
        lid: float(
            sum(traffic[p] for p, _ in inputs["path_to_link"][i]) / capacity[i]
        )
        for i, lid in enumerate(meta["link_ids"])
    }
