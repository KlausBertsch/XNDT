# XNDT reproducibility artifact

This repository reproduces the data-based experiments from **“XNDT: Towards
Explainable Network Digital Twins using a High-Fidelity Hardware Testbed”**.
It starts from the published hardware measurements; access to the original
testbed is not required.

The repository contains the fixed dataset splits, the reported checkpoint,
packet-loss evaluation, Integrated Gradients and occlusion analyses, Semantic
Alignment, and Evidence Integration with the required test telemetry. Raw
hardware captures and testbed automation are deliberately out of scope.

## Citing XNDT

When referencing XNDT in academic work, please cite the following paper:

Klaus J. Bertsch and Oliver P. Waldhorst, “XNDT: Towards Explainable Network
Digital Twins using a High-Fidelity Hardware Testbed,” *Proc. Upper-Rhine
Artificial Intelligence Symposium*, Windisch, Switzerland, 2026. To appear.

Consider using the following BibTeX entry:

```bibtex
@inproceedings{bertsch2026xndt,
  author    = {Bertsch, Klaus J. and Waldhorst, Oliver P.},
  title     = {{XNDT}: Towards Explainable Network Digital Twins using a High-Fidelity Hardware Testbed},
  booktitle = {Proc. Upper-Rhine Artificial Intelligence Symposium},
  address   = {Windisch, Switzerland},
  year      = {2026},
  note      = {To appear}
}
```

## Setup

Python 3.12 is recommended; the published training run used Python 3.12.3.
The pinned environment was also verified with Python 3.13.

```bash
git clone https://github.com/KlausBertsch/XNDT.git
cd XNDT
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The processed dataset is already in `data/`; no hardware or raw-data download
is needed. File hashes and the deterministic split assignment are recorded in
`data/dataset_manifest.json` and `data/dataset_split.csv`.

## Reproduce the reported results

Run the commands from the repository root:

```bash
python -m xndt.evaluate \
  --dataset-dir data --training-dir model \
  --output-dir outputs/predictions

python -m xndt.ig_test \
  --dataset-dir data --training-dir model \
  --output-dir outputs/xai

python -m xndt.semantic \
  --dataset-dir data --xai-dir outputs/xai \
  --prediction-dir outputs/predictions --training-dir model \
  --output-dir outputs/semantic

python -m xndt.evidence \
  --dataset-dir data --telemetry-dir data/telemetry \
  --semantic-dir outputs/semantic --xai-dir outputs/xai \
  --prediction-dir outputs/predictions --training-dir model \
  --output-dir outputs/evidence
```

The main reference values are:

| Result | Expected value | Generated artifact |
|---|---:|---|
| Test MAE | 0.0017258 | `outputs/predictions/test_metrics.json` |
| Test RMSE | 0.0049767 | `outputs/predictions/test_metrics.json` |
| IG completeness checks (tolerance `1e-3`) | 55/56 | `outputs/xai/ig_test_summary.json` |
| IG/occlusion forward-demand agreement | 56/56 | `outputs/xai/ig_test_summary.json` |
| Test scenarios with d1/d2 on the shared resource | 14/14 | `outputs/semantic/xndt_semantic_summary.json` |
| Test telemetry joins | 42/42 runs | `outputs/evidence/xndt_complete_telemetry_summary.json` |

The primary human-readable results are the figures and JSON summaries. The CSV
files in the prediction, XAI, and semantic stages are compact bridge artifacts
for the following pipeline stage; `xndt_complete_scenario_summary.csv` is the
final scenario-level evidence table. Small floating-point differences between
platforms are possible.
The 512 integration intervals were selected on the validation split, where all
24 completeness checks passed the `1e-3` tolerance. The test-split completeness
count was not used to select this setting.

## Retrain the model

Retraining is optional for the main reproduction path:

```bash
python -m xndt.train \
  --dataset-dir data --output-dir outputs/retraining --seed 42
```

The training script derives normalization statistics only from `train.jsonl`,
uses the fixed validation split for early stopping, and does not open the test
split. To repeat the validation-only IG step-selection analysis, run:

```bash
python -m xndt.ig_convergence \
  --dataset-dir data --training-dir model \
  --output-dir outputs/ig-convergence --seed 42
```

## Code and data provenance

`routenet_fermi/loss_model.py` is the unchanged RouteNet-Fermi model from
[BNN-UPC/RouteNet-Fermi](https://github.com/BNN-UPC/RouteNet-Fermi), commit
`ef72a37bb6f677e149f354f088f5836f29c862b5`. The XNDT adaptation is kept
separately in `xndt/model.py`; the remaining files in `xndt/` implement the
offline experiment pipeline.

The recorded testbed data do not contain physical queue capacities or
buffer-occupancy measurements. Since the queue configuration remained fixed
throughout all experiments, the RouteNet-Fermi queue-size input is replaced by
a shared trainable initialization value in the XNDT model. This parameter is
not interpreted as a physical queue capacity, and the reported results do not
establish generalization across different queue or buffer configurations.

The Integrated Gradients analysis follows the method introduced by
Sundararajan, Taly, and Yan, [“Axiomatic Attribution for Deep
Networks”](https://proceedings.mlr.press/v70/sundararajan17a.html), ICML 2017.
The implementation in `xndt/integrated_gradients.py`, together with the
XNDT-specific convergence and test protocols, was written for this artifact;
no source files from the authors' reference implementation are included.

The code is licensed under Apache-2.0. The processed data in `data/` is
licensed separately under CC BY 4.0. See `LICENSE`, `NOTICE`, and
`data/LICENSE-DATA.md`.
