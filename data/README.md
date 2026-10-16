# Published XNDT data

The JSONL files are the fixed train, validation and test splits used for the
XNDT experiments. The split is scenario-level, uses seed `20260808`, and keeps
all three repetitions of a scenario together. `dataset_split.csv` records the
complete assignment.

Only fields consumed by training, inference, explanations or result
interpretation are included. IP and MAC addresses, traffic-generator server
addresses, endpoint host identifiers, the full testbed topology and raw
artifact paths were removed. Logical switch and directed-link identifiers are
retained because they define the graph and are required for semantic evidence
integration.

`telemetry/` contains only the held-out test measurements needed by the final
evidence-integration step. The raw hardware captures are not part of this
repository.

The contents of this directory are licensed under CC BY 4.0; see
`LICENSE-DATA.md`.

