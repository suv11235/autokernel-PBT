# Generation experiments

Experiments in which an agent writes or optimizes a kernel and a correctness gate decides what it
keeps. The pipelines these measure (G0–G5, O1) are in
[`docs/pipelines/generation.md`](../../docs/pipelines/generation.md).

| Directory | What it is |
|---|---|
| [`autoresearch-pbt/`](./autoresearch-pbt/) | karpathy/autoresearch's loop on a NumPy GPT, gated by the four oracle arms (pipeline G1), plus the saboteur matrix that pins what the gate admits |
| [`log-ratio-headroom/`](./log-ratio-headroom/) | The log-ratio law's statistic on every softmax we hold, device runs included: design input for E2 |
| [`e2-log-ratio-law/`](./e2-log-ratio-law/) | E2, softmax half: the log-ratio law in the gate, against every kernel set the pre-registration names |
