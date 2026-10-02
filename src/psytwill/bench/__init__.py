"""Benchmark harness: score any stimulus embedding on a task, one protocol for every model.

A *model* here is whatever a feature table carries under one ``model`` name —
a battery member (``ebind``, ``clip``), a psytwill-space family
(``pspace_v``), or a baseline built by psytwill's own code path. The harness
never knows which; every model goes through the same task code, the same
folds and the same bootstrap, so a difference between two scores is a
difference between the embeddings.

Tasks (each module exposes one scoring function and writes per-item rows, so
a CI is always recomputable from the output alone):

- ``retrieval``   — cross-modal identification between two keyed sets
- ``oddoneout``   — triplet odd-one-out against human choices
- ``nextwindow``  — identify a film's next window among its other windows
- ``congruence``  — per-pair cross-modal similarity scored against
  per-pair human outcomes (ratings, memory)

Every task has a ``zero-shot`` mode (raw cosine; only meaningful when both
sides share one trained space, e.g. EBind image vs EBind text) and a
``mapped`` mode (cross-validated ridge between the two sides, held-out
items only), which is the mode that is comparable across models.
"""

BENCH_SCHEMA_VERSION = "0.1"
