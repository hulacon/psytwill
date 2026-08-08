# CLAUDE.md

## What this project is

psyquilt turns the chunk × feature CSVs produced by
[word2psy](../word2psy) (text) and [viz2psy](../viz2psy) (images/video) into
**chunk-by-chunk relational matrices**: stacks of N×N similarity/distance
matrices (one per representational space — semantic embeddings, emotion
profiles, …) plus the adjacent-transition series ("coherence curves") as a
tidy time-series table. In cog-neuro terms: model RDMs over narrative/stimulus
time, RSA-ready.

psyquilt consumes only the siblings' **scores CSVs, never raw stimuli** — so
it is modality-agnostic by construction: text × text, frame × frame, and
text × frame (cross-modal) share one code path. Design decisions mirror the
siblings deliberately: same CLI feel, flat-CSV-plus-`.meta.json` outputs, a
registry architecture (here a registry of *feature spaces / matrix types*
instead of models), lazy imports so `--help` stays fast.

### Sibling CSV conventions psyquilt depends on

- Embedding columns are `{model}_{i:03d}` (`minilm_000`, `clip_text_511`,
  viz2psy's `clip_000`). word2psy's `crossmodal.py` regexes keep `clip_###`
  and `clip_text_###` apart — that pattern is reused here.
- word2psy chunk CSVs carry `chunk_idx`, `chunk_label`, `n_words`, passthrough
  columns, and optionally word-feature aggregates (`{feat}_{mean,sd,min,max}`).
  viz2psy CSVs carry `filename` / `image_idx` / `time` identifiers.
- NaN rows exist (e.g. word2vec OOV) and are handled explicitly, never crashed on.
- `clip_text` (word2psy) and `clip` (viz2psy) share one OpenCLIP ViT-B-32
  checkpoint — the cross-modal space.

## Architecture (`src/psyquilt/`)

- **`spaces.py`** — feature-space detection, the registry analog. Two tiers:
  (1) *generic embedding detection*: any column group matching
  `{prefix}_{i:03d}` (>= 2 dims, contiguous from 000) becomes an embedding
  space — the greedy prefix regex keeps `clip_text_###` and `clip_###` apart,
  generalizing word2psy's crossmodal regex trick; (2) *named-profile
  registry* (`PROFILE_REGISTRY`): explicit patterns for emotion, sentiment,
  readability, and `word_aggregates` (`{feat}_mean` columns only — mixing
  mean/sd/min/max would mix scales). `COMPATIBLE_SPACES` declares cross-modal
  pairs (`clip_text` x `clip`, the shared OpenCLIP space). `match_spaces`
  pairs spaces across two CSVs: same-name first, then compatible pairs.
- **`metrics.py`** — `METRIC_REGISTRY`: cosine, correlation (Pearson),
  spearman, euclidean; each carries a `form` ("similarity"/"distance").
  Pure numpy (sklearn pairwise rejects NaN); NaN rows propagate to NaN
  row/col in the output. Zero-norm and constant rows follow the crossmodal.py
  convention (similarity 0, not NaN/raise).
- **`matrices.py`** — `compute_matrix` (self or cross; defensive
  re-normalization lives in the metrics; profile spaces are z-scored per
  column — pooled across both inputs in cross mode — before scale-sensitive
  metrics cosine/euclidean), `transition_records` (adjacent off-diagonal
  band), `diagonal_records` (aligned pairs, cross mode, equal N).
  `resolve_labels` unifies the sibling label chains: `chunk_label` →
  `filename` → `filepath` → `image_idx` → `time` → `chunk_idx` → `row_{i}`.
  Input row order is taken as narrative order.
- **`pipeline.py`** — `build_quilt()`: read → detect → compute → write.
  `-o` is a **directory** (deliberate deviation from the siblings'
  single-file `-o`: the product is inherently a file set): one
  `{space}__{metric}.csv` per matrix (cross-name pairs:
  `{a}__x__{b}__{metric}.csv`), labeled rows/cols (`index_label="label"`),
  `transitions.csv` (long: boundary, from_label, to_label, space, metric,
  value) or `diagonal.csv`, plus `matrices.meta.json`.
- **`metadata.py`** — sidecar builder, viz2psy-style: embeddings described as
  `"pattern": "{name}_{NNN}"` + range, profiles as full column lists; per
  matrix: metric, form, shape, n_valid, nan_labels.
- **`cli.py`** — `psyquilt matrices <csv> [<csv2>] -o out/` and
  `psyquilt spaces <csv>`; heavy imports deferred into the command functions
  so `--help` stays fast (~0.3 s). `--spaces minilm:euclidean,emotion`
  (subset + per-space metric), `--metric` (global), `--distance` (1 − sim),
  `--diagonal` (cross, equal N).
- **`exceptions.py`** — `PsyquiltError` base; Input/Space/Metric errors;
  CLI catches and exits 1.

**To add a space**: embedding column groups are picked up automatically;
named profiles get a `PROFILE_REGISTRY` entry; cross-modal pairs go in
`COMPATIBLE_SPACES`. **To add a metric**: add a `MetricConfig` to
`METRIC_REGISTRY` (NaN-aware, pure numpy).

## Dev environment

- `uv` for environments, **Python 3.11** (system 3.14 is not for projects).
  `.venv` at repo root; recreate with
  `uv venv --python 3.11 && uv pip install -e ".[dev]"`.
- Dependencies are deliberately light: numpy, pandas, scikit-learn only.
  No model downloads; tests are offline with synthetic frames.
- Validation inputs come from the siblings' working venvs:
  `../word2psy/.venv` (fully working, caches warm) and `../viz2psy/.venv`
  (working except its saliency model).

## Working style

Ben works in explicitly approved phases: propose, wait for go-ahead at design
checkpoints, summarize finished work with concrete validation numbers.
Commit/push only when asked.

## Roadmap

1. **Phase 1 — matrices + transitions + CLI** (built Aug 2026, skeleton
   approved; options/ergonomics still open for iteration). 49 offline tests
   pass in ~0.1 s. Validated end-to-end against real sibling outputs
   (Aug 2026):
   - *Two-topic passage*: 10 interleaved ocean/finance sentences (one grief,
     one joy) through `word2psy --all --by-sentence`; psyquilt auto-detected
     6 spaces (minilm, clip_text, emotion, sentiment, readability,
     word_aggregates). Semantic RDM (minilm cosine): within-topic mean .27
     vs between-topic .03; neutral-only separation is total (within min .19
     vs between max .09). Affective RDM (emotion correlation):
     neutral–neutral r ≈ .998 while grief/joy sentences sit at r ≈ 0 to
     neutrals and −.07 to each other — the two lowest entries in the matrix
     both involve the emotional sentences.
   - *Icon demo through the general cross path*: 6 redrawn PIL icons scored
     with `viz2psy clip`, 6 words with `word2psy clip_text`;
     `psyquilt matrices words.csv icons.csv` pairs `clip_text` x `clip` via
     `COMPATIBLE_SPACES`: match sims .28–.34, non-match ≤ .24, all 6 words
     rank their icon first (matches the original .29–.34 vs ≤ .25 with
     freshly drawn icons).
   - Note: `word2psy -o dir/file.csv` does not create `dir/` — pre-create
     output directories when scripting it.

### Explicitly deferred (do not build without discussion)

- **Event segmentation / change-point chunking** (HMM/GSBS-style).
- **Entity/coreference overlap matrices** — needs NER+coref, new heavy deps.
- **LLM-judged narrative-dependence / suspense matrices** — research-grade,
  validation strategy unresolved.
- **Raw-stimulus ingestion** — revisit only if the CSV route proves limiting.
