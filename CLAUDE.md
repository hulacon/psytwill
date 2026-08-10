# CLAUDE.md

## What this project is

psytwill turns the chunk × feature CSVs produced by
[word2psy](../word2psy) (text) and [viz2psy](../viz2psy) (images/video) into
**chunk-by-chunk relational matrices**: stacks of N×N similarity/distance
matrices (one per representational space — semantic embeddings, emotion
profiles, …) plus the adjacent-transition series ("coherence curves") as a
tidy time-series table. In cog-neuro terms: model RDMs over narrative/stimulus
time, RSA-ready.

psytwill consumes only the siblings' **scores CSVs, never raw stimuli** — so
it is modality-agnostic by construction: text × text, frame × frame, and
text × frame (cross-modal) share one code path. Design decisions mirror the
siblings deliberately: same CLI feel, flat-CSV-plus-`.meta.json` outputs, a
registry architecture (here a registry of *feature spaces / matrix types*
instead of models), lazy imports so `--help` stays fast.

### Sibling CSV conventions psytwill depends on

Codified 2026-08-10 as the **extractor-output convention** in
`mmmdata-agents/docs/constellation-contracts.md` §4.1 (schema_version 1.0);
that spec is authoritative. Working summary:

- Embedding columns are `{model}_{i:03d}` (`minilm_000`, `clip_text_511`,
  viz2psy's `clip_000`). word2psy's `crossmodal.py` regexes keep `clip_###`
  and `clip_text_###` apart — that pattern is reused here.
- word2psy chunk CSVs carry `chunk_idx`, `chunk_label`, `n_words`, passthrough
  columns, and optionally word-feature aggregates (`{feat}_{mean,sd,min,max}`).
  viz2psy CSVs carry `filename` / `image_idx` / `time` identifiers. The
  canonical `stimulus_id` column (§4.1) is preferred over all of these as the
  row label when present. `INDEX_COLUMNS` in `spaces.py` mirrors §4.1's
  reserved-column registry.
- NaN rows exist (e.g. word2vec OOV) and are handled explicitly, never crashed on.
- `clip_text` (word2psy) and `clip` (viz2psy) share one OpenCLIP ViT-B-32
  checkpoint — the cross-modal space. Since v0.2.0 this is **verified, not
  assumed**: `sidecar.py` loads each input's `.meta.json` (including the
  stem-family form, `X_chunks.csv` → `X.meta.json`), gates on
  `schema_version` major, and `build_quilt` refuses cross pairs whose
  recorded `models.<name>.checkpoint` strings differ. CSVs without a sidecar
  are legacy inputs: warned about, exempt from checks. Upstream provenance
  (extractor, versions, checkpoints) is passed through into
  `matrices.meta.json`'s `inputs` entries. Contract fixtures live in
  `tests/test_contract.py`.

## Architecture (`src/psytwill/`)

- **`spaces.py`** — feature-space detection, the registry analog. Two tiers:
  (1) *generic embedding detection*: any column group matching
  `{prefix}_{i:03d}` (>= 2 dims, contiguous from 000) becomes an embedding
  space — the greedy prefix regex keeps `clip_text_###` and `clip_###` apart,
  generalizing word2psy's crossmodal regex trick; (2) *named-profile
  registry* (`PROFILE_REGISTRY`): explicit patterns for emotion, sentiment,
  readability, `word_aggregates` (`{feat}_mean` columns only — mixing
  mean/sd/min/max would mix scales), and — added Aug 2026 with aud2psy
  v0.1 — the aud2psy frame-feature profiles (per-model `loudness`/`pitch`/
  `spectral`/`onsets` plus a combined 13-d `acoustic` incl. `speech_prob`;
  aud2psy's model-prefixed column naming exists to feed these patterns). `COMPATIBLE_SPACES` declares cross-modal
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
  `resolve_labels` unifies the sibling label chains: `stimulus_id` →
  `chunk_label` → `filename` → `filepath` → `image_idx` → `time` →
  `chunk_idx` → `row_{i}`.
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
- **`cli.py`** — `psytwill matrices <csv> [<csv2>] -o out/` and
  `psytwill spaces <csv>`; heavy imports deferred into the command functions
  so `--help` stays fast (~0.3 s). `--spaces minilm:euclidean,emotion`
  (subset + per-space metric), `--metric` (global), `--distance` (1 − sim),
  `--diagonal` (cross, equal N).
- **`exceptions.py`** — `PsytwillError` base; Input/Space/Metric errors;
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
     one joy) through `word2psy --all --by-sentence`; psytwill auto-detected
     6 spaces (minilm, clip_text, emotion, sentiment, readability,
     word_aggregates). Semantic RDM (minilm cosine): within-topic mean .27
     vs between-topic .03; neutral-only separation is total (within min .19
     vs between max .09). Affective RDM (emotion correlation):
     neutral–neutral r ≈ .998 while grief/joy sentences sit at r ≈ 0 to
     neutrals and −.07 to each other — the two lowest entries in the matrix
     both involve the emotional sentences.
   - *Icon demo through the general cross path*: 6 redrawn PIL icons scored
     with `viz2psy clip`, 6 words with `word2psy clip_text`;
     `psytwill matrices words.csv icons.csv` pairs `clip_text` x `clip` via
     `COMPATIBLE_SPACES`: match sims .28–.34, non-match ≤ .24, all 6 words
     rank their icon first (matches the original .29–.34 vs ≤ .25 with
     freshly drawn icons).
   - Note: `word2psy -o dir/file.csv` does not create `dir/` — pre-create
     output directories when scripting it.

### Long-term ambition: full movie input (visual + dialogue + audio)

Goal: quilt a whole film — viz2psy covers the visual stream; the gap is
dialogue and the non-verbal audio channel. Staged plan (revised Aug 2026
after reviewing the target stimulus set):

**Target stimuli**: Ben's likely movie list ([Google Doc](https://docs.google.com/document/d/1SHtS5yGZ5WcZQJSHVMX_3LlHW6BRugDopnEUBFm7z5U/edit)) —
~60 clips of 3–6 min across 10 sessions plus backups: creator-posted
indie/student animated & live-action shorts, official-channel commercial
film excerpts (Movieclips/JoBlo/Focus/Universal), and three figshare clips
from the event-schema study distribution (check whether that dataset ships
transcripts before doing any work on those). This composition killed the
original SRT-subtitle plan: creator shorts have no distributed SRTs;
commercial *excerpts* would need per-clip retiming of full-film SRTs
(unknown offsets + Ben's own cuts); and a large fraction of the animated
shorts are dialogue-free — music and sound design carry the narrative.
Whisper on the actual clip audio beats subtitle-hunting: verbatim to the
exact presented file, timestamps natively on the clip timeline, uniform
across all clips, and cleanly flags "no speech".

2. **aud2psy v0.1** (next up — design checkpoint opened Aug 2026, repo at
   `../aud2psy`): the Whisper front-end and the cheap acoustic tier
   together. Since Whisper is needed regardless and the wordless shorts
   make acoustic features the *only* auditory signal for a third of the
   stimuli, the marginal cost of the registry skeleton around the
   transcription script is small. Scope: `transcribe` (faster-whisper,
   word-level timestamps) as an *export* emitting word2psy-ready
   `text`/`onset`/`offset` CSVs, plus librosa-cheap frame-level features
   (loudness/RMS envelope — workhorse naturalistic-fMRI regressor —
   pitch, spectral, onset/tempo, speech-presence). Output mirrors
   viz2psy's video mode (row per timepoint, `time` column) so psytwill
   consumes it with zero changes. SRT ingestion is demoted to an alternate
   input format inside the transcription path (useful for full-length
   commercial films). Prior art to study: `pliers` (Yarkoni lab),
   studyforrest annotations.
3. **Time-aware cross mode in psytwill** (the real integration enabler,
   needed for any time-stamped input pair): dialogue chunks and video
   frames sample time on different irregular grids. Full cross matrices
   don't care, but same-moment comparisons, aligned coherence curves, and
   common-time-base movie RDMs need a temporal join — map chunks to frames
   by overlap, or resample both streams onto a shared grid (e.g. TR-locked,
   which fMRI wants anyway). Modality-agnostic, lives in psytwill.
4. **aud2psy v0.2 — CLAP embeddings** (the flagship, deferred until the
   skeleton is validated): **CLAP is to audio what CLIP is to images** — a
   shared audio–text space, so `clap`/`clap_text` reproduces the
   `clip`/`clip_text` cross-modal precedent and psytwill absorbs it as one
   `COMPATIBLE_SPACES` line. Also deferred to v0.2+: speaker diarization,
   prosodic-emotion models (how a line is said vs. its content — a
   different affective signal than text emotion). Division of labor stands:
   aud2psy owns acoustic/paralinguistic features + transcription export;
   verbal content stays word2psy's job — no duplication.

### Explicitly deferred (do not build without discussion)

- **Event segmentation / change-point chunking** (HMM/GSBS-style).
  Human standard for validation: narRaters Step 2 (below).
- **Entity/coreference overlap matrices** — needs NER+coref, new heavy deps.
- **LLM-judged narrative-dependence / suspense matrices** — research-grade,
  validation strategy unresolved. Human standard: narRaters Step 6 causal
  matrices (below) are the natural ground truth for a narrative-dependence
  matrix — compare model matrix to human matrix exactly as RSA compares RDMs.
- **Raw-stimulus ingestion** — revisit only if the CSV route proves limiting.

### Reference: narRaters — the human-generated standard

[narRaters](https://github.com/xianNeuro/narRaters) (Xian Li, v0.3.x,
research/non-commercial license) is a human-in-the-loop platform whose
raters produce, for a narrative, exactly the structures psytwill's
aspirational matrix types would generate computationally:

- **Step 2, event segmentation**: raters place event boundaries in a story
  transcript (optional 1–5 boundary-strength ratings) → numbered event list
  (`{story}_events.xlsx`, columns `event`, `story_texts`). Human standard for
  the deferred event-segmentation chunking (and a source of human-defined
  chunk boundaries that word2psy could score per event).
- **Step 6, causal rating**: raters score **all event pairs** on causal
  strength (0–3) through a grid interface → a completed event × event matrix
  (`{story}_causal-{method}.xlsx`). This is a human-generated relational
  matrix in psytwill's exact output shape: an aspirational
  narrative-dependence matrix should be validated by correlating against
  these (upper-triangle Spearman, standard RSA practice).
- Also relevant: Steps 4–5 parse subject recalls into clauses matched to
  story events — a route to memory-weighted matrices later.
- Ships sample narratives ("pieman_edited" — the classic Pieman stimulus,
  "the_siren", demo "lighthouse") with transcripts, event lists, and example
  recalls, but no participant rating datasets.

Integration idea (when the deferred items come up): a small reader that
ingests narRaters event lists / causal matrices into psytwill's conventions
(events as chunks; causal xlsx → labeled matrix CSV) so human and model
matrices are directly comparable in one pipeline.
