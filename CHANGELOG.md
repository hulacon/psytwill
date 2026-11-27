# Changelog

Releases before 0.47.0 are recorded in the commit history (`git log`), one
commit per version.

## 0.74.0 — 2026-11-27

Minor: one extraction file per recording (rule C). Packing retires.

### Changed
- **`stage.py` stages one unit per recording**, layout `per-recording` in
  `inputs/layout.json`, wavs under `<scratch>/<corpus>/recordings/`. A
  recording is the staging group: a clip, a chapter, an episode, a track, a
  movement, a word.
  - Unit ids are `<prefix><group>` (e.g. `spk103-1240`). Clip corpora (clotho,
    avcaps) use the clip's own id, so a unit's `stimulus_id` is the clip's,
    and the caption tables join on it.
  - Packing several recordings into a unit is no longer staged, and
    `--target-minutes` is gone. `pack_groups` stays in the library, and
    packed roots on disk still read as before.
  - `--per-item` still stages into a 0.2.x per-item root (the older name for
    the same layout).
- **`extract.py`** treats `per-recording` as it treats `per-item`: batched
  aud2psy calls, and the centre rule's one-group-per-unit check.

### Added
- **`staging.split_recordings`** re-stages one packed unit as per-recording
  units. Each is checked to be laid out exactly as `item_units` would lay it
  out, so old time `t` is new time `t − shift`. A one-group unit stays as it
  is, unless `keep_single=False`. `staging.ONE_RECORDING_LAYOUTS` names the
  layouts.
- **`stage.py <corpus> --restage-from OLD_DURABLE`** re-stages a packed root
  into fresh roots:
  - split units are cut sample-exactly from the packed wav, with their
    segment and transcript rows shifted and re-keyed;
  - units holding one recording are kept (clip corpora excepted);
  - captions are copied;
  - held-out markers must agree;
  - `inputs/recording_map.csv` (new ↔ old unit and stimulus id, group,
    `shift_sec`, `unchanged`) is written. It is the join that lets unwindowed
    features carry from the packed extraction.

## 0.73.0 — 2026-11-27

Minor: the pass manifest, and framing parity on each file's box.

### Added
- **`psytwill.passmanifest`:** one declared extraction pass per release. A
  TOML lists every cell (corpus × source × models), each with:
  - its stage (`pass` / `screen` / `top-up`);
  - the log entry that decided it;
  - its producer version pin;
  - optionally, a sha-pinned unit subset (required for screens).

  A cell added after `frozen` must carry a dated `deviation`.
  `PassManifest.record()` is what a release will record.
- **`extract.py --pass-manifest`** (or `$PSYTWILL_PASS_MANIFEST`):
  - `run` refuses cells outside the pass, units outside a cell's pinned
    subset, and a producer whose installed version is off its pin. Each
    refused cell is listed, and the exit code is non-zero.
  - `plan` reports every pass cell against the driver (`ok` / `NO DRIVER
    SOURCE` / unknown models), and the driver cells the pass does not cover.
  - Without a manifest, `run` says so and runs as before.

### Changed
- **`extract.py is_done` compares each video file's box, not the framing
  table's sha** (fit ↔ projection framing parity, per file). A video cell is
  done when its sidecar's stage-0 version and frame source match, and each
  file's crop, pixel aspect and stored size match that file's row in the
  current framing table. A new table version that keeps a file's box no
  longer re-extracts it. A file absent from the table is not done. Done
  counts are unchanged on the existing framed corpora.

## 0.72.0 — 2026-11-27

Minor: the corpus registry becomes typed bookkeeping, and a release records it
(release schema 1.6). It is warn-only: no fit, relation, bench read or
membership pass refuses on it.

### Added
- **Typed `fitcorpus.CorpusSpec` fields:**
  - `licence` and `media` (`local` / `fetch` / `fetch-gated` / `manual`);
  - `roles`, each a `Role(kind, product, split)` with kinds `fit` / `gate` /
    `register` / `dev` / `bar` / `report` / `screen` / `brain`, and a
    `member-training` role naming the probe or detector the corpus trained or
    calibrated;
  - per-member `training_overlap` (`none` / `present` (split) / `undisclosed` /
    `unchecked`, with a `*` default);
  - `dropped`, a reason: the key stays registered so minted ids still parse.

  Checked at import.
- **Registered:** `coco` (draws outside NSD), `tvc`, `vidor`, `song-describer`,
  `paraspeechcaps`, `peterson`, `meadows`, `giordano`, `envisounds`,
  `semeval-2022`, `semeval-2026` and `veatic`. `peoples-speech` and `jamendo`
  are marked dropped.
- **`psytwill.registry`:**
  - the consumer's **usage file**, `<data root>/registry/usage.toml`. It holds
    the corpus's overlap with the consumer's own stimuli, by grain (`recording`
    / `stimulus` / `item` / `concept` / `none`), and per-split `read_by`
    history. These are data, not code. A release with a data root needs the
    file and names its path when it is missing;
  - the uses a release's manifests show (block corpus rows, relation
    registers);
  - the conflict matrix as warnings: `undeclared-use`, `declared-not-used`,
    `unregistered`, `dropped-used`, `unrecorded`, `dev-and-bar`, `same-rows`,
    `bar-read`, `member-training`, `consumer-overlap`, `no-gate` / `gates`,
    `training-overlap`. Analytic members (no checkpoint) are skipped for
    training overlap.
- **Release schema 1.6, `registry`:** the table, the usage file's path and
  sha256, the uses, the analytic members and the warnings. `space release write`
  takes `--registry-usage`. `write` and `verify` both print every warning, and
  verify recomputes them from what the release recorded.

### Changed
- AVCaps's registry text now matches its use: V↔A fit (train), and A↔L / V↔L
  coupling registers over every clip.

## 0.71.0 — 2026-11-27

Minor: AVA becomes a fit corpus.

### Added
- **`fitcorpus.CORPORA["ava"]`** (tier b): AVA films, minutes 15:00–30:00, one
  segment per film. Native id = 10 hex of sha1(YouTube id), because YouTube ids
  are case-sensitive and may start with `-`.
- **`extract.py` sources `ava/frames`** (every V model, `motion` included, no
  `caption_vlm`) and **`ava/audio`** (the full A battery). Video and audio are
  separate files under `<durable>/ava/inputs/{video,audio}/`, paired by YouTube id.
- **`extract.py inputs` writes `ava/inputs/ids.csv`**: native ↔ YouTube id and
  each film's declared clock. The video keeps its original timestamps (first
  frame at the keyframe before 895 s) and the audio starts at 900 s, so a frame
  at video time t sits at t + `video_start_s` − `audio_start_s` on the audio's
  clock. Consumers pairing V and A rows must apply it.

## 0.70.1 — 2026-11-23

Patch: two per-row Python loops vectorised. Results are unchanged, and each is
checked against the loop it replaced in the test suite.

### Changed
- **`windows.item_masks`** counts group changes with a prefix sum: a run of
  segments holds two groups exactly when it contains an adjacent change. It no
  longer builds a set per row.
- **`compare._overlap_from_knn`** (neighbour overlap, its permutation null, and
  readout's k-NN controls) counts shared neighbours as an equality count over
  each row's k × k pairs, in row chunks. It falls back to per-row sets when a
  supplied list repeats an index within a row (`knn_indices` never does).
  `neighbor_overlap_null` checks that once, not once per permutation.

## 0.70.0 — 2026-11-19

Minor bump: a group-bootstrap interval for readout set reads.

### Added
- **`readout(..., n_boot=, boot_level=)`** gives each set read (`strict`,
  `nameable`) a percentile interval for its held-out R² (`SetRead.r2_ci`):
  whole groups (e.g. films) resampled with replacement over the read's
  out-of-fold predictions, no refit. Needs `groups`. Control draws and
  per-member reads carry none. Readout schema 1.2.
- **`compare.group_bootstrap_r2(y, pred, groups, ...)`**: that interval for any
  out-of-fold predictions, recomputing variance-weighted R² from per-group sums.

### Changed
- `ridge_predictivity` now delegates to a private `_ridge_fit` that also keeps
  the out-of-fold predictions. Its result is unchanged.

## 0.69.0 — 2026-11-18

Minor bump: coupling-only registers on a frozen relation.

### Added
- **`psytwill space relate registers`** (`relate.add_registers`) adds a
  relation's coupling on rows it was not fitted on, such as another corpus, as
  registers beside the fit's own.
  - Inputs: a relation manifest and two paired score tables, checked against
    the relation's block manifests and views exactly as `relate check` and
    `relate project` check them.
  - The rows are projected through the **frozen** map. Their joint moments go
    into a **new** manifest + weights version (`--stem`, required), and k, the
    map and every fit statistic are copied unchanged.
  - Register names: `--register NAME` (every row) or `--registers-from-label`
    (the corpus parse `relate fit` uses).
  - `RelationFit.conditional` reads an added register exactly as it reads a fit
    register.
- Why: `relate fit --registers-from-label` keeps moments only for registers
  among the fit rows. A single-corpus relation therefore carried one register,
  and the off-register coupling a prior needs had no place in the manifest.
- Refused:
  - writing over the source manifest or over any existing version;
  - a register name the relation already has (a fit register is never
    overwritten);
  - labels that do not match the paired rows.

### Changed
- **`relate_schema_version` 1.4.**
  - Every `per_register` entry carries `fit`: true for a fit register, false for
    an added one.
  - Added entries also record `key`, `n_dropped_nan` and `tables` (both score
    tables, sha256-pinned).
  - The manifest's `derived_from` lists the manifests it was derived from,
    oldest first.
  - An entry without `fit` (schema ≤ 1.3) is a fit register.

## 0.68.0 — 2026-11-15

Minor bump: an event-boundary task.

### Added
- **`psytwill bench boundary`** (`bench/boundary.py`): does a film grid move more
  at annotated event boundaries than at cut-matched non-boundary bins?
  - Inputs: a grid model (comma-separated models are joined side by side in raw
    units on their common bins), `--boundaries` (`stimulus_id, level, onset,
    annotator`), `--cuts` (`stimulus_id, time`), and `--durations`
    (`stimulus_id, duration`, each film's scored span).
  - Statistics per bin: `change` (cosine distance between consecutive rows, with
    rows centred on the mean over all films) and, given `--sfa-kernel` (`W`,
    `lam`), `pe_sfa` (AR(1) prediction error of the slow features, `rho = 1 -
    lam / 2`, per-film centring).
  - A boundary's window is ±`--tol` s. Controls lie further than `--exclude` s
    from every boundary at every level and are matched on distance to the
    nearest cut (calipers .25 / .5 s, at least 3 controls, otherwise the
    boundary is dropped). Deltas are in raw and film-SD units.
  - Film effect: the mean over annotators of film means, with an
    annotator-stratified sign-flip p (one-sided) and an annotator-stratified film
    bootstrap CI, also for boundaries within and beyond `--tol` of a cut.
- **`bench compare`** for boundary runs: paired `a - b` and `a / b` per
  statistic, in film-SD units, on the boundaries both runs kept, with films
  resampled within annotator on shared draws.

## 0.67.0 — 2026-11-15

Minor bump: both retrieval sides can be pooled grids; subsets and retention in
the retrieval comparison. Existing outputs and their numbers are unchanged
(`compare` draws the same resamples as before for `a - b`).

### Added
- **`bench retrieval --query-segments / --query-window`**: the query side can be a
  time grid mean-pooled into segments, as `--target-segments` already did for the
  target. A film's sound and its picture over the same segments is then one
  within-stimulus retrieval.
- **`--subset NAME=FILE`** (repeatable) on `bench retrieval` and on `bench compare`
  for retrieval runs: the statistics again over the items whose stimulus or item
  label is listed. Maps and candidate sets stay those of the full run; only the
  averaging and the resampling are restricted to the subset's own units.
- **Retention in `bench compare` (retrieval):** `(a - 0.5) / (b - 0.5)` on
  `pct_beaten`, computed within each paired draw, with `retention_defined: false`
  when b's interval reaches chance. Both arms' own intervals from the same draws are
  reported too.
- `bench.core.paired_bootstrap`, `parse_subsets` and `in_subset`, and
  `bench.retrieval.summarize` (the summary of any items table or subset of one).

## 0.66.0 — 2026-11-15

Minor bump: two bench tasks for human judgements on items.

### Added
- **`psytwill bench rdm`** (`bench/rdm.py`): a model's item distances
  (`1 - cosine` of item prototypes) against human dissimilarity judgements, from a
  pairs table `participant, image_a, image_b, dissimilarity`. The score is the mean
  over participants of Spearman(model, participant), with both noise-ceiling bounds
  (lower: leave one participant out; upper: against the mean of all) and the score
  as a share of each. The CI resamples **items**: each draw rebuilds every RDM from
  the drawn items and drops pairs of an item with its own duplicate, because pairs
  share items and a pair bootstrap is too narrow. Participants are fixed and
  reported one by one. Draws depend only on the seed and the item set, so runs
  share them. Refuses self-pairs, a pair judged twice by one participant (in
  either order) and judged items with no embedding.
- **`psytwill bench rating`** (`bench/rating.py`): a mapped ridge from a model's
  coordinates to each item's mean rating (ratings table `participant,
  stimulus_id, <dimension>...`), item folds, scored by Pearson r of the
  out-of-fold predictions over items, with an item-bootstrap CI. Each dimension
  carries a split-half reliability between participants (random halves,
  Spearman-Brown, mean over splits) and `r_ceiling = sqrt(reliability)`.
- **`bench compare`** for both: the paired `a - b` and the ratio `a / b` on the same
  item draws. The ratio is retention against a reference with chance 0; it is
  flagged `ratio_defined: false` when the reference's own interval reaches 0.

## 0.65.0 — 2026-11-15

Minor bump: the fit-corpus driver leaves attached models out by default.

### Changed
- **`extract.py run` / `plan` / `aggregate` skip attached models** (battery
  `attached=True`: `context_affect`, `experiential`,
  `lexical_norms_experiential`). They ship beside the space and are in no
  block, so the fit corpora never need them. `--with-attached` brings them
  back, and `--model <attached>` still runs that one model. Screen drivers that
  call `build_sources` directly are unaffected.

## 0.64.0 — 2026-11-15

Minor bump: the centre rule for per-item corpora, and block declarations.
Space schema 1.20. Nothing already fitted or extracted changes.

### Added
- **`psytwill.windows.outside_items`** (the centre rule): a row is outside when
  its centre (bin start + hop / 2) lies in no segment's `[onset, offset)` of its
  unit. It drops the rows that belong to no item: a per-item clip's grid padding
  (a 0.53 s word fills a 1.0 s clip) and the silence between packed items.
  Units with no segments table keep every row.
- **`extract.py centre-rule --corpus C | --table PATH --out <list>`**: one row
  exclusion list (`stimulus_id,time` on the binned grid, `reason =
  centre_outside_item`) over the named corpora's frames tables, per-item or
  packed, for `space fit` / `space project --exclude-rows`. `--table` reads a
  table in another root. It refuses no named corpus, a corpus named twice, a
  per-item unit holding more than one item, a table without recorded segments,
  and overwriting a written list with different rows. The sidecar pins each
  frames table by sha256 with its layout and its unit, row and excluded counts.
- **`space fit --declare TEXT`** (repeatable): statements about the block's
  inputs or domain, recorded verbatim as `declarations` in the manifest (space
  schema 1.20). They are not copied by `--like`.

## 0.63.0 — 2026-11-14

Minor bump: per-item staging and batched extraction for the packed short-item
A corpora (E5's "re-extract per item"). Nothing
already staged or extracted changes.

### Added
- **`psytwill.staging.item_units`**: one unit per group (a word in one voice,
  a track, a movement), id `<prefix><group>`, no trailing gap (the unit ends on
  the first grid point at or after its last file). A windowed A member computed
  on such a unit reads only its own item, so E5's item masks find nothing to
  mask. `layout_unit` gains `trailing_gap` (default: `gap`, as before).
- **`stage.py --per-item`** for twp-unpresented, fma and musopen only. Wavs go
  to `<scratch>/<corpus>/items/` (never beside packed `units/`); each unit keeps
  a one-row segments table with the corpus's columns; `inputs/layout.json`
  records the layout. A durable root holds one layout per corpus: staging the
  other layout over it is refused.
- **`extract.py run`, per-item corpora:** aud2psy cells run batched, one
  `aud2psy --inputs-from` call per model per `--batch-items` units (default
  500), each model loaded once and each unit writing its own family and
  sidecar. Producers (diarize, transcribe) run first; a consumer's manifest
  row names its unit's stored `diarize_speakers.csv` /
  `transcribe_transcript_words.csv`, so the diarize speech gate, conversation
  and speech_rate read each item's own tables (needs aud2psy >= 0.24.0). A
  unit whose producer table is missing is skipped and reported.
  `--unit-slice START:END` (plan and run) takes a slice of each source's
  units, for array tasks over items.


## 0.62.0 — 2026-11-13

Minor bump: shared and unique subspaces of a fitted block, and their
split-half reproducibility gate (`psytwill.shared_unique`, `space
subspaces`). Numbered after 0.61.0 (`feat/attached`).

### Added
- **`psytwill.shared_unique`**: under `pr` member scaling the block PCA is
  multi-set CCA, so a direction's eigenvalue is 1 + how much the members share
  it. The module reads a block as subspaces:
  - `S`, the leading block directions with eigenvalue >= 1 + c;
  - `U:<m>`, member m's directions that keep >= `unique` of their variance
    after regressing m on every other member;
  - `U`, their union.
  Every object is a linear read of the block's concatenated member
  coordinates with an orthonormalising basis from the fit rows, so its scores
  are uncorrelated and unit-variance there. `S` is placed by the block's own
  least-squares route, so a row with an absent member still gets `S`. A member
  absent from >= `max_absent` of the rows gets no unique subspace.
  `reproducibility` compares each object's two half-fit estimates by mean
  squared canonical correlation on common rows, against the mean of random
  span pairs of the same dimension. `draw_subspaces` makes the random spans
  a reader compares them with: paired d-dimensional draws inside `S` and `U`
  (d = min of their dims), random d-dimensional reference spans of the
  block's coordinates, and per-member draws inside `S` at each unique
  subspace's dimension. Every draw is orthonormal on the fit rows.
  `save_subspaces` / `load_subspaces` /
  `project_subspaces` write, read and place rows (subspaces schema 1.0).
- **`psytwill space subspaces --like <frozen manifest> -o DIR`**: re-reads the
  frozen fit's inputs and rows as `space fit --like` does, and refuses them if
  the row count or any member's columns changed. It estimates the objects on
  the frozen map, refits the block on each half of the units (sha256 parity
  of `<seed>:<stimulus id>`), and gates every object (defaults: c 0.5, unique
  0.5, msc >= .5 and >= random + .25 over 20 pairs), and saves 20 draws of
  each kind with the objects (`--draws`, `--draw-seed`). It writes
  `<stem>.{json,npz}` and `<stem>_gate.csv`. The json records the S/U overlap
  and how far the refit's top-k span is from the frozen one's.
- **`space project --subspaces <json> [--objects a,b]`**: writes every
  subspace object's (and draw's) orthonormal coordinates on a corpus's rows,
  as `<object>__<jjj>` columns. Rows are loaded, aligned, masked and chunked
  as for block scores; an object that cannot place a row leaves NaN there. It
  refuses subspaces estimated on another fit (manifest sha256) and `--k` /
  `--view`. The `.meta.json` lists each object's kind, dim, columns and gate
  verdict.
- **`fit_block(keep_rows=True)`** keeps the members' prepared rows and the
  final map's row weights on the returned fit (`BlockFit.rows`,
  `row_weights`; never saved). **`BlockFit.aligned`** is the alignment and
  masking `BlockFit.project` uses, now callable on its own.

### Changed
- `cli._space_fit_setup` holds everything `space fit` reads before it fits;
  `space fit` behaves as before.


## 0.61.0 — 2026-11-13

Minor bump: releases carry **attached members** (release schema 1.5).
(Numbered after the open 0.59.0 / 0.60.0 branches.)

### Added
- **`space release write --attached SPEC.json`**: opt-in producer features shipped
  beside the space and in no block, relation or refit. The spec is
  `{"root": <stimulus feature store>, "attached": [{model, basis, gap, evidence,
  columns?, name?, package?, root?, note?}, ...]}`.
  - `basis` is `bar` (met the pre-registered attached bar) or `decided` (attached by a
    decision without the bar), kept apart so a reader can tell the two.
  - `gap` and `evidence` are required.
  - `columns` selects base feature columns; each must be declared by every sidecar of
    the model (default: all declared). Chunk aggregates follow their base column.
  - Every sidecar under the root that records the model, and every table of its
    Contract B §4.1 family, is pinned by sha256. A file is claimed by the longest
    sidecar base it matches.
  - Refused: an unknown basis or field, an empty gap or evidence, a duplicate, a model
    with no sidecar, an undeclared column, and a model that is a member of a released
    block.
- `release verify` / `load_release` refuse a release whose attached files changed or
  went missing, and `release write` refuses an attached entry naming a battery model
  that is not marked attached.
- **Battery 0.4 (clamped 2026-11-13): attached models.** `BatteryModel.attached`; three
  models are pinned as attached: `context_affect` (viz2psy), `experiential` (viz2psy)
  and `lexical_norms_experiential` (word2psy; the nested prefix is documented).
  - An attached model passes `check_sidecar`, so its tables may sit in group tables.
  - It is never a block member: `space fit` leaves it out of a block's default members
    and refuses it in `--members`. It may still be an outside target.
  - `vitl14` (the ViT-L-14 embedding cache the heads read) is listed NOT_ADMITTED,
    and `context_affect` leaves NOT_ADMITTED. The live-registry clamp tests pass again.
  - `battery --json` rows carry `attached`.

### Changed
- `RELEASE_SCHEMA_VERSION` 1.4 → 1.5; release manifests always carry `attached` (a
  list, possibly empty).

## 0.60.0 — 2026-11-13

Minor bump: the readout's R² read sees a set in the caller's metric, as its
captured share and k-NN read already did (readout schema 1.1); and
`psytwill.directions`, per-direction member energy, universality and
split-half reliability of a fitted block.

### Changed
- **`psytwill.readout.readout`**: the ridge no longer z-scores each predictor
  column. Before, the three reads of one set used two metrics: the captured
  share and the Euclidean k-NN graph took the columns as given, the R² read
  took them z-scored. Two consequences:
  - A control draw is an arbitrary orthonormal basis of its span, with
    unequal coordinate variances. Z-scoring undid the variance match the
    control was drawn for, and its R² depended on which basis the draw came
    in.
  - A member's prior weight in the ridge grew with its column count, not
    with its variance in the caller's metric.
  R², `r2_per_fold`, `control_r2`, `r2_percentile` and the per-member R²
  columns change for any battery whose columns are not already equal in
  variance. Records carry `params.ridge_scaling = "set"`; `READOUT_SCHEMA`
  is 1.1.

### Added
- **`psytwill.directions`**, per-direction properties of a fitted block
  (`BlockMap`):
  - `member_energy`: each direction's squared loading on each member's
    whitened columns, normalised over members (long form, with member width).
  - `universality`: u_d, the participation ratio of a direction's member
    energies over the member count (1/M = one member, 1 = all equally).
  - `direction_reliability(reference, half_a, half_b, align=)`: rho_d, the
    correlation of each direction's scores between two half fits after
    aligning both to the reference, by one orthogonal Procrustes rotation
    (default) or by sign only (strict: a swap of near-equal eigenvalues
    scores low).
  - `direction_table`: `component, eigenvalue, u[, rho]`, one row per direction.
  - `python -m psytwill.directions <manifest> --out <table> [--energy <long>]`.
  - Disclosed: inside a degenerate eigenspace the PCA's basis is arbitrary,
    and so is u. PR whitening gives every kept member direction unit variance,
    so private directions of different members can sit at near-equal
    eigenvalues.
- `compare.ridge_predictivity(scaling=)`: `"column"` (default, unchanged:
  per-column z-score) or `"set"` (centre, then divide by one number, the root
  mean column variance from the training fold, so the alpha grid applies).
  With `"set"` the read depends on X's geometry only: rotating X's columns
  leaves R² unchanged. `space` and `geometry` keep `"column"`.

### Tests
- Rotating an anisotropic member's columns leaves every read unchanged:
  share, k-NN overlap, R², and each per-member R². Before the fix, R² moved
  (0.134 → 0.138 on the planted battery).
- A control-shaped predictor (`X @ B`, `B` orthonormal) gives the same R² in
  any basis of its span under `"set"`, and a different one under `"column"`.
- `test_directions.py`: energies and u worked by hand; u bounds; on a fitted
  block, a latent two members share has u = 2/3 of three and a private one 1/3;
  u changes with the basis of a degenerate eigenspace; rho high for stable
  directions and near 0 for noise; sign alignment is strict where Procrustes
  absorbs a swap; refusals.
## 0.59.0 — 2026-11-13

Minor bump: an outside target can come from its own tables and can be scored
without narrowing the block's rows. Space schema 1.19.

### Added
- `space fit --target-features TABLE ...`: tables read for the `--targets` only,
  never for members; allowed with `--like`, so a candidate extracted outside the
  battery is scored against an existing fit's exact inputs, rows and folds.
  Recorded as `target_inputs`.
- `space fit --target-rows {narrow,block}` / `fit_block(target_rows=)`. `narrow`
  (default) is the existing behaviour: the fit's rows are those every member and
  target hold. `block`: the members' rows alone. A target is absent (masked, never
  filled) from a block row its tables lack, and from rows on which it is
  undefinable, where `narrow` would drop that row for the whole block. The
  target's NaN refusal, column drops and support are decided on its own rows.
- Manifest `target_rows` {`mode`, `rows_absent`}.

## 0.58.1 — 2026-11-12

Patch: `scripts/fitcorpus/extract.py` lists NSD units again.

### Fixed
- `Roots.__init__` sets `nsd_hdf5` (from `--nsd-hdf5` / `$PSYTWILL_NSD_HDF5`). Since
  0.50.0 the assignment sat after the `return` in `framing_sha256()`, so it never
  ran, and listing `nsd/image` units (`plan`, `run`, `aggregate`) raised
  `AttributeError`. Regression test added.

## 0.58.0 — 2026-11-12

Minor bump: a readout of an opaque target through a battery's nameable
members, each read against a matched random-subspace control.

### Added
- **`psytwill.readout.readout(battery, target, nameable=, strict=, groups=)`**:
  how much of a target table (a network layer, another model, a brain
  response, on the battery's rows) the nameable members recover.
  - Two reads per member set: held-out grouped ridge R² (global) and k-NN
    overlap against a block-permutation null (local), Euclidean on both
    sides by default.
  - The control: random subspaces of the whole battery matched to the set's
    dimensionality and captured variance share
    (`psytwill.subspace.matched_subspaces`), read through the same two
    measures, reported as a percentile.
  - `strict` (a subset of the nameable members) is read first with its own
    control, so strict and full form a nested curve.
  - Per member: R² alone, without, and unique (full minus without).
  - Non-finite rows are dropped and counted, never filled. The record
    (`Readout.to_dict()`) carries `schema` `1.0` and every parameter.
  - No CLI verb yet, and no thresholds: what a share means is the caller's.

## 0.57.1 — 2026-11-12

Patch: account for viz2psy's new `context_affect` model.

### Changed
- `NOT_ADMITTED["viz2psy"]` lists `context_affect` (viz2psy 0.22.0): a
  candidate V member, extracted for its admission screen only. Without the
  entry, the live-registry cross-check (`test_live_registry_matches_clamp[viz2psy]`)
  fails once viz2psy ships the model. Listing it before that release is
  harmless: the check only subtracts names.

## 0.57.0 — 2026-11-11

Minor bump: LibriSpeech held-out splits for a separate held-out root.

### Added
- **`stage.py librispeech --split {train-clean-100,test-clean,test-other}`**
  (default `train-clean-100`, the fit split). A test split is staged exactly
  as the fit split, packed per speaker with curated transcripts.
- **Held-out root guard:** a test split stages only into a durable root
  holding a `HELDOUT` marker file, and the fit split is refused in one. A
  held-out root is laid out like the fit store, so `extract.py` runs on it
  unchanged via `--durable-root`, and no fit (which reads the fit store
  only) can see it. The test splits are speaker-disjoint from
  train-clean-100, so their `ext-librispeech-spk<N>-<k>` ids cannot collide.
- `fetch.py librispeech --subset test-other`.

### Fixed
- `battery.NOT_ADMITTED` lists aud2psy's `phonetics` (new in aud2psy
  0.23.0), so the live-registry check passes again: the model exists, but it
  is not a battery member unless its admission screen admits it.

## 0.56.0 — 2026-11-10

Minor bump: transcript QC flags for the Whisper fit corpora, and the L
exclusion list built from them.

### Added
- **`psytwill.transcript_qc`**: flags on a transcript table keyed
  `(stimulus_id, chunk_idx)`, where `chunk_idx` is row order within
  `stimulus_id` (how word2psy numbers the chunks it scores).
  - `loop_exclude`: chunks 2..n of a run of `N_LOOP` (5) or more consecutive
    chunks with identical normalised text. The first chunk of the run is kept.
    Shorter runs carry `run_len` and are kept.
  - `timing_compressed`: duration < 0.1 s or > 15 words/s. Flag only.
  - `lower_style`: lowercase, unpunctuated text, plus a per-stimulus share.
    Flag only.
  - `attach_asr` joins decoder confidences back on
    `(stimulus_id, onset, offset, text)` and refuses an ambiguous decoder table.
  - `RULES_VERSION` changes with any rule constant; written tables record it.
- **`extract.py qc`**: writes `<corpus>/qc/transcript_flags.parquet` (+
  `.meta.json` with the rules, and each input's sha256) for every corpus whose
  L text is Whisper output (the video corpora and `STAGED_WHISPER`). A corpus
  with some units lacking inputs is refused unless `--allow-partial`.
- **`extract.py exclusions`**: one `--exclude-rows` file = `--base-rows` ∪
  every Whisper corpus's loop rows, with a `reason` column and a sidecar. It
  refuses flags older than their inputs or written under other rules, and
  refuses to overwrite an existing list with different rows.

## 0.55.0 — 2026-11-10

Minor bump: AVCaps V goes through viz2psy's video path, as viz2psy video sets.

### Added
- **`extract.py` source `avcaps/video`:** AVCaps clips through viz2psy's video
  path, so the framing table applies, frames every 0.5 s, with the frame
  models minus `motion` (as for the stills). A cell is one viz2psy video set
  (viz2psy >= 0.20.0): `AVCAPS_VIDEO_UNIT` (100) whole clips of one split per
  call, in native-id order, each model loaded once per call.
  - Clips are read from `<media root>/avcaps/<split>_videos/<split>_videos/`,
    spelled as found there (the framing table's keys).
  - Units are `<split>-NNN`; rows are keyed `ext-avcaps-<native id>` through
    `--id-pattern`.
  - A clip whose name is not a native id fails at planning.
  - A cell counts as done only when its sidecar has the current table's
    sha256 and lists exactly the unit's clips, in order.
- **`psytwill.sidecar.VIDEO_INPUT_TYPES`** = (`video`, `video_set`).
  `input_framing_rule` reads a video set's rule (recorded once under
  `input.framing`) exactly as a single video's, so set and single-video
  outputs under one table are uniform.

### Changed
- `avcaps/frames` (staged PNG stills, which bypass the framing table) is
  marked superseded by `avcaps/video` in its description. It stays in the
  plan, so the 0.1.0 outputs remain addressable.
- `is_video_cell` accepts several video inputs (a set), not just one.

## 0.54.0 — 2026-11-09

Minor bump: the blender fit corpus gets a fetch verb and an extraction path.

### Added
- **`fetch.py blender`** (`--media-root` / `$PSYTWILL_FITCORPUS_MEDIA`):
  the Blender open movies that Blender serves as free official single files
  (Big Buck Bunny, Sintel, Caminandes 2, Elephants Dream), one file per film
  as `<media root>/blender/<native><ext>`, with a `MANIFEST.sha256`. Each
  download is checked against its pinned size (a changed file is an error).
  The files go to durable media, not scratch, because the framing table keys
  on their paths.
- **`extract.py`: blender runs like movie10 and friends**: `frames` (V +
  `caption_vlm`), `audio` (A), `transcript` (Whisper) and `vlmcap`. Units are
  one per film, with ids `ext-blender-<native>`. The three corpora share
  `VIDEO_CORPORA`. The `_cneuromod_*` unit builders became `_video_*`;
  `_cneuromod_media` is kept.

## 0.53.0 — 2026-11-09

Minor bump: battery 0.3 admits viz2psy 0.18.0's `caption_vlm`; the fit-corpus
extractor runs it on video corpora and scores its text.

### Added
- **Battery 0.3** (clamped 2026-11-09 against viz2psy 0.18.0): `caption_vlm`,
  a text-kind producer like `caption` (column `caption_vlm_text`, string;
  checkpoint `Qwen/Qwen3-VL-8B-Instruct@<revision>`). Nested-prefix exception
  `(caption_vlm, caption)`, resolved by the longest-match rule. Text-kind
  models never enter a block basis, so no fit changes.
- **`scripts/fitcorpus/extract.py`:** `caption_vlm` joins the movie10 and
  friends `frames` sources, and a new word2psy `vlmcap` source (prefix
  `vlmcap_`, every L model) scores each segment's `caption_vlm.csv`.
  `caption_vlm` runs at `--batch-size 32` (the reference run's), every other
  viz2psy model at 64.
- **GPU guard:** `run` refuses, before any cell, a selection holding
  `caption_vlm` cells unless every visible GPU's name contains
  `$CAPTION_VLM_GPU` (default `H100`; empty turns it off). Greedy bf16
  captions differ across GPU types, so one campaign uses one type. The error
  names the fix (`--skip-model caption_vlm`).

### Changed
- `V_MODELS` excludes viz2psy models by kind (`text`) instead of by the name
  `caption`, so no captioner can reach NSD stills or AVCaps frames.

## 0.52.0 — 2026-11-08

Minor bump: read-time window handling for time-keyed tables, driven by the
Contract B 1.4 window labels (0.51.0). Space schema 1.17 → 1.18.

### Added
- **Item masks.** Packed short-file corpora concatenate items with a silence
  gap (`staging`), so a windowed column's value near an item boundary was
  computed over two unrelated items. `space fit` and `space project` now make
  the member **absent** from a row where a column it kept read a window
  across two groups of the unit's segments table: the "absent unless
  complete" path, never a fill and never a row drop for the other members.
  A deselected column cannot mask its member. Per member
  `item_masked_rows`; top-level `item_masks`.
- **Segments record.** `psytwill features --segments-dir DIR` (at build) and
  `--refresh-segments DIR` (JSON only) record the staging `*_segments.csv`
  tables a features table was extracted against, each with its sha256, keyed
  by the `stimulus_id` inside the file. A missing or changed table refuses
  the read. `scripts/fitcorpus/extract.py aggregate` records them for every
  packed corpus. A table with no record is read as one item per clip, and
  the manifest says which tables had none.
- **`space project --causal`.** Each centred windowed column is shifted
  later by half its window (rounded to the grid; the rounding is recorded),
  so a row's value reads only input up to the row's end. The first
  half-window of each clip has no value: the member is absent there. A value
  whose source row was item-masked stays masked where it lands. Refuses a
  fit without window labels. The projection sidecar records `causal`,
  `window_reads` (shifts, item-masked and absent rows per member) and the
  segments per table.
- `BlockFit.project(..., absent=)` and `fit_block(..., absent=)`; a member
  made absent on a row never triggers that row's `undefinable` drop.

## 0.51.0 — 2026-11-08

Minor bump: Contract B 1.4 window labels on the consumer side (aud2psy
0.22.0 declares, per column, how much audio each row reads). Space schema
1.16 → 1.17, release schema 1.3 → 1.4.

### Added
- `window_sec` and `window_align` are label keys: validated (seconds > 0,
  `grid` or `event`; a number needs `window_align` `centred`/`trailing`),
  carried through `features`, and structural, so two inputs of one model
  whose windows differ (including declared vs undeclared) refuse to stack.
  They appear in every manifest's per-column labels and in the release labels
  table.
- `space fit --select` takes `window_sec` clauses: `window_sec<=0.5` (bounds
  `<=`, `<`, `>=`, `>`; `grid` counts as 0 s, `event` fails every bound) and
  `window_sec=event` / `!=grid`. A window clause refuses a member whose
  columns carry no window labels, naming the refresh to run.
- Per member (space schema 1.17): `window_sec` is the longest window over
  the columns in the fit (seconds, `grid` or `event`), with `window_align`,
  `window_columns` (grid / event / windowed counts),
  `window_undeclared_columns`, and `window_source: "labels"`. Labels must
  agree with a model-level declaration (producer `window_sec` or registry
  pin) where both speak.

### Changed
- **A battery entry with no window pin is undeclared, not "no window".**
  Before, `resolve_member_windows` returned `("battery", None)` for it and
  `release write` accepted that as grid-native, so tonal (3 s), rhythm
  (3 / 8.92 / 4 s), onsets_tempo (8 s) and psychoacoustic_fluctuation (2 s)
  would have been certified as unwindowed. The registry now pins `"grid"`
  explicitly (loudness, pitch, spectral, timbre, egemaps, speech, motion) and
  tonal at 3.0; models whose columns use different windows are not pinned
  and need 1.4 labels. `release write` refuses a time-keyed block member that
  is undeclared, including a 1.14–1.16 manifest's `("battery", null)`.
- `member_windows` in a release (schema 1.4) may hold `"grid"` / `"event"`.

## 0.50.0 — 2026-11-08

Minor bump: Contract B 1.3 input framing on the consumer side (viz2psy
0.17.0's stage 0: bar crop + square pixels from a reviewed framing table).
Features table schema 1.2 → 1.3 (additive).

### Added
- `psytwill features` records each input's framing rule (`inputs[].framing`:
  `frame_source`, `stage0`, `stage0_version`, `framing_table_sha256`) and the
  rule the table's video inputs share (`input_framing`; `null` without video
  inputs). It **refuses to stack video inputs framed under different rules**:
  a different table, a different stage-0 version, or a video extracted
  before stage 0 existed mixed with one after it (a partial re-extraction).
  A table of only pre-0.17.0 video inputs is legacy and still builds. Stills
  are never compared.
- The space member loader (`space fit`, `project`, `check`, ...) refuses member
  tables whose video framing rules differ, and refuses a table built before
  0.50.0 that holds viz2psy inputs when another table declares a rule
  (rebuild it with `psytwill features`; aggregation is cheap).
- `scripts/fitcorpus/extract.py`: `--media-root` / `$PSYTWILL_FITCORPUS_MEDIA`
  for raw source video (CNeuroMod is read there, spelled as found, matching the
  framing table's keys; `--scratch-root` keeps staged, derived media), and
  `--framing-table` / `$PSYTWILL_FRAMING_TABLE`, passed to every viz2psy cell
  that reads a video file (required for those cells). A video cell counts as
  done only if its sidecar was framed with the current table's sha256.

## 0.49.1 — 2026-11-05

### Fixed
- `labels.inventory_matcher` now accepts every inventory a model entry
  declares, not only the top level / `features`: word2psy's
  `chunk_aggregates` columns and `chunk_pooling`'s pooled pattern and
  `count_column`. 0.49.0 refused valid producer labels keyed on those
  columns (`<model>_n_pooled`, `{feature}_{stat}`), so
  `features --refresh-labels` failed on every table holding a pooled or
  aggregated word2psy model.

## 0.49.0 — 2026-11-05

Minor bump: Contract B schema_version 1.2 per-column `labels` on the consumer
side. Three schemas gain fields: features table 1.1 → 1.2, space manifest
1.15 → 1.16, release 1.2 → 1.3. Every older file still reads.

### Contract B 1.2 labels

- **Reader** (`psytwill.labels`). Each model's `labels` block (`default` plus
  `columns`, merged key by key; a `null` override removes a default key)
  resolves to one label set per column. `interpretability`, `derivation` and
  `derived_from` are checked against the contract's vocabularies, and
  `derived_from` must be present exactly for `formula-on-model`, `probe` and
  `zero-shot`. A `columns` key must be a column of the model; an embedding
  model's `pattern` inventory (`clip_{NNN}`) is matched as a pattern. A
  malformed producer block fails `features` loudly, naming the sidecar and
  model. A 1.0 or 1.1 sidecar reads as unlabelled.
- **Features table carry-through** (features schema 1.2). The sidecar records
  `model_labels` ({model: block or null}) and `model_labels_from` (the
  sidecar whose block was kept). Each input's model entry gets
  `labels_sha256`, a digest rather than the block, so a table of many inputs
  stays small. **Stacking rule:** two inputs of one model whose columns
  resolve to a different `interpretability`, `derivation` or `derived_from`
  are refused, as is a labelled input stacked with an unlabelled one. A
  difference only in `Description`, `LongName`, `Units`, `Levels`,
  `TermURL` or `source_validation` warns, and the newer sidecar
  (`created_at`) wins. `space fit` applies the same rule across tables.
- **`features --refresh-labels TABLE...`** patches a table's sidecar from its
  inputs' refreshed producer sidecars, reading only their headers. It is JSON
  only and combines with `--refresh-nulls`. `--refresh-nulls` no longer
  lowers a newer table's `schema_version` to 1.1.
- **Bookkeeping columns are never swept into a block.** `space fit` drops
  `derivation: bookkeeping` columns from every member and outside target
  before anything else, and records them as `per_member.<m>.bookkeeping_columns`.
  `space project` and `space check` select a fit's recorded columns by name,
  so a frozen fit is unaffected by labels its tables gain later.
- **`space fit --select EXPR`** keeps only member columns whose labels match:

      EXPR   := CLAUSE (';' CLAUSE)*        clauses ANDed; --select may repeat
      CLAUSE := KEY OP VALUE (',' VALUE)*   a value list is a set
      KEY    := interpretability | derivation
      OP     := '=' (in the set) | '!=' (not in the set)

  e.g. `--select 'interpretability=nameable;derivation!=probe,zero-shot'`.
  Selection is column-level and runs before the null policy: each member is
  whitened on its selected columns alone (its PR, rank and block weight are
  the subset's), only the selected columns' `nulls` act (an unselected
  column's NaN neither refuses nor masks, an unselected `undefinable` column
  drops no rows), and `max_nan_frac` and minimum support run on the selected
  columns. A member with no matching column is dropped from the block and
  listed in `selection.dropped_members`. A member with no labels is refused,
  naming the tables to `--refresh-labels`. Outside targets are scored whole
  (minus bookkeeping). `--like` copies the source's selection unless
  `--select` is given (`like.selection_from`).
- **Space manifest** (space schema 1.16): `selection` (null, or
  `expression`, `clauses`, `scope`, `dropped_members`), `labels_source`, and
  per member `labelled`, `checkpoint`, `labels` ({column: resolved labels}),
  `bookkeeping_columns`, `deselected_columns` and `circular_with`
  ({column: [members]}: the column's `derived_from` equals another member's
  checkpoint, a flag and never a refusal). Per target: `labelled` and
  `bookkeeping_columns`.
- **Export.** `space labels (--space MANIFEST... | --release RELEASE) -o
  labels.csv` writes a stimulus-free per-column table (block, member, column,
  `in_fit`, the labels, `circular_with`) and, beside it,
  `labels.columns.json`: a BIDS-style column dictionary with one entry per
  column. The BIDS keys (`LongName`, `Description`, `Units`, `Levels`,
  `TermURL`) sit at the top of each entry, with `interpretability`,
  `derivation`, `derived_from` and `source_validation` as extra keys.
  `space release write` writes both beside the release
  (`<stem>_labels.csv`, `<stem>_labels.columns.json`) when any pinned block
  recorded labels, and names them under `labels` (release schema 1.3).

### Back-compat

- With no `--select`, and no labelled bookkeeping column, a fit's weights are
  bit-identical to 0.48.0's.
- Space manifests of schema ≤ 1.15 load, project and release as before. They
  carry no labels, so `space labels` refuses them and a release of only such
  blocks records `labels: null`.
- Features tables of schema ≤ 1.1 read as unlabelled.

### Also in this release

- **`space project --k K` projects through the first K block components.**
  K runs from 1 to the fit's k, and the default is the fit's k, so existing
  calls are unchanged. A row with every member present gets the first K
  columns of the full projection. A row with a masked member is placed when
  its present members span K directions, so a smaller K places rows the
  fit's k leaves out. For example, a fixed-k audio block truncated close to
  its concatenation rank otherwise places only the rows where every gated
  member is present. The sidecar's `k` is the number of columns written,
  which is what `relate` checks. `fit_k` records the fit's own k when the
  two differ. `BlockFit.project` takes the same `k`.
- **`scripts/setup_env.sh --prefix` no longer rewrites the shared records.**
  A build at a non-shared prefix used to overwrite psytwill's committed lock
  manifests (`conda/`) and every family repo's `.conda-prefix`, repointing
  them at the personal build. It now leaves both alone unless `--record` is
  given. A build at the shared prefix writes them as before. Script only; the
  package is unchanged.

## 0.48.0 — 2026-10-16

Minor bump: relation manifests gain a field (schema 1.2 → 1.3), and a mix
that was silent is now refused or corrected.

### Changed outputs

- **Relation sides record their score view.** `space relate fit` writes
  `sides.<a|b>.view` (`covariance` or `whitened`, from the score table's
  sidecar), and `relate_schema_version` is 1.3. `space relate project` writes
  the input's `view` into its sidecar.
- **`space release project` feeds each relation side the view it was fitted
  on.** The release places blocks in the covariance view; a relation side
  fitted on whitened scores was handed those covariance scores and returned
  wrong coordinates without an error. It now gets the block's whitened scores,
  matching `space relate project` on a `--view whitened` table. A relation
  fitted on covariance scores is unaffected.

### Now refused (previously silent)

- **A score table in the other view.** `space relate project` and
  `space relate check` refuse a table whose view differs from the side's,
  and the message names the `--view` to re-project with.
- **A sidecar naming an unknown view** is refused at read.

### Back-compat

- A score-table sidecar without `view` was written before `space project
  --view` existed (0.44.0) and is read as `covariance`, the only view there
  was. A relation manifest without `sides.*.view` (schema ≤ 1.2) is read the
  same way. Such relations keep working on covariance tables and now refuse
  whitened ones.

## 0.47.0 — 2026-10-15

Minor bump: several fixes below change values that existing commands write.
Outputs produced by 0.46.0 in the affected cases should be regenerated.

### Changed outputs

- **`bench reconstruct` keeps whole-second bins.** Score tables written by
  `space project` carry string key columns (`"1.0"`), which never matched the
  member's respelled labels (`"1"`); every integer-valued time bin dropped out
  of the fit. `n_fit_rows`, the map and its CV R² change. Labels that collapse
  onto one spelling are now refused.
- **Nested-interval grid bins are kept** (`project`). A bin inside a long
  interval was dropped once a shorter interval nested in it had ended; it now
  goes to the latest-starting interval that contains it.
  `n_overlapped_bins` counts any earlier containing interval.
- **`matrices` pairs profile columns by name** in cross mode. Inputs whose CSVs
  order the same profile columns differently were compared column by column
  in the wrong order; inputs with different column sets of equal size are now
  refused.
- **`stimulus_id` strings are preserved.** Scores CSVs (`features`,
  `matrices`) and CSV feature tables read by `timelines` keep numeric-looking
  ids as written (`"001"`, not `1` or `1.0`).
- **`compose` numbers `chunk_idx`/`word_idx` once per stream**, across every
  text store, so one index names one item in every model.
- **`compose` fills empty grid bins per model.** A model with no rows for an
  item now gets explicit NaN rows at that item's bins, like any other empty
  bin, instead of no rows.
- **`--drop-ids` is honoured by `bench oddoneout` and `bench congruence`**
  (it was recorded in `params` but not applied).
- **`bench nextwindow` counts the horizon and gap in time bins**, not rows, so
  dropped NaN bins no longer make "next window" seconds away. The CLI passes
  `--window`; direct `next_window()` calls without `window=` keep row counting.
- **`timelines` pools replicate item-level rows by mean** (several captions of
  one image, a chunk's words), as `store.load_spaces` does, instead of keeping
  the first row.
- **Places dimension 416 → 467** in the battery (365 scene + 102 `sunattr`
  columns), following viz2psy's Places365 label fix (hulacon/viz2psy#4), which
  stopped every `/x/<name>/indoor|outdoor` class collapsing onto bare
  `indoor`/`outdoor` (314 scene columns instead of 365). The checkpoint string
  is unchanged, so `battery --check` cannot tell old and new stores apart.
  The battery is therefore re-clamped as **battery 0.2** (clamped
  2026-10-15, viz2psy pinned at 0.13.0, the release carrying the fix); fits
  and records citing battery 0.1 were founded on the 314-column inventory.
  **Fitted spaces whose members include `places` and were fitted on pre-fix
  Places columns must be refit**; projecting a new store through such a fit
  refuses (`lacks ... feature(s) the fit needs`).
- `compose` viewer media: same-named audio files in different directories
  (per-voice files) are no longer decoded once and reused.
- `SpaceMatrix.modality`/`extractor` load as `None`, not the string `"nan"`,
  for tables without a sidecar.

### Now refused (previously silent)

- `compose` checks unmatched stimulus ids before publishing anything, so a
  refused run no longer leaves tables behind that make the identical re-run
  report `up_to_date`.
- `compose` raises on an unreadable store sidecar instead of recording
  `checkpoint: null`.
- `timelines` refuses two non-BIDS events files with the same name stem, which
  would otherwise share one clock.
- `relate` refuses null join keys, which were dropped uncounted.
- `scripts/fitcorpus/extract.py aggregate` checks completeness per model, not
  per unit.

### Fixed

- `relate` score columns sort numerically (`V_999`, `V_1000`), not as
  strings; this matters only from k = 1000.
- Wrong-length `groups` in `compare` raise `SpaceError`, not `IndexError`.
- An empty registry cell no longer crashes the `viz movies` bundle build.
- `compose --modality-map` validates entries like `features` does.

### Requirements

- Python >= 3.11 and scikit-learn >= 1.8 (scikit-learn 1.8 dropped 3.10, so a
  3.10 install resolved a different scikit-learn API). The MDS compatibility
  shim for scikit-learn < 1.8 is removed.
