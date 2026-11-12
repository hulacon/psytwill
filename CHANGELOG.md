# Changelog

Releases before 0.47.0 are recorded in the commit history (`git log`), one
commit per version.

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
