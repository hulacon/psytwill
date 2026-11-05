# Changelog

Releases before 0.47.0 are recorded in the commit history (`git log`), one
commit per version.

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
