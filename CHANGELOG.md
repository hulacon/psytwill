# Changelog

Releases before 0.47.0 are recorded in the commit history (`git log`), one
commit per version.

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
