# psytwill

Chunk-by-chunk relational matrices from [word2psy](https://github.com/hulacon/word2psy)
and [viz2psy](https://github.com/hulacon/viz2psy) feature CSVs: N×N
similarity/distance matrices per representational space (RSA-ready model RDMs)
plus adjacent-transition "coherence curves" over narrative/stimulus time.

Since v0.4.0 psytwill also produces the long-form **feature table** — the
other half of the Contract B consumer surface — aggregating any number of
extractor CSVs into one tidy `(stimulus_id, modality, extractor, model,
feature[, time], value)` table (parquet or CSV):

    psytwill features clip.csv ebind.csv caption.csv -o features.parquet
    psytwill timelines sub-01_ses-*_task-TB*_events.tsv --registry stimuli/stimulus_registry \
        --features features.parquet --context-model clip -o sub-01_timeline.parquet

### Per-column labels (Contract B 1.2)

Extractor sidecars at schema 1.2 label every feature column with its
`interpretability` (`nameable` | `opaque`) and `derivation` (`formula`,
`formula-on-model`, `trained-on-labels`, `probe`, `zero-shot`, `embedding`,
`bookkeeping`). psytwill carries them into the features table's sidecar, drops
`bookkeeping` columns from every block fit, and can fit a block on a labelled
subset of columns:

    # patch an existing table after `<extractor> sidecar refresh` (JSON only)
    psytwill features --refresh-labels features.parquet
    # fit on nameable columns that are not probes or zero-shot similarities
    psytwill space fit --features features.parquet --block V -o space/ \
        --select 'interpretability=nameable;derivation!=probe,zero-shot'
    # stimulus-free per-column table + BIDS-style column dictionary
    psytwill space labels --space space/V_v1.json -o V_labels.csv

`--select` clauses are `KEY=a,b` (keep values in the set) or `KEY!=a,b` (drop
them), ANDed with `;` or by repeating the flag; keys are `interpretability`
and `derivation`. Selection is per column, and each member is whitened on its
selected columns alone. The manifest records the expression, each member's
resolved labels, and `circular_with` for columns derived from another
member's checkpoint. `space release write` writes the labels table and column
dictionary beside the release.

*Under construction.*
