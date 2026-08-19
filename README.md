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

*Under construction.*
