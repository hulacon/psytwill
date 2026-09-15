# Neuroscientific models and receptive-field scaffolding

Status: aspirational research and design notes, not an approved implementation specification\
Discussion date: 2026-09-15

## Purpose

Survey how psytwill and its sibling extractors could support more psychological and neuroscientific models, especially population receptive-field (pRF) estimation. The proposed direction is to preserve spatial layout and temporal context, add interpretable models of feature integration, and compare their predictions with behavior or neural responses.

This complements [Event structure from multimodal stimulus features](event-structure-notes.md) and its [data-resource survey](event-structure-data-resources.md). It does not supersede that modeling ladder or authorize implementation.

## Starting point and scope

The ecosystem already contains substantial psychological scaffolding:

- [viz2psy](https://github.com/hulacon/viz2psy): Gabor-based GIST, a spatial saliency grid, low-level image statistics, motion, faces, depth, and learned visual representations.
- [aud2psy](https://github.com/hulacon/aud2psy): pitch, rhythm, timbre, psychoacoustics, vocal affect, speaker turns, and audio embeddings.
- [word2psy](https://github.com/hulacon/word2psy): sensorimotor and other lexical norms, contextual surprisal, affect, and semantic embeddings.
- [psytwill](../README.md): consumes feature tables and produces representational similarity/distance matrices and adjacent-transition curves.

The sibling descriptions above reflect their GitHub documentation reviewed on the discussion date. The original survey examined the older local psytwill snapshot at `274ea18`. Before publication, the checkout was updated to `52944b6` (v0.18.0), which already includes long-form feature tables, presentation timelines, and composition of sparse runs onto a movie grid. The timing recommendations below should therefore extend and audit those existing components, rather than assume alignment must be built from scratch. The sensory-field and response-fitting proposals remain future directions; this survey is not an exhaustive audit of current capabilities.

Many additional psychological models can already be expressed as feature tables. Other models require structure that a flat vector or an RDM does not preserve. The important distinction is between:

1. **Stimulus representations:** what information is present, with coordinates and temporal support.
2. **Observer models:** assumptions about pooling, integration, adaptation, prediction, or latent state.
3. **Estimated response parameters:** quantities fitted to neural or behavioral observations.

A model-derived stimulus feature is not automatically a measurement of an observer's psychological state. Similarly, an RDM is useful for representational comparisons but is not sufficient input for every encoding model.

## What would it mean to get pRF maps from embeddings?

Three possible products should be distinguished:

| Product | Required evidence | Interpretation |
|---------|-------------------|----------------|
| Responses through a bank of hypothetical receptive fields | Spatial stimulus features and chosen pooling functions | Stimulus-derived model predictions; no neural data needed |
| Receptive fields of artificial network units | Network access and controlled spatial probes | Properties of the artificial network |
| Individual cortical pRF maps | Neural responses, stimulus geometry and timing, fitted model, anatomical correspondence | Estimated spatial response properties of measured neural populations |

A conventional visual pRF describes the region of visual space that drives a measured neural population. Its center determines eccentricity and polar angle, and its spread describes receptive-field size. These are estimated response properties, not intrinsic image labels. See [Dumoulin & Wandell, Population receptive field estimates in human visual cortex](https://stanford.edu/~wandell/data/papers/2007-Dumoulin-NI.pdf).

Whole-frame embeddings may predict neural responses well without identifying spatial tuning. A similarity matrix further abstracts away the explicit feature coordinates. Neither should be treated as enough to recover individual cortical pRF maps without additional spatial information and response data. A pretrained neural-response model could supply predictions, but those would inherit its training population and assumptions rather than measure a new individual's pRFs.

### Feature-weighted receptive fields as the bridge

The most direct extension is to retain a feature field:

```text
frame × vertical position × horizontal position × feature channel
```

For each voxel, fit which spatial region to pool over and which channels within that region predict its response. The [feature-weighted receptive field framework of St-Yves & Naselaris](https://pmc.ncbi.nlm.nih.gov/articles/PMC5886832/) separates spatial pooling parameters from feature-tuning weights and supports both interpretable filters and learned feature maps.

A schematic linear version is:

```text
pooled_feature[k, t] = sum_xy spatial_pool(x, y; center, spread) * F[t, y, x, k]
neural_drive[t] = sum_k feature_weight[k] * pooled_feature[k, t]
predicted_BOLD = baseline + HRF_convolution(neural_drive)
```

This is an illustrative model, not a proposed fixed implementation. Nonlinearities, feature preprocessing, and hemodynamic assumptions must be specified and compared. Spatial pooling parameters in a feature-based model also require interpretation relative to the receptive fields of the underlying feature extractor.

### Required components

1. **Spatial feature exports from viz2psy.** Begin with local contrast and Gabor energy maps; later consider spatial network activations. The existing 24 × 24 saliency grid is a useful format precedent, but predicted fixation density is a different construct from contrast or feature drive. A spatial token address does not guarantee local information when a network mixes context across positions.
2. **Explicit geometry.** Preserve the original frame, crop, resize, padding, display dimensions, viewing distance, coordinate origin, and axis directions. Pixels become visual degrees only with display calibration. State whether coordinates are image-, screen-, or retina-relative.
3. **Fixation or gaze information.** Controlled fixation establishes a reference frame. Free-viewing movies require eye-position information or explicit assumptions; screen-centered fits should not silently be labeled retinal pRFs. Predicted saliency is not a substitute for measured gaze.
4. **Presentation timing and responses.** Use actual presentation timing and neural data. For fMRI, model the hemodynamic response and acquisition sampling. Preserve source timing before any common-grid aggregation.
5. **Fitting and model comparison.** Start with a simple Gaussian pooling baseline. Later compare alternatives such as [compressive spatial summation](https://pmc.ncbi.nlm.nih.gov/articles/PMC3727075/). Separate spatial, feature, and temporal parameters sufficiently to assess identifiability.
6. **Validation and anatomical export.** Use held-out runs or clips, parameter uncertainty/stability, and voxel or surface correspondence. Independent retinotopy localizers would provide an especially useful check before interpreting movie-derived estimates as interchangeable with conventional pRF maps.

Natural movies have correlated spatial and semantic content, editorial cuts, and uneven coverage of the visual field. These can support response prediction while leaving spatial parameters poorly constrained. Validation should assess parameter recovery and stability as well as predictive accuracy.

### Phase, eccentricity, and coordinate semantics

In phase-encoded retinotopy, response phase indexes position within a periodic stimulus sweep. Conversion to polar angle or eccentricity depends on the stimulus sequence and response delay. This differs from local image phase, auditory envelope phase, beat phase, and neural oscillatory phase.

Each quantity needs a declared meaning, coordinate system, and unit. Circular variables need circular comparisons: 359° and 1° are close. Orientation also has a different periodicity from direction. Eccentricity is distance from the fixation reference, not distance from an arbitrary image center unless that equivalence is established by the presentation setup.

## Candidate intermediate representations

These are proposed additions or extensions, not assertions of current support.

| Domain | Scaffolding | Scientific use and interpretation |
|--------|-------------|-----------------------------------|
| Visual position and scale | Eccentricity × polar-angle bins; multiscale spatial pooling | Compare central/peripheral information and spatial integration under explicit fixation assumptions |
| Early vision | Orientation × spatial frequency × position; quadrature energy; color-opponent channels | Interpretable encoding and control for low-level visual similarity |
| Visual dynamics | Local direction/speed; motion-energy channels; expansion/contraction | Preserve motion structure that global summaries collapse |
| Auditory tuning | Cochleagrams; log-frequency channels; spectral and temporal modulation banks | Fit preferred frequency, bandwidth, and spectrotemporal tuning |
| Rhythm | Beat phase, metrical position, envelope modulation, onset predictability | Separate rhythmic position and expectation from tempo and onset strength |
| Speech and language | Phoneme/articulatory features, lexical competition, syntactic structure, contextual entropy | Compare acoustic, phonological, lexical, and compositional explanations |
| Temporal integration | Lagged features; causal integration windows; adaptation and decay kernels | Test how preceding context contributes to a representation |
| Narrative state | Character/location/goal tracks; event-state probabilities; uncertainty; recurrence | Describe persistence and change beyond adjacent embedding distance |
| Cross-modal relations | Lagged audiovisual correspondence; semantic and affective agreement | Distinguish temporal co-occurrence from meaningful correspondence |

### Auditory tuning: a close receptive-field analogue

Frequency-resolved and spectrotemporal representations retain information beyond MFCCs and scalar spectral summaries. They support models of preferred frequency and bandwidth, or joint tuning to spectral and temporal modulation. Relevant precedents include [population receptive-field estimates of human auditory cortex](https://pmc.ncbi.nlm.nih.gov/articles/PMC4262557/) and [encoding natural sounds at multiple spectral and temporal resolutions](https://pmc.ncbi.nlm.nih.gov/articles/PMC3879146/).

As in vision, a bank of hypothetical auditory filters is a stimulus model; a cortical tuning map requires fitting response data. Acoustic frequency is not interchangeable with perceived pitch, and frequency-channel coordinates should remain explicit.

### Temporal integration: the strongest match to narrative aims

Apply candidate integration models to each existing feature family: short and long preceding contexts, lag bases, exponential memory, or adaptation. This creates alternative representations whose RDMs and response predictions can be compared.

The motivation connects to [the hierarchy of temporal receptive windows described by Hasson et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC2556707/). However, a chosen smoothing window is a hypothesis, not a measured neural integration timescale. fMRI hemodynamics, extractor window length, stimulus autocorrelation, and neural integration must not be conflated.

For online cognitive interpretations, use causal context. Centered windows or models that see later frames may be valid for offline description but should be marked as using future information.

### Narrative and predictive models

Preserve the distinctions already developed in the [event-structure notes](event-structure-notes.md):

- **Local novelty:** a difference between nearby feature contexts.
- **Prediction error:** discrepancy between an actual forecast and an observation.
- **Posterior state change:** a change in inferred event or situation state.

Surprisal, entropy, and uncertainty also need distinct definitions. A feature difference is not automatically prediction error, and a predictable transition may still be a meaningful event boundary. Character, setting, intention, and causal tracks could provide interpretable content for latent-state models, provided extraction and human validation are established.

## Proposed representation contract

Extend a feature space into a representation with declared axes and valid operations. Retain simple CSV summaries where useful, while allowing structured arrays through referenced artifacts when needed. Storage format is an open design choice.

Suggested metadata:

- **Identity:** stimulus, sample/event identifier, model, and representation name.
- **Axes and coordinates:** time, position, frequency, orientation, channel, and their ordering.
- **Units and reference frames:** visual degrees versus pixels, Hz versus log-frequency, seconds, angular conventions, and image/screen/retinal origin.
- **Temporal support:** actual source interval, extraction window, preceding/following context, effective sampling, and causal versus offline computation.
- **Representation type:** scalar, vector, spatial field, probability distribution, event sequence, relational matrix, or fitted parameter map.
- **Provenance:** checkpoint/version, preprocessing, calibration, pooling, normalization, and whether values are measured, predicted, simulated, or fitted.
- **Missingness and uncertainty:** validity masks, unavailable channels, confidence, and fit uncertainty where applicable.
- **Valid operations:** supported aggregation, resampling, pooling, and distance functions.

A CLAP vector computed over a 10-second interval and an image feature at one frame do not have equal temporal support merely because their timestamps coincide. Resampling cannot recover temporal detail lost during extraction. Fine acoustic or visual dynamics should be represented at appropriate source resolution before downstream aggregation.

Distance choices also require representation-specific treatment. Angles are circular; distributions may warrant distribution distances; small scalar profiles need suitable scaling. Correlation on a two-dimensional profile is degenerate. Shared affective terminology does not imply calibrated, interchangeable scales across separately trained models.

For prediction experiments, fit normalization, dimensionality reduction, and other learned transforms on training data only. Split by runs or clips rather than randomly interleaving strongly autocorrelated timepoints; account for overlapping extraction windows across split boundaries.

## Architectural division of labor

- **Sibling extractors:** retain responsibility for raw stimuli and modality-specific feature computation, including spatial or spectrotemporal fields.
- **psytwill:** align representations, preserve coordinate/support metadata, apply declared transformations, and compute comparable relational outputs.
- **Optional response-fitting component:** consume representations plus neural/behavioral data, fit observer models, and export predictions, parameter estimates, uncertainty, and anatomical correspondence where appropriate.

The response-fitting component could be a module or a sibling package; that decision remains open. Keeping it optional would preserve psytwill's lightweight, modality-agnostic core. This proposal does not require moving raw-stimulus ingestion into psytwill.

For example, a hypothetical spatial-pooling bank can produce time × filter responses that psytwill compares as a new representational space. A fitted voxelwise model additionally produces parameters indexed by voxel or surface vertex. Those parameter maps should not be confused with stimulus-indexed feature tables.

## Suggested priorities and validation gates

1. **Common timing and temporal support.** Extend and audit the existing timeline/composition machinery for interval-aware alignment across extractors. Validate on known offsets and mixed sampling rates, preserving extraction-window information.
2. **Coordinate-aware sensory fields.** Start with interpretable visual filter banks and auditory frequency/modulation representations. Check coordinate transforms and expected responses to controlled synthetic stimuli.
3. **Spatial pooling and temporal integration banks.** Generate competing model representations and RDMs. Compare with simpler existing features and preserve all pooling/context assumptions.
4. **Optional neural fitting.** Begin with a tractable visual pRF or auditory tuning benchmark. Demonstrate synthetic parameter recovery, held-out prediction, stability, and comparison with independent mapping data where available.
5. **Latent narrative-state models.** Continue the existing event-model ladder, evaluating against human judgments and neural responses while separating novelty, forecasting, and state inference.

The guiding question is: **what information is present, where and over what interval is it integrated, and what evidence shows that the integration resembles an observer's?** This provides a coherent route from stimulus description to psychological and neuroscientific modeling without treating every new feature as an established neural mechanism.
