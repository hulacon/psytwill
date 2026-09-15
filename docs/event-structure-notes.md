# Event structure from multimodal stimulus features

Status: research and design notes, not an approved implementation specification\
Discussion date: 2026-08-10

Publication note (2026-09-15): implementation-status statements below describe the August snapshot. Main now includes timeline and composition machinery; treat the alignment proposals as requirements to reassess against those components.

Related survey: [Neuroscientific models and receptive-field scaffolding](neuroscientific-models-and-receptive-fields.md) extends these ideas to spatial and auditory tuning, temporal integration, and optional neural-response fitting.

## Purpose

This note records ideas for extending psytwill from representational matrices and adjacent-transition curves toward computational proposals about event structure in movies and stories.

The intended claim should be modest: the system would propose event structures supported by selected features and modeling assumptions. It would not recover a single objectively true segmentation. Human event judgments depend on task, prior knowledge, granularity, and which dimensions of a situation a viewer attends to.

## Starting point in psytwill

psytwill already provides much of the substrate needed for this work:

- Separate representational spaces for semantic embeddings, visual embeddings, affective profiles, word-level aggregates, and acoustic or prosodic profiles.
- Within-space time-by-time similarity or distance matrices.
- Adjacent transition curves taken from the first off-diagonal of each self-similarity matrix.
- Cross-modal shared spaces such as CLIP text/image and CLAP text/audio.
- A metadata and sidecar architecture that can preserve feature provenance.

The most important missing prerequisite is time-aware alignment. Dialogue chunks, visual frames, and audio windows occur on different and sometimes irregular grids. They need to be mapped onto common intervals before evidence can be combined. Reasonable targets include fixed one-second bins for stimulus analysis and optional TR-locked bins for fMRI analyses.

## Core conceptual distinctions

A future system should represent at least four related but distinct objects.

1. **Boundary evidence**: graded evidence that the currently relevant structure has changed near a timepoint.
2. **Event instances**: contiguous periods inferred to belong together.
3. **Event types**: reusable latent situations or schemas that can recur in separated instances.
4. **Event hierarchy or tracks**: organization at multiple timescales and, eventually, partially overlapping processes such as setting, conversation, goals, and character arcs.

This separation matters because a loud onset, shot cut, or unusual sentence can produce a strong local change without creating a substantive event. Conversely, a predictable transition can be a meaningful boundary despite producing little surprise.

The distinction between instance and type is also essential. In an ABA sequence there are three event instances, even if the two A intervals instantiate the same event type.

## Lessons from the two inspiration papers

Yates, Sherman, and Yousif, *More than a moment: What does it mean to call something an event?* argues that boundary-like behavioral effects should not automatically be treated as evidence for substantive events. Useful event proposals should say something about the content between boundaries and should be capable of representing meaningful parts, hierarchy, longer timescales, and potentially overlapping events.

Shin and DuBrow, *Structuring Memory Through Inference-Based Event Segmentation* provides a more direct computational target. A viewer maintains a distribution over possible latent event types; segmentation becomes likely when this posterior distribution changes. Temporal persistence, reuse of earlier event types, uncertainty, and creation of new types are all part of the inference problem.

Together, these papers motivate treating observable discontinuities and prediction errors as cues to an inferred change, rather than definitions of events by themselves.

## Three different kinds of change evidence

These quantities should remain terminologically and computationally distinct.

### Local novelty

Local novelty asks whether feature context differs across a candidate boundary. For feature space `s`, time `t`, and window width `w`:

```text
novelty_s(t, w) = distance(
    aggregate(features_s before t),
    aggregate(features_s after t)
)
```

This can be calculated at several widths, such as 2, 4, 8, and 16 seconds. Windowed context is preferable to adjacent-row differences because it discounts momentary fluctuations that do not persist.

### Prediction error

Feature difference is not prediction error. Prediction error requires an actual forecast from recent history or an active event model:

```text
prediction_error_s(t) =
    distance(observed_s(t), predicted_s(t | history, active event))
```

Prediction error should be interpreted relative to estimated environmental volatility. A large deviation during a stable passage is more informative than the same deviation during a rapid, noisy montage.

### Posterior event-model change

An inference-based system maintains `p(event_type | observations)` and measures how much that distribution changes between timepoints. Jensen-Shannon divergence is one possible summary:

```text
posterior_shift(t) = JS(
    p(z_t | observations through t),
    p(z_(t-1) | observations through t-1)
)
```

This is closest to the Shin-DuBrow proposal. It can assign low boundary probability to a perceptual change that leaves the inferred event type stable, and high probability to a meaningful predictable transition.

## Combining many feature families

Early concatenation should not be the default. A 512-dimensional embedding would otherwise dominate a two-dimensional affect profile, and modalities with denser sampling could dominate sparse dialogue.

A better initial approach is late fusion:

1. Compute evidence independently within each meaningful feature family.
2. Normalize each evidence curve robustly within a clip or against a training distribution.
3. Record missingness and reliability explicitly.
4. Fuse standardized evidence using a median, a prespecified weighted combination, or weights learned from human annotations.
5. Preserve the per-space evidence so every fused boundary remains auditable.

Features likely to be especially useful include:

- Semantic, visual, and audio embeddings already supported by the constellation.
- Text emotion, vocal emotion, musical emotion, and low-level acoustics.
- Character and speaker continuity.
- Location or setting continuity.
- Action or activity representations.
- Shot cuts, fades, silence, and musical transitions as editorial cues.
- Dialogue topics and discourse acts.
- Eventually, inferred goals and causal relations.

Editorial cues should remain evidence rather than ground-truth boundaries; otherwise a system may simply rediscover cuts.

Small profile spaces need metric-specific care. Pearson correlation is degenerate for two-dimensional profiles and brittle for very small dimensionalities. Standardized Euclidean or shrinkage Mahalanobis-style distances are better candidates for valence/arousal and similarly small profiles.

## Computational frameworks worth comparing

These frameworks operate at different conceptual levels and are best treated as a benchmark ladder rather than mutually exclusive choices.

### 1. Penalized and kernel change-point detection

Change-point methods partition a sequence into contiguous intervals with relatively stable feature distributions. They are strong boundary-only baselines and make few cognitive claims.

Kernel change-point detection is particularly compatible with psytwill because an existing similarity matrix can serve as a kernel, and several kernels can be combined:

```text
K_fused = sum_s weight_s * K_s
```

PELT is a simpler and computationally efficient option when a segment cost can be specified over explicit features. Bayesian online change-point detection is useful when an online posterior over current run length and change probability is desirable.

These methods do not, by themselves, discover recurring event types.

### 2. Event HMM and Greedy State Boundary Search

The Baldassano event HMM models an event as a temporally stable, event-specific representational pattern and infers a globally coherent sequence of state transitions. Varying the number of states provides segmentations at different timescales.

Greedy State Boundary Search (GSBS) targets a similar stable-state definition while focusing directly on the number and placement of state boundaries.

Both are natural fits for aligned psytwill feature matrices and should be included as strong stable-state baselines. Their usual formulations do not naturally represent returning to an earlier event type, and fine and coarse solutions are not automatically a strictly nested hierarchy.

### 3. Sticky HMM or hidden semi-Markov model

A sticky HMM adds a self-transition bias that prevents rapid, redundant state switching. A nonparametric version can infer an effective number of reusable states. A hidden semi-Markov model adds an explicit duration distribution, which is attractive because event duration is substantive and extremely short events should require stronger evidence.

This family provides a practical way to infer persistent, recurring event types. Consecutive occupancy of a state is an event instance; recurrence of the same state later is another instance of the same type.

### 4. Structured Event Memory and a possible SEM-lite model

Structured Event Memory (SEM) combines nonparametric inference over event schemas with schema-specific predictive dynamics and memory consequences. It is the closest existing framework to the desired cognitive interpretation.

A tractable first approximation could:

1. Reduce each feature family separately.
2. Give every latent event type a simple autoregressive predictor.
3. Compare predictive likelihood under the current type, previously learned types, and a proposed new type.
4. Add a persistence or duration prior.
5. Infer boundaries from changes in posterior event-type probability.
6. Preserve event-instance and event-type identities separately.

This retains the central SEM claim without initially requiring neural event dynamics or full symbolic role binding.

### 5. Switching dynamical systems

HMM-like models usually define an event through a stable feature distribution. Switching linear dynamical systems instead define regimes by how features evolve.

That distinction may matter for events such as chases, conversations, and rising tension, whose identity is expressed through characteristic trajectories rather than a constant centroid. Recurrent switching linear dynamical systems can also make state transitions depend on the evolving continuous state or external inputs. Tree-structured variants offer a route to multiscale dynamics.

This family is promising but more data-hungry and sensitive to dimensionality, so it is better suited to a later phase.

### 6. Temporal-community structure

Temporal-community models define event structure through clusters of mutually predictive observations. They can recover communities even when individual transitions do not produce unusual surprise.

For psytwill, timepoints or short windows could become graph nodes, with edges determined by representational similarity, temporal transition structure, or both. A contiguous visit to a community would be an event instance; the community would be a reusable event type.

This approach makes fuller use of the RDM than an adjacent-transition curve and provides an appealing relational complement to generative models such as SEM.

### 7. Event-indexing and situation-model dimensions

The Event-Indexing Model is not a complete inference algorithm, but it supplies a valuable feature ontology. Narrative situation models can be represented along at least:

- Time.
- Space.
- Protagonists or entities.
- Causality.
- Intentions and goals.

Rather than compressing these into one narrative embedding, a future system could maintain separate continuity tracks and learn which dimensions matter for different event types. This would also make a boundary proposal more explainable.

### 8. Factorial and hierarchical state models

A standard HMM permits one latent event state at a time. A factorial model allows several latent chains to evolve concurrently, for example:

```text
setting:       office ---------------- restaurant -------
conversation:  work talk -- pause ----- personal talk ---
character arc: anxiety ------------------------- relief --
music regime:  silence ---- tension ---- resolution -----
```

This is the clearest response to partially overlapping events and asynchronous changes across situation dimensions. Hierarchical state models similarly provide a principled route to fine and coarse structure.

These models should be deferred until the component feature tracks are reliable. Otherwise latent factors may be statistically interchangeable and difficult to interpret.

### 9. Supervised temporal segmentation

Temporal convolutional or transformer models can learn long-range segmentation and smoothing from labeled sequences. They become relevant if a sufficiently large, representative set of human event annotations is available.

They should not be the initial framework. Action-segmentation labels from instructional or activity datasets are not necessarily the same construct as cognitive event boundaries in narrative film, and an opaque supervised system would provide a weak theoretical baseline.

## Recommended modeling ladder

The proposed sequence of work is:

### Phase A: time alignment and interpretable boundary evidence

- Map every input to a shared interval representation.
- Compute per-space multiscale novelty curves.
- Normalize evidence within feature family.
- Fuse evidence late using an interpretable rule.
- Apply constrained peak selection and minimum-duration rules.
- Preserve all component evidence and uncertainty.

This is the simplest useful multimodal event-proposal baseline.

### Phase B: global offline partitions

- Run kernel change-point detection on individual and fused kernels.
- Compare PELT or another explicit-feature change-point method.
- Add Event HMM and GSBS stable-state segmentations.
- Compare fine and coarse solutions without initially claiming strict hierarchy.

This phase asks how far discontinuity and stable-state models can go.

### Phase C: recurring types and durations

- Fit a finite sticky HMM or HSMM with a deliberately generous state count.
- Evaluate recurrence and event-instance/type separation.
- Add temporal-community clustering as a relational alternative.
- Consider a nonparametric state prior only if uncertainty about state count is empirically important.

### Phase D: predictive dynamics and structured events

- Implement and evaluate a SEM-lite model.
- Consider switching dynamical systems if events appear to be defined more by trajectories than centroids.
- Add explicit situation-model dimensions, causal structure, or symbolic roles only with clear extraction and validation plans.

### Phase E: hierarchy and overlap

- Enforce nested fine/coarse partitions if the scientific question requires them.
- Explore factorial or hierarchical state models for asynchronously overlapping event tracks.
- Consider supervised temporal models only after acquiring enough appropriate human annotations.

## Proposed output contract

Segmentation should add artifacts rather than replace the existing RDM outputs.

### `boundary_scores.csv`

Suggested fields:

```text
time
scale
space
evidence_kind
raw_evidence
standardized_evidence
fused_evidence
boundary_probability
uncertainty
```

### `events.csv`

Suggested fields:

```text
event_instance
event_type
level
onset
offset
duration
start_confidence
end_confidence
model
model_run
```

### `event_features.csv`

One summarized feature representation per event instance and feature space, including within-event variability and missingness.

### Event relation matrices

Event-by-event matrices could describe:

- Representational similarity.
- Recurrence or common event-type probability.
- Temporal transition probability.
- Eventually, proposed causal or narrative dependence.

All outputs should retain enough provenance to reconstruct the aligned inputs, model configuration, feature weights, duration assumptions, and boundary thresholds.

## Validation plan

Human segmentation should be the principal validation target.

- Boundary precision and recall with a prespecified temporal tolerance.
- Area under the precision-recall curve for graded human boundary agreement or strength.
- Fine and coarse annotations evaluated separately.
- Leave-one-film-out validation for any learned fusion weights or hyperparameters.
- Modality and feature-family ablations.
- Comparisons with semantic-only novelty, shot cuts, uniform-duration segmentation, and duration-matched random boundaries.
- Robustness to sampling rate, temporal window width, and minimum-duration settings.
- Calibration of boundary probability and uncertainty where probabilistic models are used.

Internal cohesion and separation from neighboring events are useful diagnostics, but they should not replace human validation because optimizing those quantities can favor mathematically tidy partitions that people do not perceive as events.

Human causal event-pair ratings are a separate validation target for proposed event-relation matrices, not a substitute for validating boundaries.

## Important failure modes

- Calling any large feature difference a prediction error.
- Calling any detected boundary an event.
- Allowing high-dimensional embeddings or densely sampled modalities to dominate fusion.
- Treating shot cuts as ground truth rather than editorial cues.
- Using adjacent-row differences without testing whether change persists.
- Forcing every representation into a single sequential event track.
- Confusing recurrence of an event type with continuation of one event instance.
- Selecting the number of events on the same stimuli used for final evaluation.
- Claiming hierarchy from several unrelated values of `K` without checking nesting.
- Using causal or goal features whose extraction quality cannot be independently assessed.

## Immediate design recommendation

The strongest near-term combination is:

1. A multiscale, late-fusion novelty baseline.
2. Kernel change-point detection operating on psytwill RDMs.
3. Event HMM and GSBS stable-state baselines.
4. A finite sticky HSMM for reusable event types and explicit durations.

These four approaches form an interpretable progression from local evidence to global partitions and then to recurring latent states. Their shared output contract can later support SEM-lite, switching dynamics, temporal communities, and factorial structure without changing the upstream extractors.

## Open questions for the next checkpoint

- What is the primary timebase: one-second stimulus bins, shot intervals, sentence/dialogue spans, fMRI TRs, or several exported versions?
- Should initial work target within-film boundaries only, or also learn event types shared across films?
- Are human annotations available at both fine and coarse levels, with boundary-strength judgments?
- Should coarse events be required to contain fine events strictly, or may scales overlap without nesting?
- Which situation-model dimensions can be extracted reliably enough to treat as distinct tracks?
- Is the first scientific target human boundary agreement, neural event-state agreement, memory consequences, or all three in a preregistered hierarchy?
- How should missing modalities, especially dialogue-free periods, affect fusion weights and uncertainty?

## Selected references

- Adams, R. P., & MacKay, D. J. C. (2007). [Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742).
- Arlot, S., Celisse, A., & Harchaoui, Z. (2019). [A Kernel Multiple Change-point Algorithm via Model Selection](https://www.jmlr.org/papers/volume20/16-155/16-155.pdf).
- Baldassano, C., Chen, J., Zadbood, A., Pillow, J. W., Hasson, U., & Norman, K. A. (2017). [Discovering Event Structure in Continuous Narrative Perception and Memory](https://doi.org/10.1016/j.neuron.2017.06.041).
- Fox, E. B., Sudderth, E. B., Jordan, M. I., & Willsky, A. S. (2011). [A Sticky HDP-HMM with Application to Speaker Diarization](https://doi.org/10.1214/10-AOAS395).
- Franklin, N. T., Norman, K. A., Ranganath, C., Zacks, J. M., & Gershman, S. J. (2020). [Structured Event Memory: A Neuro-symbolic Model of Event Cognition](https://doi.org/10.1037/rev0000177).
- Geerligs, L., van Gerven, M., & Güçlü, U. (2021). [Detecting Neural State Transitions Underlying Event Segmentation](https://doi.org/10.1016/j.neuroimage.2021.118085).
- Ghahramani, Z., & Jordan, M. I. (1997). [Factorial Hidden Markov Models](https://doi.org/10.1023/A:1007425814087).
- Killick, R., Fearnhead, P., & Eckley, I. A. (2012). [Optimal Detection of Changepoints With a Linear Computational Cost](https://doi.org/10.1080/01621459.2012.737745).
- Linderman, S. W., Johnson, M. J., Miller, A. C., Adams, R. P., Blei, D. M., & Paninski, L. (2017). [Bayesian Learning and Inference in Recurrent Switching Linear Dynamical Systems](https://proceedings.mlr.press/v54/linderman17a.html).
- Schapiro, A. C., Rogers, T. T., Cordova, N. I., Turk-Browne, N. B., & Botvinick, M. M. (2013). [Neural Representations of Events Arise from Temporal Community Structure](https://doi.org/10.1038/nn.3331).
- Zwaan, R. A., Langston, M. C., & Graesser, A. C. (1995). [The Construction of Situation Models in Narrative Comprehension: An Event-Indexing Model](https://doi.org/10.1111/j.1467-9280.1995.tb00513.x).
