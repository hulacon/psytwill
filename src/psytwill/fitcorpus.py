"""External fit-corpus identity — the ``ext-<corpus>-<native>`` namespace.

psytwill-space block weights are fit on non-mmmdata corpora and frozen
(fit-on-external-then-freeze decision; see the psytwill-space charter in
mmmdata-agents ``docs/workbench/psytwill-space/``). Feature tables extracted
over those corpora carry a namespaced ``stimulus_id`` so they can never
collide with mmmdata stimulus ids: registry ids (``shared####_nsd#####``,
``twp####_...``, movie slugs) never start with ``ext-``.

The corpus registry below is the single source of truth for the namespace.
Parsing is longest-match over registered keys, so corpus keys may themselves
contain hyphens (``ext-peoples-speech-chunk0042`` splits at the registered
key, not the first hyphen). Two invariants keep that unambiguous, enforced
by ``_validate_registry`` at import time:

- keys are lowercase ``[a-z0-9-]``, and
- no key is another key plus a ``-``-joined suffix.

Acquisition tiers (licensing decision, 2026-08-25):

- ``local``  — already on disk beside mmmdata; nothing to fetch.
- ``a``      — CC/public-domain: usable, shippable, scripted fetch.
- ``b``      — nameable-and-obtainable commercial media (CNeuroMod stimuli):
  fit corpus is named, never redistributed; acquisition is a manual,
  documented procedure.
- ``validation`` — external-brain validation targets, not fit stimuli.

Typed declarations (0.3.0 registry, light form): each corpus declares its
roles per product and split, its licence, its media route and each member's
training-set overlap. They are bookkeeping, not permission: no fit, relation,
bench read or membership pass refuses on them. A release records the table,
and ``release verify`` warns on conflicts (:mod:`psytwill.registry`). What a
consumer has done with a corpus (overlap with its own stimuli, splits already
read) is the consumer's usage file, not code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from psytwill.exceptions import CorpusError

EXT_PREFIX = "ext-"

_KEY_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_NATIVE_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")

# --------------------------------------------------------------------------
# typed declarations (bookkeeping, not permission)
# --------------------------------------------------------------------------

#: The products a role names: the blocks and the relations of the space.
PRODUCTS = ("V", "A", "L", "D", "VL", "AL", "ALs", "VA")
#: The kinds of use a corpus split can be declared for.
#:
#: - ``fit``: rows of a block fit, or the fit rows of a relation;
#: - ``gate``: a relation's held-out gate split (sets its k);
#: - ``register``: a coupling-only register on a frozen relation (``fit: false``);
#: - ``dev``: a development (D-tier) read, re-readable, where views are tuned;
#: - ``bar``: a release bar (B tier), read once, whole;
#: - ``report``: a read whose number is reported, never a bar;
#: - ``screen``: a candidate-screen subset;
#: - ``brain``: external brain data for development reads;
#: - ``member-training``: the corpus trained a member (a probe) or set a
#:   detector's threshold. It names that member or detector as ``target``
#:   and no product.
ROLE_KINDS = ("fit", "gate", "register", "dev", "bar", "report", "screen", "brain", "member-training")
#: Overlap between a corpus split and a member's training data.
OVERLAP_VALUES = ("none", "present", "undisclosed", "unchecked")
#: How a corpus's media reaches extraction.
#:
#: ``local`` (read in place), ``fetch`` (a scripted public download),
#: ``fetch-gated`` (scripted, behind an account or token), ``manual`` (a
#: documented manual procedure).
MEDIA_ROUTES = ("local", "fetch", "fetch-gated", "manual")
UNRECORDED = "unrecorded"


@dataclass(frozen=True)
class Role:
    """One declared use of a corpus split."""

    kind: str
    product: str | None = None
    split: str = "all"
    target: str | None = None  # member-training only: the member or detector
    note: str = ""


@dataclass(frozen=True)
class MemberOverlap:
    """A member's training-data overlap with a corpus.

    ``member`` ``"*"`` covers every member the corpus does not name. A member
    named nowhere is ``unchecked``. ``present`` names the split it covers.
    """

    member: str
    value: str
    split: str | None = None
    note: str = ""


@dataclass(frozen=True)
class CorpusSpec:
    """One external corpus in the fit-corpus ledger.

    The fields here are facts about the corpus. What a consumer has done with
    it (its overlap with the consumer's own stimuli, which splits have been
    read) lives in the consumer's usage file (:mod:`psytwill.registry`).
    """

    key: str
    tier: str  # local | a | b | validation
    serves: str  # which block(s) / role, free text for humans
    id_width: int | None = None  # zero-pad width for integer native ids
    notes: str = field(default="")
    licence: str = UNRECORDED
    media: str = UNRECORDED  # one of MEDIA_ROUTES
    roles: tuple[Role, ...] = ()
    training_overlap: tuple[MemberOverlap, ...] = ()
    #: Non-empty = no longer used, and why. The key stays registered so ids
    #: already minted under it still parse.
    dropped: str = ""

    def overlap(self, member: str) -> MemberOverlap:
        """The declared overlap for ``member`` (its own entry, else ``*``, else unchecked)."""
        by = {o.member: o for o in self.training_overlap}
        return by.get(member) or by.get("*") or MemberOverlap(member, "unchecked")


def _r(kind: str, product: str | None = None, split: str = "all", *, target: str | None = None,
       note: str = "") -> Role:
    return Role(kind, product, split, target, note)


def _o(member: str, value: str, split: str | None = None, note: str = "") -> MemberOverlap:
    return MemberOverlap(member, value, split, note)


_COCO_TRAINED = tuple(_o(m, "undisclosed", note="COCO images / captions are likely in web-scale "
                         "image-text training; not disclosed")
                      for m in ("clip", "clip_text", "ebind", "ebind_text"))
_CLAP_SPEECH_MUSIC = tuple(_o(m, "undisclosed", note="the checkpoint's speech / music training mix is "
                              "not disclosed")
                           for m in ("clap", "clap_text"))


# Keys mirror the corpus ledger in mmmdata-agents
# docs/workbench/psytwill-space/out/fit-corpus-pipeline.md §1. Roles are the
# current development cycle's declarations; a release records the table it
# was written under.
CORPORA: dict[str, CorpusSpec] = {
    spec.key: spec
    for spec in (
        CorpusSpec(
            "nsd",
            tier="local",
            serves="V private; V<->L (COCO captions ride these ids)",
            id_width=6,
            notes="73k-image NSD hdf5 read in place; ids are 1-based nsdId",
            licence="NSD terms of use (data use agreement); never redistributed",
            media="local",
            roles=(_r("fit", "V"), _r("fit", "L", note="COCO human captions"), _r("fit", "VL"),
                   _r("screen", "V"), _r("brain", "V", note="reported, not barred")),
            training_overlap=_COCO_TRAINED,
        ),
        CorpusSpec(
            "coco",
            tier="a",
            serves="V<->L gate and bar: COCO images outside NSD, drawn by a pinned seed",
            notes="two disjoint draws from COCO images outside NSD (>= 1 non-crowd annotation; "
                  "stratified on supercategory flags); seed and ids pinned before extraction",
            licence="annotations CC BY 4.0; images per-image Flickr licences",
            media="fetch",
            roles=(_r("gate", "VL", "draw-1"), _r("bar", "VL", "draw-2")),
            training_overlap=_COCO_TRAINED,
        ),
        CorpusSpec(
            "twp-unpresented",
            tier="local",
            serves="A private; A<->L (domain-matched voice/word audio)",
            notes="the 3,000 twp recordings frozen out of twp1000 (every "
                  "word in its three never-presented voices), packed per voice",
            licence="lab-owned recordings",
            media="local",
            roles=(_r("fit", "A"), _r("screen", "A")),
        ),
        CorpusSpec(
            "librispeech",
            tier="a",
            serves="A private; A<->L (read speech + transcripts)",
            licence="CC BY 4.0",
            media="fetch",
            roles=(_r("fit", "A", "train-clean-100"), _r("fit", "L", "train-clean-100"),
                   _r("fit", "ALs", "train-clean-100"), _r("screen", "A", "train-clean-100"),
                   _r("gate", "ALs", "dev-other"), _r("bar", "ALs", "dev-clean"),
                   _r("dev", "ALs", "test-clean"), _r("dev", "ALs", "test-other")),
            training_overlap=_CLAP_SPEECH_MUSIC,
        ),
        CorpusSpec(
            "gigaspeech",
            tier="a",
            serves="A private; A<->L (conversational speech)",
            licence="GigaSpeech research licence",
            media="fetch-gated",
            roles=(_r("fit", "A"), _r("fit", "L"), _r("register", "ALs"), _r("screen", "A")),
            training_overlap=_CLAP_SPEECH_MUSIC,
        ),
        CorpusSpec(
            "peoples-speech",
            tier="a",
            serves="A private; A<->L (conversational speech)",
            dropped="never fetched; no named gap it fills (corpora enter for a gap, read or gate)",
        ),
        CorpusSpec(
            "narratives",
            tier="a",
            serves="L (spoken-story transcripts)",
            licence="CC0 (OpenNeuro ds002345)",
            media="fetch",
            roles=(_r("fit", "L"),),
        ),
        CorpusSpec("jamendo", tier="a", serves="A private (music arm)",
                   dropped="inclusion never decided; never fetched"),
        CorpusSpec(
            "fma",
            tier="a",
            serves="A private (music arm: genre-balanced clips)",
            notes="fma_small, a seeded per-genre draw packed into units; "
                  "ratified 2026-08-25 (decision 4)",
            licence="Creative Commons, per track",
            media="fetch",
            roles=(_r("fit", "A"),),
            training_overlap=_CLAP_SPEECH_MUSIC,
        ),
        CorpusSpec(
            "musopen",
            tier="a",
            serves="A private (music arm: public-domain orchestral score)",
            notes="the Musopen DVD compilation (PD); the orchestral-style "
                  "add to decision 4",
            licence="public domain",
            media="fetch",
            roles=(_r("fit", "A"),),
            training_overlap=_CLAP_SPEECH_MUSIC,
        ),
        CorpusSpec(
            "friends",
            tier="b",
            serves="A, L and D fit rows; V<->A coupling register; external brains (dev episodes)",
            licence="commercial media; never redistributed",
            media="manual",
            roles=(_r("fit", "A"), _r("fit", "L"), _r("fit", "D"), _r("register", "VA"),
                   _r("screen", "V"), _r("brain", "V", "dev-episodes"), _r("brain", "A", "dev-episodes")),
        ),
        CorpusSpec(
            "movie10",
            tier="b",
            serves="A, L and D fit rows; V<->A coupling register; external brains (AV)",
            licence="commercial media; never redistributed",
            media="manual",
            roles=(_r("fit", "A"), _r("fit", "L"), _r("fit", "D"), _r("register", "VA"),
                   _r("brain", "V"), _r("brain", "A")),
        ),
        CorpusSpec(
            "tvc",
            tier="a",
            serves="L fit rows: human TV-moment captions (a film-annotation register)",
            notes="TVC train captions, excluding episodes of a fit corpus's shows that are "
                  "outside that corpus's own fit episodes",
            licence="CC BY-NC-SA",
            media="fetch",
            roles=(_r("fit", "L", "train"),),
        ),
        CorpusSpec(
            "blender",
            tier="a",
            serves="D rows: the animated wedge (CC-BY open movies)",
            notes="Big Buck Bunny (2008) onward; Tears of Steel excluded as "
                  "live action; ratified 2026-09-07. 2026-11-09: only the "
                  "films Blender serves as free official files (Big Buck "
                  "Bunny, Sintel, Caminandes 2) plus Elephants Dream (2006, "
                  "admitted for its dialogue); the rest need a Blender Studio "
                  "login (fetch.py blender)",
            licence="CC BY",
            media="fetch",
            roles=(_r("fit", "D"),
                   _r("member-training", target="yunet", note="animation calibration of the face detector")),
        ),
        CorpusSpec(
            "ava",
            tier="b",
            serves="D rows (motion) and V<->A on live-action film: many titles, one 15-min "
                   "segment each",
            notes="AVA (Gu et al. 2018) minutes 15:00-30:00 of each film, from the CVDF "
                  "mirror; third-party YouTube films, research use, never redistributed. "
                  "Native id = 10 hex of sha1(YouTube id): YouTube ids are case-sensitive "
                  "and may start with '-'. inputs/ids.csv maps native <-> YouTube id and "
                  "declares each file's clock: the video keeps its original timestamps "
                  "(first frame at the keyframe before 895 s), the audio starts at 900 s, "
                  "so a frame at video time t sits at t + video_start_s - audio_start_s "
                  "on the audio clock",
            licence="labels CC BY 4.0; films third-party, research use, never redistributed",
            media="fetch",
            roles=(_r("fit", "D"), _r("report", "VA", note="film V<->A read"),
                   _r("member-training", target="diarize", note="speech-gate calibration (AVA-Speech)"),
                   _r("member-training", target="yunet", note="face-gate calibration (AVA-ActiveSpeaker)")),
            training_overlap=(_o("ebind_audio", "none", note="no AVA video is in AudioSet"),),
        ),
        CorpusSpec(
            "clotho",
            tier="a",
            serves="A<->L (audio-only captions: annotators heard the clip, saw nothing)",
            notes="Clotho v2.1, Zenodo 4783391; captions non-commercial, audio "
                  "per-clip Freesound CC; research use, never redistributed. "
                  "Native id = '<dev|eval>-' + 10 hex of sha1(file_name): file "
                  "names collide once slugged. One clip ships in both splits "
                  "(drop it from eval). Ratified 2026-10-09 (l-hub)",
            licence="captions non-commercial; audio Creative Commons per Freesound clip; research use",
            media="fetch",
            roles=(_r("fit", "AL", "development"), _r("gate", "AL", "validation"),
                   _r("dev", "AL", "evaluation")),
            training_overlap=(_o("clap", "present", "development"), _o("clap_text", "present", "development")),
        ),
        CorpusSpec(
            "avcaps",
            tier="a",
            serves="V<->A fit (train); A<->L and V<->L coupling registers over every clip",
            notes="AVCaps, Zenodo 14536325 (VidOR clips); licence CC-BY 4.0 on "
                  "Zenodo vs CC-BY-NC on HF, treated as research-only, never "
                  "redistributed. Native id = the VidOR video id. Ratified "
                  "2026-10-09 (l-hub)",
            licence="CC BY 4.0 (Zenodo) / CC BY-NC (HF); treated as research-only",
            media="fetch",
            roles=(_r("fit", "VA", "train"), _r("register", "AL"), _r("register", "VL"),
                   _r("dev", "VA", "val+test")),
            training_overlap=(_o("ebind", "none", note="VidOR comes from YFCC100M, not AudioSet"),
                              _o("ebind_audio", "none", note="VidOR comes from YFCC100M, not AudioSet"),
                              *_CLAP_SPEECH_MUSIC),
        ),
        CorpusSpec(
            "vidor",
            tier="a",
            serves="V<->A gate: VidOR clips no AVCaps split uses, drawn to match AVCaps",
            notes="drawn from the AVCaps source pool outside AVCaps's clips, matched on duration, "
                  "audio presence, level and speech share; seed and ids pinned before extraction. "
                  "Native id = the VidOR video id",
            media="fetch",
            roles=(_r("gate", "VA", "draw"),),
            training_overlap=(_o("ebind", "none", note="YFCC100M, not AudioSet"),
                              _o("ebind_audio", "none", note="YFCC100M, not AudioSet"),
                              *_CLAP_SPEECH_MUSIC),
        ),
        CorpusSpec(
            "song-describer",
            tier="a",
            serves="A<->L coupling register: music captions",
            licence="CC BY-SA",
            media="fetch",
            roles=(_r("register", "AL"),),
        ),
        CorpusSpec(
            "paraspeechcaps",
            tier="a",
            serves="A<->L coupling register: tag-rendered captions on read / acted speech",
            notes="the Expresso and EARS subsets (EARS dev + holdout)",
            licence="CC BY-NC-SA",
            media="fetch",
            roles=(_r("register", "AL", "expresso"), _r("register", "AL", "ears-dev"),
                   _r("register", "AL", "ears-holdout")),
        ),
        CorpusSpec(
            "things",
            tier="validation",
            serves="V view development set (train); CNeuroMod-THINGS external brains",
            notes="CNeuroMod-THINGS images + CC0 betas; Binder attribute ratings ride its concepts",
            licence="data CC BY 4.0; images research / education only",
            media="manual",
            roles=(_r("dev", "V", "train"), _r("dev", "V", "validation"), _r("brain", "V"),
                   _r("member-training", target="experiential_v", note="probe trained on Binder-rated images")),
        ),
        CorpusSpec(
            "peterson",
            tier="b",
            serves="V metric bar: Peterson et al. 2018 human similarity, six image sets",
            notes="the mean of the six per-set Spearman reads; concepts overlap THINGS",
            licence="images third-party; use, never redistribute",
            media="fetch",
            roles=(_r("bar", "V"),),
            training_overlap=(_o("*", "undisclosed", note="web images; member training sets not disclosed"),),
        ),
        CorpusSpec(
            "meadows",
            tier="b",
            serves="V metric development reads: NSD Meadows arrangement RDMs and ratings",
            licence="NSD terms of use; never redistributed",
            media="manual",
            roles=(_r("dev", "V"),),
        ),
        CorpusSpec(
            "giordano",
            tier="a",
            serves="A metric development set: Giordano et al. 2010 environmental-sound dissimilarity",
            notes="two groups of 40 sounds with no between-group pairs; the held-out check of a "
                  "learned view splits by group, both directions",
            licence="CC0 (Dryad)",
            media="manual",
            roles=(_r("dev", "A", "group-1"), _r("dev", "A", "group-2")),
        ),
        CorpusSpec(
            "envisounds",
            tier="b",
            serves="A metric read: within-category environmental-sound similarity",
            notes="reported only until its ratings licence is granted; a grant makes it a bar",
            licence="ratings licence not yet granted",
            media="manual",
            roles=(_r("report", "A"),),
        ),
        CorpusSpec(
            "semeval-2022",
            tier="a",
            serves="L metric development set: SemEval-2022 Task 8 (en-en) news similarity",
            media="fetch",
            roles=(_r("dev", "L", "train"), _r("dev", "L", "eval")),
        ),
        CorpusSpec(
            "semeval-2026",
            tier="a",
            serves="L metric: SemEval-2026 Task 4 narrative similarity (dev = development, test = bar)",
            media="fetch",
            roles=(_r("dev", "L", "dev"), _r("bar", "L", "test")),
        ),
        CorpusSpec(
            "veatic",
            tier="b",
            serves="trained the context_affect probe (dimensional affect on video)",
            licence="research use; copyright stays with the owners; never redistributed",
            media="manual",
            roles=(_r("member-training", target="context_affect"),),
        ),
    )
}


def _validate_registry(corpora: dict[str, CorpusSpec]) -> None:
    keys = sorted(corpora)
    for key in keys:
        if not _KEY_RE.match(key):
            raise CorpusError(
                f"corpus key {key!r} must be lowercase [a-z0-9-] and "
                "start with a letter"
            )
    for short in keys:
        for long in keys:
            if long != short and long.startswith(short + "-"):
                raise CorpusError(
                    f"corpus keys {short!r} and {long!r} are parse-ambiguous: "
                    "no key may be another key plus a '-'-joined suffix"
                )
    for key in keys:
        spec = corpora[key]
        bad = []
        if spec.media not in (*MEDIA_ROUTES, UNRECORDED):
            bad.append(f"media {spec.media!r} is not one of {MEDIA_ROUTES}")
        seen = set()
        for r in spec.roles:
            if r.kind not in ROLE_KINDS:
                bad.append(f"role kind {r.kind!r} is not one of {ROLE_KINDS}")
            elif r.kind == "member-training":
                if not r.target or r.product is not None:
                    bad.append("a member-training role names its target (member or detector) and no product")
            elif r.product not in PRODUCTS or r.target is not None:
                bad.append(f"role {r.kind} names product {r.product!r}: needs one of {PRODUCTS} and no target")
            ident = (r.kind, r.product, r.split, r.target)
            if ident in seen:
                bad.append(f"role {ident} declared twice")
            seen.add(ident)
        members = [o.member for o in spec.training_overlap]
        if len(set(members)) != len(members):
            bad.append("a member's training overlap is declared twice")
        for o in spec.training_overlap:
            if o.value not in OVERLAP_VALUES:
                bad.append(f"overlap {o.member}={o.value!r} is not one of {OVERLAP_VALUES}")
            if (o.value == "present") != (o.split is not None):
                bad.append(f"overlap {o.member}: `present` names the split it covers, and only `present` does")
        if spec.dropped and spec.roles:
            bad.append("a dropped corpus declares no roles")
        if bad:
            raise CorpusError(f"corpus {key!r}: " + "; ".join(bad))


_validate_registry(CORPORA)


def ext_id(corpus: str, native_id: str | int) -> str:
    """Build the namespaced stimulus_id ``ext-<corpus>-<native>``.

    Integer native ids are zero-padded to the corpus's ``id_width``
    (``ext_id("nsd", 21384)`` -> ``"ext-nsd-021384"``).
    """
    spec = CORPORA.get(corpus)
    if spec is None:
        raise CorpusError(
            f"unknown fit corpus {corpus!r}; registered: {sorted(CORPORA)}. "
            "New corpora enter via the ledger "
            "(fit-corpus-pipeline.md) and this registry together."
        )
    if isinstance(native_id, int):
        if native_id < 0:
            raise CorpusError(f"native id must be non-negative, got {native_id}")
        native = f"{native_id:0{spec.id_width or 0}d}"
    else:
        native = native_id
    if not _NATIVE_RE.match(native):
        raise CorpusError(
            f"native id {native!r} must be lowercase [a-z0-9-] and start "
            "alphanumeric; normalize upstream in the corpus adapter"
        )
    return f"{EXT_PREFIX}{corpus}-{native}"


def parse_ext_id(stimulus_id: str) -> tuple[str, str]:
    """Split ``ext-<corpus>-<native>`` -> (corpus, native), longest-match."""
    if not stimulus_id.startswith(EXT_PREFIX):
        raise CorpusError(
            f"{stimulus_id!r} is not an external stimulus id "
            f"(no {EXT_PREFIX!r} prefix)"
        )
    rest = stimulus_id[len(EXT_PREFIX):]
    matches = [
        key for key in CORPORA
        if rest == key or rest.startswith(key + "-")
    ]
    if not matches:
        raise CorpusError(
            f"{stimulus_id!r} names no registered corpus; "
            f"registered: {sorted(CORPORA)}"
        )
    corpus = max(matches, key=len)
    native = rest[len(corpus) + 1:]
    if not native:
        raise CorpusError(f"{stimulus_id!r} has an empty native id")
    return corpus, native


def is_external(stimulus_id: str) -> bool:
    """True if the id lives in the fit-corpus namespace."""
    return stimulus_id.startswith(EXT_PREFIX)
