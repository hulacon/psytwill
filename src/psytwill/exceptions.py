"""Structured errors, mirroring the siblings' exception style."""


class PsytwillError(Exception):
    """Base class for all psytwill errors."""


class InputError(PsytwillError):
    """A scores CSV could not be read or is not usable."""


class EmptyInputError(InputError):
    """A scores CSV is well-formed but has no rows.

    Distinct from a malformed input because it is often legitimate: a movie
    with no speech has an empty transcript, a 0.54 s spoken word has no
    detectable beat. Single-input verbs still refuse it; `features`, which
    aggregates many inputs, skips and records it.
    """


class SpaceError(PsytwillError):
    """A requested or detected feature space is invalid."""


class MetricError(PsytwillError):
    """An unknown or inapplicable metric was requested."""


class CorpusError(PsytwillError):
    """An external fit-corpus id or registry entry is invalid."""
