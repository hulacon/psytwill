"""Structured errors, mirroring the siblings' exception style."""


class PsytwillError(Exception):
    """Base class for all psytwill errors."""


class InputError(PsytwillError):
    """A scores CSV could not be read or is not usable."""


class SpaceError(PsytwillError):
    """A requested or detected feature space is invalid."""


class MetricError(PsytwillError):
    """An unknown or inapplicable metric was requested."""
