"""Structured errors, mirroring the siblings' exception style."""


class PsyquiltError(Exception):
    """Base class for all psyquilt errors."""


class InputError(PsyquiltError):
    """A scores CSV could not be read or is not usable."""


class SpaceError(PsyquiltError):
    """A requested or detected feature space is invalid."""


class MetricError(PsyquiltError):
    """An unknown or inapplicable metric was requested."""
