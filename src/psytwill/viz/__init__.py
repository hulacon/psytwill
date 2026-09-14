"""Static feature viewers over the long-form features table.

``psytwill viz`` builds self-contained HTML bundles (no server, no network
— they work over ``file://``) that render features from every modality of
one stimulus set on shared axes, with the stimuli themselves (frames,
audio, transcript) resolved from the features tree.

P0 (this subpackage's first surface) is the movies timeline viewer:
``psytwill viz movies``. See ``movies.py`` for the payload contract and
``build.py`` for the bundle layout.
"""
