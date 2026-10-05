"""
core.config
-----------
Minimal configuration loading for the accent-converter project.

Phase 4B: implemented the minimum required (YAML dict loading).
Full config validation is a future milestone.
"""

from __future__ import annotations

import os


def load_config(path: str) -> dict:
    """Load a YAML configuration file and return it as a plain dict.

    Parameters
    ----------
    path:
        Absolute or relative path to a YAML configuration file.

    Returns
    -------
    dict
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file parses to something other than a dict (e.g. a list or None).
    """
    import yaml  # deferred: keeps this module importable without PyYAML

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Configuration file not found: {path!r}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration file must be a YAML mapping (dict), "
            f"got {type(data).__name__!r}: {path!r}"
        )
    return data
