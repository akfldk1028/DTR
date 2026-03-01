"""param_utils.py — Shared YAML parameter loading utilities

Provides common functions for loading and extracting values from
the project's schema-aware YAML parameter files.

All YAML parameter files use the following schema convention:
    key:
      value: <actual value>
      unit: "<SI unit>"
      range: [min, max]
      description: "<Korean description>"

Usage:
    from scripts.param_utils import load_yaml, get_value

    config = load_yaml(Path("params/control.yaml"))
    stiffness = get_value(config["drive"]["stiffness"])  # → 40.0

Phase: shared utility (all phases)
"""

from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    """Load a YAML parameter file and return its contents.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML contents as a dict.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_value(param_entry):
    """Extract the 'value' field from a schema-aware YAML parameter entry.

    If the entry is a dict with a 'value' key, return that value.
    Otherwise return the entry as-is (for plain values or nested dicts).

    Args:
        param_entry: A YAML parameter entry (dict with 'value' key, or raw value).

    Returns:
        The extracted value.
    """
    if isinstance(param_entry, dict) and "value" in param_entry:
        return param_entry["value"]
    return param_entry
