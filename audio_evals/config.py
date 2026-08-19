"""Configuration helpers shared by registry and benchmark YAML loaders."""

import os
import re
from typing import Any, Optional


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_CANONICAL_PREFIX = "VOXMATRIX_"
_LEGACY_PREFIX = "ULTRAEVAL_"


def environment_names(name: str) -> tuple[str, ...]:
    """Return canonical and legacy names for a VoxMatrix environment option."""
    if name.startswith(_CANONICAL_PREFIX):
        suffix = name[len(_CANONICAL_PREFIX) :]
        return name, f"{_LEGACY_PREFIX}{suffix}"
    if name.startswith(_LEGACY_PREFIX):
        suffix = name[len(_LEGACY_PREFIX) :]
        return f"{_CANONICAL_PREFIX}{suffix}", name
    return (name,)


def get_environment(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a setting with ``VOXMATRIX_*`` taking priority over its legacy alias.

    ``ULTRAEVAL_*`` remains supported so existing launch scripts and manifests do
    not need to change atomically with the VoxMatrix rename.
    """
    for candidate in environment_names(name):
        configured = os.environ.get(candidate)
        if configured:
            return configured
    return default


def expand_environment(value: Any) -> Any:
    """Recursively expand ``${VAR}`` and shell-like ``${VAR:-default}`` values."""
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replacement(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        configured = get_environment(name)
        if configured:
            return configured
        if default is not None:
            return default
        return match.group(0)

    return _ENV_PATTERN.sub(replacement, value)
