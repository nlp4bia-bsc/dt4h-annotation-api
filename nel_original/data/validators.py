"""Validation helpers for tabular inputs."""

from __future__ import annotations


def require_columns(columns: list[str], required: list[str], name: str) -> None:
    """Raise a clear error when required columns are missing."""
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")


def parse_bool(value: object) -> bool | None:
    """Parse common string and integer boolean representations."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None
