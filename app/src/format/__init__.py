"""
Import formatters from here rather than from their individual modules so that
internal file organisation can change without affecting callers::

    from app.src.format import PassthroughFormatter

Available formatters
--------------------
PassthroughFormatter
    No-op default.  Returns pipeline output wrapped in a minimal envelope
    with no field renaming or type coercion.  Use this when no project-specific
    CDM is required.

Extending
---------
Add new formatters by subclassing :class:`~app.src.format.base.DataFormatter`
and exporting the class here.  See ``base.py`` for the full extension guide.
"""

from app.src.format.base import DataFormatter
from app.src.format.passthrough import PassthroughFormatter

__all__ = [
    "DataFormatter",
    "PassthroughFormatter",
]