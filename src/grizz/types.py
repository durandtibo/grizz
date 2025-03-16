r"""Contain types."""

from __future__ import annotations

__all__ = ["Self"]

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing_extensions import (
        Self,  # use backport because it was added in python 3.11
    )
