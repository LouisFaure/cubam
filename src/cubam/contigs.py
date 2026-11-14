"""Contig names to reference ids of a BAM header, and masks of the contigs to keep."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def ref_ids(chroms: pd.Series | Iterable[str], names: list[str]) -> np.ndarray:
    """Reference ids of contig names, with or without a ``chr`` prefix (as cellsnp-lite); -1 if absent."""
    chroms = pd.Series(chroms, dtype=object) if not isinstance(chroms, pd.Series) else chroms
    index = {n: i for i, n in enumerate(names)}
    out = {}
    for c in chroms.unique():
        if c in index:
            out[c] = index[c]
        elif c.startswith("chr"):
            out[c] = index.get(c[3:], -1)
        else:
            out[c] = index.get("chr" + c, -1)
    return chroms.map(out).to_numpy(np.int64)


def keep_mask(names: list[str], keep: Iterable[str] | None = None, drop: Iterable[str] = ()) -> np.ndarray:
    """A boolean mask over ``names``: the contigs of ``keep`` (all when ``None``), minus those of ``drop``.

    Names in ``keep`` must be in the header; names in ``drop`` may be absent.
    """
    mask = np.ones(len(names), bool)
    if keep is not None:
        keep = list(keep)
        unknown = sorted(set(keep) - set(names))
        if unknown:
            raise ValueError(f"Contigs not in the BAM header: {unknown[:5]}")
        mask[:] = False
        mask[[names.index(c) for c in keep]] = True
    for c in drop:
        if c in names:
            mask[names.index(c)] = False
    return mask
