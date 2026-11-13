"""Cell barcodes of reads: a lookup in a barcode list, with an optional group (cluster) per cell."""

from __future__ import annotations

import gzip
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from cubam.bam import Tag, fnv1a


def read_barcodes(path: str | Path) -> list[str]:
    """Barcodes of a file with one per line (plain or gzip), e.g. ``barcodes.tsv.gz`` of Cell Ranger."""
    path = Path(path)
    with open(path, "rb") as fh:
        gz = fh.read(2) == b"\x1f\x8b"
    with gzip.open(path, "rt") if gz else open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


class CellIndex:
    """Cells of a barcode list, looked up on the GPU from the hash of a tag.

    The order of ``barcodes`` is the order of the cells: :meth:`lookup`
    returns the index of each read's barcode in it, or -1. Any string works
    as a barcode. Barcodes are matched on their 64-bit FNV-1a hash.

    ``labels`` gives a group to each cell (a cluster, a sample), as a
    sequence of the same length as ``barcodes``. Missing labels (``None`` or
    NaN) put the cell in no group. ``groups`` lists the groups in order of
    first appearance, and :meth:`group` maps cells to indices into it.
    """

    def __init__(self, barcodes: Sequence[str], labels: Sequence | None = None):
        import cupy as cp

        self.barcodes = list(barcodes)
        if not self.barcodes:
            raise ValueError("Empty barcode list")
        h = fnv1a(self.barcodes)
        if len(np.unique(h)) != len(self.barcodes):
            raise ValueError("Duplicate barcodes in the barcode list")
        order = np.argsort(h)
        self._sorted = cp.asarray(h[order])
        self._index = cp.asarray(order.astype(np.int32))
        self._packed = None
        self.groups: list = []
        self._group = None
        if labels is not None:
            labels = pd.Series(list(labels) if not isinstance(labels, pd.Series) else labels.to_numpy())
            if len(labels) != len(self.barcodes):
                raise ValueError(f"{len(labels)} labels for {len(self.barcodes)} barcodes")
            codes, uniq = pd.factorize(labels, sort=False)
            self.groups = list(uniq)
            self._group = cp.asarray(codes.astype(np.int32))

    @classmethod
    def from_file(cls, path: str | Path, labels: Sequence | None = None) -> CellIndex:
        return cls(read_barcodes(path), labels)

    def __len__(self) -> int:
        return len(self.barcodes)

    def lookup(self, tag: Tag):
        """Index of each read's cell (int32 cupy array), -1 when its barcode is missing or not in the list."""
        import cupy as cp

        h = tag.hash()
        j = cp.minimum(cp.searchsorted(self._sorted, h), len(self.barcodes) - 1)
        return cp.where(self._sorted[j] == h, self._index[j], -1)

    def lookup_packed(self, codes):
        """Index of the cell of packed barcodes (:func:`packed_barcodes`, the ``bc`` of fragments); -1 if absent.

        Barcodes of the list that do not fit a code match nothing.
        """
        import cupy as cp

        if self._packed is None:
            code = pack_acgt(self.barcodes)
            fits = np.flatnonzero(code != 0)
            order = fits[np.argsort(code[fits])]
            self._packed = (cp.asarray(code[order]), cp.asarray(order.astype(np.int32)))
        keys, index = self._packed
        codes = cp.asarray(codes, cp.uint64)
        if keys.size == 0:
            return cp.full(codes.size, -1, cp.int32)
        j = cp.minimum(cp.searchsorted(keys, codes), keys.size - 1)
        return cp.where((keys[j] == codes) & (codes != 0), index[j], -1).astype(cp.int32)

    def group_sizes(self) -> np.ndarray:
        """Number of cells in each group."""
        if self._group is None:
            raise ValueError("This CellIndex has no labels")
        codes = self._group.get()
        return np.bincount(codes[codes >= 0], minlength=len(self.groups))

    def group(self, cell):
        """Group index of cells (from :meth:`lookup`); -1 for no cell or no group."""
        import cupy as cp

        if self._group is None:
            raise ValueError("This CellIndex has no labels")
        cell = cp.asarray(cell)
        return cp.where(cell >= 0, self._group[cp.maximum(cell, 0)], -1)


def pack_acgt(values: Sequence[str]) -> np.ndarray:
    """Codes of :meth:`Tag.packed` computed on the host; 0 for a value that does not fit."""
    out = np.zeros(len(values), np.uint64)
    base = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, v in enumerate(values):
        seq, dash, suffix = v.partition("-")
        if not 1 <= len(seq) <= 24 or any(c not in base for c in seq):
            continue
        if dash and not (1 <= len(suffix) <= 3 and suffix.isascii() and suffix.isdigit()):
            continue
        code = 0
        for k, c in enumerate(seq):
            code |= base[c] << (2 * k)
        code |= len(seq) << 48
        if dash:
            code |= (int(suffix) + 1) << 53
        out[i] = code
    return out


def packed_barcodes(tag: Tag):
    """Barcodes packed into uint64 codes (:meth:`Tag.packed`), and whether each read has one.

    For applications that keep every barcode, without a list. Raises when a
    barcode does not fit the code (``[ACGT]{1,24}`` with an optional ``-N``).
    """
    code, ok = tag.packed()
    if bool((tag.present & ~ok).any()):
        raise ValueError(f"{tag.name} values other than [ACGT]+ with an optional -N are not supported yet")
    return code, ok
