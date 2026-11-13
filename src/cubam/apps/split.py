"""
A BAM file split into one BAM file per group of cells (clusters) or per cell, on the GPU.

The records of each batch are looked up in a :class:`~cubam.cells.CellIndex`
and sorted by group, stably, so that each output keeps the order of the input
(a coordinate-sorted input gives coordinate-sorted outputs). Their bytes are
gathered per group and buffered on the GPU. Once enough is buffered, the full
BGZF blocks of every group are compressed in one batched nvCOMP call
(:func:`cubam.bgzf.bgzf_compress`) and appended to the files. Each output has
the header of the input, with a ``@PG`` line.

``samtools index`` indexes the outputs.
"""

from __future__ import annotations

import re
import shlex
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cubam._kernels import copy_spans
from cubam.bam import BamHeader, read_bam
from cubam.bgzf import BLOCK_DATA, EOF_BLOCK, bgzf_compress
from cubam.cells import CellIndex
from cubam.filters import ReadFilter


@dataclass
class SplitResult:
    """The files written (``{label: path}``) and the records in each (``{label: n}``)."""

    paths: dict
    records: dict
    stats: dict = field(default_factory=dict)
    seconds: float = 0.0


class _Outputs:
    """GPU buffers of record bytes per output, compressed in batches and appended to the files."""

    def __init__(self, paths: list[Path], header: bytes, level: int, flush_bytes: int):
        import cupy as cp

        self.paths = paths
        self.level = level
        self.flush_bytes = flush_bytes
        self.pending: list[list] = [[] for _ in paths]
        self.size = np.zeros(len(paths), np.int64)
        self.handles: OrderedDict = OrderedDict()  # open files, least recently used first
        head = cp.asarray(np.frombuffer(header, np.uint8))
        first = bgzf_compress(head, level).get().tobytes()  # the header in blocks of its own
        for p in paths:
            p.write_bytes(first)

    def add(self, group: int, data) -> None:
        self.pending[group].append(data)
        self.size[group] += int(data.size)

    def maybe_flush(self) -> None:
        if self.size.sum() >= self.flush_bytes:
            self.flush(final=False)

    def flush(self, final: bool) -> None:
        import cupy as cp

        take = self.size if final else self.size // BLOCK_DATA * BLOCK_DATA
        groups = np.flatnonzero(take > 0)
        if groups.size == 0:
            return
        bufs, rests = [], {}
        for g in groups:
            buf = cp.concatenate(self.pending[g]) if len(self.pending[g]) > 1 else self.pending[g][0]
            bufs.append(buf[: take[g]])
            rest = buf[take[g] :]
            rests[g] = rest.copy() if rest.size else None
        # blocks of BLOCK_DATA bytes that do not cross groups (the last one of a group may be shorter)
        start = np.cumsum(np.r_[0, take[groups][:-1]])
        n_blocks = (take[groups] + BLOCK_DATA - 1) // BLOCK_DATA
        first_block = np.r_[0, np.cumsum(n_blocks)[:-1]]
        k = np.arange(n_blocks.sum()) - np.repeat(first_block, n_blocks)
        gi = np.repeat(np.arange(groups.size), n_blocks)
        boff = start[gi] + k * BLOCK_DATA
        bsize = np.minimum(BLOCK_DATA, take[groups][gi] - k * BLOCK_DATA)
        data = cp.concatenate(bufs) if len(bufs) > 1 else bufs[0]
        out, out_off = bgzf_compress(data, self.level, boff, bsize, return_offsets=True)
        out_off = np.r_[out_off.get(), out.size]
        host = out.get()
        for j, g in enumerate(groups):
            lo, hi = out_off[first_block[j]], out_off[first_block[j] + n_blocks[j]]
            self._handle(g).write(host[lo:hi].tobytes())
            r = rests[g]
            self.pending[g] = [r] if r is not None else []
            self.size[g] = 0 if r is None else int(r.size)

    def _handle(self, g: int, max_open: int = 256):
        fh = self.handles.pop(g, None)
        if fh is None:
            if len(self.handles) >= max_open:
                self.handles.popitem(last=False)[1].close()
            fh = open(self.paths[g], "ab")  # noqa: SIM115  (closed in close())
        self.handles[g] = fh
        return fh

    def close(self) -> None:
        try:
            self.flush(final=True)
            for g in range(len(self.paths)):
                self._handle(g).write(EOF_BLOCK)
        finally:
            for fh in self.handles.values():
                fh.close()
            self.handles.clear()


def _stem(label) -> str:
    return re.sub(r"[^A-Za-z0-9._+-]+", "_", str(label))


def split_bam(
    bam: str | Path,
    cells: CellIndex,
    outdir: str | Path,
    *,
    by: str | None = None,
    cell_tag: str = "CB",
    read_filter: ReadFilter | None = None,
    prefix: str = "",
    level: int = 2,
    flush_bytes: int = 1 << 30,
    chunk_bytes: int = 256 << 20,
) -> SplitResult:
    """Write the records of each group of cells (or of each cell) to ``<outdir>/<prefix><label>.bam``.

    Parameters
    ----------
    cells
        The barcodes, with ``labels`` to split by group. Records of other
        barcodes, of cells without a group, or without the tag are skipped.
    by
        ``"group"`` (the default when ``cells`` has labels) or ``"cell"``:
        one file per barcode of ``cells``.
    read_filter
        Which records to keep; ``None`` keeps every record of a cell.
    level
        nvCOMP Deflate level (:func:`cubam.bgzf.deflate_spans`): 2 by
        default, 5 for files about 9% smaller than zlib level 6, slower.
    flush_bytes
        Record bytes buffered on the GPU before compressing and writing.
    """
    import cupy as cp

    from cubam import memory_pool

    t0 = time.perf_counter()
    if by is None:
        by = "group" if cells.groups else "cell"
    if by not in ("group", "cell"):
        raise ValueError(f"by must be 'group' or 'cell', not {by!r}")
    if by == "group" and not cells.groups:
        raise ValueError("Splitting by group needs labels: CellIndex(barcodes, labels=...)")
    labels = cells.groups if by == "group" else cells.barcodes
    header = BamHeader.read(bam)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths, used = [], set()
    for label in labels:
        stem = prefix + _stem(label)
        while stem in used:
            stem += "_"
        used.add(stem)
        paths.append(outdir / f"{stem}.bam")
    cl = shlex.join([Path(sys.argv[0]).name or "python", *sys.argv[1:]])
    pg = f"@PG\tID:cubam.split_bam\tPN:cubam\tCL:{cl}\tDS:split by {by} on {cell_tag}\n"
    text = header.text if header.text.endswith("\n") or not header.text else header.text + "\n"
    n_groups = len(labels)
    counts = np.zeros(n_groups, np.int64)
    stats = dict.fromkeys(("records", "kept", "no_cell"), 0)
    with memory_pool():
        outs = _Outputs(paths, header.to_bytes(text + pg), level, flush_bytes)
        for batch in read_bam(bam, chunk_bytes):
            n = len(batch)
            stats["records"] += n
            sub = batch if read_filter is None else batch.take(read_filter.mask(batch))
            cell = cells.lookup(sub.tag(cell_tag))
            grp = cells.group(cell) if by == "group" else cell
            keep = cp.flatnonzero(grp >= 0)
            stats["no_cell"] += len(sub) - int(keep.size)
            if keep.size == 0:
                continue
            grp = grp[keep].astype(cp.int64)
            # stable sort by group: the key holds the input order
            order = cp.argsort((grp << 32) | keep.astype(cp.int64))
            idx = keep[order]
            g = grp[order]
            size = sub.record_sizes()[idx]
            dst = cp.cumsum(size) - size
            buf = cp.empty(int(dst[-1] + size[-1]), cp.uint8)
            copy_spans(sub.data, buf, sub.offset[idx], dst, size)
            gs = cp.bincount(g, minlength=n_groups)
            counts += gs.get()
            stats["kept"] += int(keep.size)
            ends = cp.cumsum(cp.bincount(g, weights=size.astype(cp.float64), minlength=n_groups)).astype(cp.int64)
            ends = ends.get()
            starts = np.r_[0, ends[:-1]]
            for j in np.flatnonzero(gs.get()):
                outs.add(int(j), buf[starts[j] : ends[j]])
            del batch, sub, buf
            outs.maybe_flush()
        outs.close()
    return SplitResult(
        dict(zip(labels, paths, strict=True)),
        dict(zip(labels, counts.tolist(), strict=True)),
        stats,
        time.perf_counter() - t0,
    )
