"""
Coverage tracks per group of cells (e.g. clusters), from a BAM file or from ATAC fragments.

Each read or fragment adds +1 where it starts and -1 where it ends, under
the group of its cell. Sorting these events by group and position and
summing them gives runs of constant coverage, as a bedGraph has. The input
is sorted by position, so events behind the last position read are final:
runs are built as the file is read, and memory stays bounded.

- :func:`coverage` covers the aligned blocks of reads (RNA, or any BAM).
  Introns (``N``) split a read into blocks; deletions do not, unless
  ``split_deletions`` (as ``bedtools genomecov -split``).
- :func:`fragment_coverage` covers ATAC fragments, called with the rules
  of Cell Ranger (:func:`cubam.apps.fragments.call_fragments`): the Tn5
  insertions at both ends of each fragment (as ArchR), or whole fragments.

:meth:`Coverage.write_bigwig` writes one bigwig per group, normalised to
counts per million (CPM) by default.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cubam.bam import (
    DUPLICATE,
    QC_FAIL,
    REVERSE,
    SECONDARY,
    SUPPLEMENTARY,
    UNMAPPED,
    BamHeader,
    read_bam,
)
from cubam.cells import CellIndex
from cubam.contigs import keep_mask
from cubam.filters import ReadFilter

#: Default filter of :func:`coverage`: primary, unique (MAPQ 30 drops multi-mappers of STAR and Cell
#: Ranger), not duplicates (Cell Ranger flags all but one read per UMI)
DEFAULT_READ_FILTER = ReadFilter(min_mapq=30, exclude=UNMAPPED | SECONDARY | SUPPLEMENTARY | QC_FAIL | DUPLICATE)


@dataclass
class Coverage:
    """Runs of constant coverage per group.

    ``runs`` is a cuDF DataFrame of ``group`` (index into ``groups``),
    ``ref`` (index into ``header.names``), ``start``, ``end`` (0-based,
    end exclusive) and ``value`` (count), sorted by group and position.
    Runs of zero coverage are left out. ``units`` holds the number of reads,
    fragments or insertions counted in each group (the CPM denominator), and
    ``n_cells`` the number of cells of each group in the barcode list.
    """

    runs: object
    header: BamHeader
    groups: list
    units: np.ndarray
    n_cells: np.ndarray
    unit: str
    stats: dict = field(default_factory=dict)
    seconds: float = 0.0

    def scale(self, normalize: str | None) -> np.ndarray:
        """Factor per group: ``"cpm"`` per million units, ``"cells"`` per cell, ``None`` for counts."""
        with np.errstate(divide="ignore"):
            if normalize == "cpm":
                return np.where(self.units > 0, 1e6 / np.maximum(self.units, 1), 0.0)
            if normalize == "cells":
                return np.where(self.n_cells > 0, 1.0 / np.maximum(self.n_cells, 1), 0.0)
        if normalize is None:
            return np.ones(len(self.groups))
        raise ValueError(f"normalize must be 'cpm', 'cells' or None, not {normalize!r}")

    def to_frame(self, group, normalize: str | None = None, bin_size: int | None = None):
        """The track of one group (a label of ``groups``) as a cuDF DataFrame: ref, start, end, value.

        With ``bin_size``, the value is the mean coverage over each bin of
        that many bases (bins with no coverage are left out).
        """
        import cupy as cp

        g = self.groups.index(group)
        r = self.runs[self.runs["group"] == g]
        ref = r["ref"].values
        start, end = r["start"].values, r["end"].values
        value = r["value"].values.astype(cp.float64) * self.scale(normalize)[g]
        if bin_size and bin_size > 1:
            ref_len = cp.asarray(self.header.lengths, cp.int64)
            ref, start, end, value = _bin(ref, start, end, value, int(bin_size), ref_len)
        import cudf

        return cudf.DataFrame({"ref": ref, "start": start, "end": end, "value": value.astype(cp.float32)})

    def write_bigwig(
        self,
        outdir: str | Path,
        normalize: str | None = "cpm",
        bin_size: int | None = None,
        prefix: str = "",
        n_jobs: int | None = None,
    ) -> dict:
        """Write ``<outdir>/<prefix><group>.bw`` for each group; returns ``{group: path}``.

        Needs pyBigWig (``pip install cubam[bigwig]``). The bigwig header lists
        every contig of the BAM header, in its order. pyBigWig writes about a
        million runs per second, so groups are written in ``n_jobs`` processes
        (default: the CPUs this process may use).
        """
        try:
            import pyBigWig  # noqa: F401
        except ImportError as e:
            raise ImportError("write_bigwig needs pyBigWig: pip install pyBigWig") from e
        import json
        import os
        import subprocess
        import sys
        import tempfile

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        contigs = json.dumps([[n, ln] for n, ln in zip(self.header.names, self.header.lengths, strict=True)])
        paths, used = {}, set()
        for group in self.groups:
            stem = prefix + re.sub(r"[^A-Za-z0-9._+-]+", "_", str(group))
            while stem in used:
                stem += "_"
            used.add(stem)
            paths[group] = outdir / f"{stem}.bw"
        n_jobs = max(1, min(n_jobs or len(os.sched_getaffinity(0)), len(self.groups)))
        running: list = []

        def wait_one():
            proc, group = running.pop(0)
            _, err = proc.communicate()
            if proc.returncode:
                raise RuntimeError(f"Writing the bigwig of {group!r} failed:\n{err.decode(errors='replace')}")

        with tempfile.TemporaryDirectory(dir=outdir, prefix=".cubam-") as tmp:
            try:
                for i, group in enumerate(self.groups):
                    df = self.to_frame(group, normalize, bin_size)
                    runs = Path(tmp) / f"{i}.npz"
                    cols = {k: df[k].values.get() for k in ("ref", "start", "end", "value")}
                    np.savez(runs, contigs=contigs, **cols)
                    del df, cols
                    while len(running) >= n_jobs:
                        wait_one()
                    cmd = [sys.executable, "-m", "cubam._bigwig", str(runs), str(paths[group])]
                    running.append((subprocess.Popen(cmd, stderr=subprocess.PIPE), group))
                while running:
                    wait_one()
            finally:
                for proc, _ in running:
                    proc.kill()
                    proc.wait()
        return paths


def _bin(ref, start, end, value, bin_size, ref_len):
    """Mean of runs over bins of ``bin_size`` bases (the last bin of a contig is shorter)."""
    import cudf
    import cupy as cp

    start, end = start.astype(cp.int64), end.astype(cp.int64)
    b0, b1 = start // bin_size, (end - 1) // bin_size
    nb = b1 - b0 + 1
    ends = cp.cumsum(nb)
    total = int(ends[-1]) if nb.size else 0
    j = cp.arange(total, dtype=cp.int64)
    rec = cp.searchsorted(ends, j, side="right")
    b = b0[rec] + (j - (ends[rec] - nb[rec]))
    ov = cp.minimum(end[rec], (b + 1) * bin_size) - cp.maximum(start[rec], b * bin_size)
    df = cudf.DataFrame({"ref": ref[rec], "bin": b, "w": value[rec] * ov})
    df = df.groupby(["ref", "bin"], sort=True).agg({"w": "sum"}).reset_index()
    r, b = df["ref"].values, df["bin"].values
    s = b * bin_size
    e = cp.minimum(s + bin_size, ref_len[r])
    return r, s.astype(cp.int32), e.astype(cp.int32), df["w"].values / (e - s)


class _Runs:
    """Streaming sum of +1/-1 events per group into runs of constant coverage.

    Events are ``(group, key, delta)`` with ``key = ref << 32 | pos``. Events
    with a key below the threshold passed to :meth:`add` are final: no later
    event can come before them. The run that starts at the last final event
    of a group stays open (``open_key``, ``carry``) until the next one.
    """

    def __init__(self, n_groups: int, ref_len):
        import cupy as cp

        self.open_key = cp.full(n_groups, -1, cp.int64)
        self.carry = cp.zeros(n_groups, cp.int64)
        self.ref_len = ref_len
        self.pending = None
        self.out = []

    def add(self, g, key, delta, threshold: int | None):
        import cupy as cp

        if self.pending is not None:
            pg, pk, pd = self.pending
            g, key, delta = cp.concatenate([pg, g]), cp.concatenate([pk, key]), cp.concatenate([pd, delta])
        if threshold is None:
            self.pending = None
        else:
            fin = key < threshold
            rest = ~fin
            self.pending = (g[rest], key[rest], delta[rest])
            g, key, delta = g[fin], key[fin], delta[fin]
        if g.size:
            self._flush(g, key, delta)

    def finish(self):
        """Flush every pending event (the end of the input)."""
        if self.pending is not None:
            g, key, delta = self.pending
            self.pending = None
            if g.size:
                self._flush(g, key, delta)

    def _flush(self, g, key, delta):
        import cupy as cp

        opened = cp.flatnonzero(self.open_key >= 0).astype(cp.int32)
        n_new = int(g.size)
        g = cp.concatenate([g.astype(cp.int32), opened])
        key = cp.concatenate([key, self.open_key[opened]])
        delta = cp.concatenate([delta.astype(cp.int64), cp.zeros(opened.size, cp.int64)])
        pseudo = cp.arange(g.size) >= n_new
        order = cp.lexsort(cp.stack([key, g.astype(cp.int64)]))
        g, key, delta, pseudo = g[order], key[order], delta[order], pseudo[order]
        # Sum the deltas of each (group, key)
        last = cp.ones(g.size, cp.bool_)
        last[:-1] = (g[1:] != g[:-1]) | (key[1:] != key[:-1])
        cs = cp.cumsum(delta)
        at = cp.flatnonzero(last)
        csl = cs[at]
        d = csl - cp.concatenate([cp.zeros(1, cp.int64), csl[:-1]])
        keep = (d != 0) | pseudo[at]
        g, key, d = g[at][keep], key[at][keep], d[keep]
        if g.size == 0:
            return
        # Coverage after each event: carry of the group plus the deltas so far in the group
        first = cp.ones(g.size, cp.bool_)
        first[1:] = g[1:] != g[:-1]
        cs = cp.cumsum(d)
        seg = cp.cumsum(first) - 1
        fi = cp.flatnonzero(first)
        val = cs - (cs[fi] - d[fi])[seg] + self.carry[g]
        # A run from each event to the next one of the same group
        nxt = cp.zeros(g.size, cp.bool_)
        nxt[:-1] = ~first[1:]
        i = cp.flatnonzero(nxt & (val != 0))
        ref = (key[i] >> 32).astype(cp.int32)
        end = cp.where((key[i + 1] >> 32) == ref, key[i + 1] & 0xFFFFFFFF, self.ref_len[ref])
        self.out.append(
            {
                "group": g[i],
                "ref": ref,
                "start": (key[i] & 0xFFFFFFFF).astype(cp.int32),
                "end": end.astype(cp.int32),
                "value": val[i].astype(cp.int32),
            }
        )
        li = cp.flatnonzero(~nxt)
        self.carry[g[li]] = val[li]
        self.open_key[g[li]] = cp.where(val[li] != 0, key[li], -1)

    def result(self):
        import cudf
        import cupy as cp

        if self.pending is not None or bool((self.open_key >= 0).any()):
            raise RuntimeError("Coverage runs still open at the end")
        if not self.out:
            cols = ("group", "ref", "start", "end", "value")
            return cudf.DataFrame({k: cudf.Series([], dtype="int32") for k in cols})
        cols = {k: cp.concatenate([o[k] for o in self.out]) for k in self.out[0]}
        return cudf.DataFrame(cols).sort_values(["group", "ref", "start"]).reset_index(drop=True)


def _events(g, ref, start, end):
    """+1 at start and -1 at end of each interval, as (group, key, delta)."""
    import cupy as cp

    r = ref.astype(cp.int64) << 32
    key = cp.concatenate([r | start.astype(cp.int64), r | end.astype(cp.int64)])
    delta = cp.concatenate([cp.ones(start.size, cp.int32), cp.full(start.size, -1, cp.int32)])
    return cp.concatenate([g, g]), key, delta


def _group_sizes(cells: CellIndex) -> tuple[list, np.ndarray]:
    if not cells.groups:
        raise ValueError("The CellIndex needs labels: CellIndex(barcodes, labels=...)")
    return cells.groups, cells.group_sizes()


def coverage(
    bam: str | Path,
    cells: CellIndex,
    *,
    cell_tag: str = "CB",
    read_filter: ReadFilter = DEFAULT_READ_FILTER,
    strand: str | None = None,
    split_deletions: bool = False,
    keep_chroms: list[str] | None = None,
    drop_chroms: tuple[str, ...] = (),
    chunk_bytes: int = 256 << 20,
) -> Coverage:
    """Coverage of the aligned blocks of reads, per group of cells.

    Parameters
    ----------
    bam
        A coordinate-sorted BAM file.
    cells
        Barcodes with a group per cell: ``CellIndex(barcodes, labels=clusters)``.
        Reads of other barcodes, or of cells without a group, are skipped.
    read_filter
        Which reads count. The default keeps primary alignments with a MAPQ
        of at least 30, without the duplicate flag (one read per UMI in a
        Cell Ranger BAM).
    strand
        ``"+"`` or ``"-"`` keeps reads aligned to that strand only (for a
        10x 3' library, the strand of the transcript). ``None`` keeps both.
    split_deletions
        Leave deletions uncovered, as introns are.
    """
    import cupy as cp

    from cubam import memory_pool

    t0 = time.perf_counter()
    header = BamHeader.read(bam)
    header.require_coordinate_sorted(bam)
    if strand not in (None, "+", "-"):
        raise ValueError(f"strand must be '+', '-' or None, not {strand!r}")
    groups, n_cells = _group_sizes(cells)
    n_ref = len(header.names)
    keep = keep_mask(header.names, keep_chroms, drop_chroms)
    stats = dict.fromkeys(("records", "reads", "no_cell", "blocks"), 0)
    with memory_pool():
        keep_ref = cp.asarray(keep)
        ref_len = cp.asarray(header.lengths, cp.int64)
        runs = _Runs(len(groups), ref_len)
        units = cp.zeros(len(groups), cp.int64)
        for batch in read_bam(bam, chunk_bytes):
            stats["records"] += len(batch)
            last_ref, last_pos = (int(x) for x in cp.stack([batch.ref[-1], batch.pos[-1]]).get())
            threshold = (n_ref << 32) if last_ref < 0 else (last_ref << 32) | last_pos
            ok = read_filter.mask(batch, keep_ref)
            if strand is not None:
                rev = (batch.flag & REVERSE) != 0
                ok &= rev if strand == "-" else ~rev
            sub = batch.take(ok)
            del batch
            grp = cells.group(cells.lookup(sub.tag(cell_tag)))
            has = grp >= 0
            stats["no_cell"] += int(sub.ref.size) - int(has.sum())
            sub, grp = sub.take(has), grp[has]
            stats["reads"] += len(sub)
            units += cp.bincount(grp, minlength=len(groups))
            rec, start, end = sub.blocks(split_deletions)
            stats["blocks"] += int(rec.size)
            runs.add(*_events(grp[rec], sub.ref[rec], start, cp.minimum(end, ref_len[sub.ref[rec]])), threshold)
            del sub, rec, start, end
        runs.finish()
        table = runs.result()
    stats["runs"] = len(table)
    return Coverage(table, header, groups, units.get(), n_cells, "reads", stats, time.perf_counter() - t0)


def fragment_coverage(
    source,
    cells: CellIndex,
    *,
    mode: str = "insertions",
    insertion_width: int = 1,
    **fragment_kw,
) -> Coverage:
    """Coverage of ATAC fragments per group of cells.

    Parameters
    ----------
    source
        A :class:`~cubam.apps.fragments.Fragments` (from ``call_fragments``)
        or a BAM file, from which fragments are called with ``fragment_kw``
        (e.g. ``keep_chroms``).
    cells
        Barcodes with a group per cell: ``CellIndex(barcodes, labels=clusters)``.
    mode
        ``"insertions"``: the Tn5 insertion at each end of a fragment, the
        bases ``start`` and ``end - 1``, each ``insertion_width`` bases wide
        and centred on it. ``"fragments"``: the whole fragment.
        Duplicates of a fragment count once.
    """
    import cupy as cp

    from cubam import memory_pool
    from cubam.apps.fragments import Fragments, call_fragments

    t0 = time.perf_counter()
    if mode not in ("insertions", "fragments"):
        raise ValueError(f"mode must be 'insertions' or 'fragments', not {mode!r}")
    if insertion_width < 1:
        raise ValueError("insertion_width must be at least 1")
    groups, n_cells = _group_sizes(cells)
    frags = source if isinstance(source, Fragments) else call_fragments(source, **fragment_kw)
    header = frags.header
    t = frags.table
    stats = {"fragments": len(t)}
    with memory_pool():
        ref_len = cp.asarray(header.lengths, cp.int64)
        runs = _Runs(len(groups), ref_len)
        units = cp.zeros(len(groups), cp.int64)
        refs = t["ref"].values
        bounds = cp.searchsorted(refs, cp.arange(len(header.names) + 1, dtype=refs.dtype)).get()
        in_cells = 0
        for r in range(len(header.names)):
            lo, hi = int(bounds[r]), int(bounds[r + 1])
            if hi == lo:
                continue
            grp = cells.group(cells.lookup_packed(t["bc"].values[lo:hi]))
            k = cp.flatnonzero(grp >= 0)
            in_cells += int(k.size)
            grp = grp[k]
            start = t["start"].values[lo:hi][k].astype(cp.int64)
            end = t["end"].values[lo:hi][k].astype(cp.int64)
            ref = cp.full(k.size, r, cp.int32)
            if mode == "fragments":
                units += cp.bincount(grp, minlength=len(groups))
                runs.add(*_events(grp, ref, start, end), None)
                continue
            units += 2 * cp.bincount(grp, minlength=len(groups))
            p = cp.concatenate([start, end - 1]) - insertion_width // 2
            s = cp.clip(p, 0, ref_len[r])
            e = cp.clip(p + insertion_width, 0, ref_len[r])
            g2, r2 = cp.concatenate([grp, grp]), cp.concatenate([ref, ref])
            w = e > s
            runs.add(*_events(g2[w], r2[w], s[w], e[w]), None)
        table = runs.result()
    stats.update(fragments_in_cells=in_cells, runs=len(table))
    unit = "insertions" if mode == "insertions" else "fragments"
    return Coverage(table, header, groups, units.get(), n_cells, unit, stats, time.perf_counter() - t0)
