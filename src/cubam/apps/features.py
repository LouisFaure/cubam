"""
Counts per cell and feature (genes, transposable elements, peaks...) from a BAM file.

Features come from an :class:`~cubam.intervals.IntervalIndex` (a GTF or a
BED file) or from a tag of the reads (``GX``, the gene of Cell Ranger).
Counting runs in two steps:

1. **Per read.** The positions of a read (its aligned blocks, or its 5' or
   3' end) are matched to features. Matches on the wrong strand
   (``strand``) or shorter than ``min_overlap`` bases are dropped. The
   ``assign`` rule then picks the features of the read, as featureCounts
   does: ``"unique"`` keeps a read that matches exactly one feature (the
   default of featureCounts), ``"any"`` counts it for each feature it
   matches (``-O``), ``"largest"`` for the feature with the most overlapping
   bases when there is a single one (``--largestOverlap``).
2. **Per molecule.** With ``umi_tag``, a (cell, UMI, feature) counts once.
   Without, every read counts.

Reads of a tag value with ``;`` (Cell Ranger lists several genes so) are
ambiguous and are not counted.
"""

from __future__ import annotations

import gzip
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cubam.bam import (
    DUPLICATE,
    QC_FAIL,
    REVERSE,
    SECONDARY,
    SUPPLEMENTARY,
    UNMAPPED,
    BamHeader,
    RecordBatch,
    read_bam,
)
from cubam.cells import CellIndex
from cubam.contigs import keep_mask
from cubam.filters import ReadFilter
from cubam.intervals import IntervalIndex

_PRIMARY = UNMAPPED | SECONDARY | SUPPLEMENTARY | QC_FAIL


@dataclass
class FeatureCounts:
    """Counts per cell and feature.

    ``counts`` is a cuDF DataFrame of ``cell`` (index into ``barcodes``),
    ``feature`` (row of ``features``) and ``count``, sorted by cell and
    feature, without zeros.
    """

    counts: object
    barcodes: list[str]
    features: pd.DataFrame
    stats: dict = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.barcodes), len(self.features)

    def sparse(self):
        """A cells × features ``cupyx.scipy.sparse.csr_matrix`` of counts (float32: cupyx has no integer matrices)."""
        import cupy as cp
        import cupyx.scipy.sparse as sp

        c = self.counts
        return sp.csr_matrix(
            (c["count"].values.astype(cp.float32), (c["cell"].values, c["feature"].values)), shape=self.shape
        )

    def to_anndata(self):
        """An AnnData of the counts (needs the ``anndata`` extra), cells as obs, features as var."""
        import anndata
        import scipy.sparse as sp

        c = self.counts.to_pandas()
        x = sp.csr_matrix((c["count"].to_numpy(np.int32), (c["cell"], c["feature"])), shape=self.shape)
        var = self.features.set_index("name")
        var.index = var.index.astype(str)
        return anndata.AnnData(x, obs=pd.DataFrame(index=pd.Index(self.barcodes, name="barcode")), var=var)

    def write_mtx(self, outdir: str | Path) -> None:
        """Write ``matrix.mtx.gz``, ``barcodes.tsv.gz`` and ``features.tsv.gz`` (features × cells), as Cell Ranger."""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        c = self.counts.sort_values(["cell", "feature"]).to_pandas()
        with gzip.open(outdir / "matrix.mtx.gz", "wt", compresslevel=1) as fh:
            fh.write("%%MatrixMarket matrix coordinate integer general\n%\n")
            fh.write(f"{len(self.features)} {len(self.barcodes)} {len(c)}\n")
            body = (c["feature"] + 1).astype(str) + " " + (c["cell"] + 1).astype(str) + " " + c["count"].astype(str)
            fh.write("\n".join(body) + ("\n" if len(body) else ""))
        with gzip.open(outdir / "barcodes.tsv.gz", "wt") as fh:
            fh.write("".join(b + "\n" for b in self.barcodes))
        f = self.features
        label = f["gene_name"] if "gene_name" in f else f["name"]
        with gzip.open(outdir / "features.tsv.gz", "wt") as fh:
            for n, g in zip(f["name"], label.fillna(f["name"]), strict=True):
                fh.write(f"{n}\t{g}\tGene Expression\n")


def _per_read_features(rec, feat, bases, rule: str, acc: dict):
    """Apply the assign rule to (read, feature, bases) rows sorted by read; returns the kept rows' indices."""
    import cupy as cp

    n = int(rec.size)
    if n == 0:
        return cp.empty(0, cp.int64)
    first = cp.ones(n, cp.bool_)
    first[1:] = rec[1:] != rec[:-1]
    seg = cp.cumsum(first) - 1
    starts = cp.flatnonzero(first)
    per_read = cp.diff(cp.concatenate([starts, cp.asarray([n])]))
    if rule == "any":
        return cp.arange(n)
    if rule == "unique":
        ok = per_read[seg] == 1
        acc["ambiguous"] += int((per_read > 1).sum())
        return cp.flatnonzero(ok)
    if rule == "largest":
        import cudf

        df = cudf.DataFrame({"seg": seg, "bases": bases})
        mx = df.groupby("seg", sort=True)["bases"].max().sort_index().values
        top = bases == mx[seg]
        ntop = cp.bincount(seg[top], minlength=starts.size)
        acc["ambiguous"] += int((ntop > 1).sum())
        return cp.flatnonzero(top & (ntop[seg] == 1))
    raise ValueError(f"assign must be 'unique', 'any' or 'largest', not {rule!r}")


def _positions(sub: RecordBatch, positions: str):
    """(read, start, end) of the positions of each read."""
    import cupy as cp

    if positions == "blocks":
        return sub.blocks()
    n = len(sub)
    rev = (sub.flag & REVERSE) != 0
    five = cp.where(rev, sub.end - 1, sub.pos)
    three = cp.where(rev, sub.pos, sub.end - 1)
    p = five if positions == "five_prime" else three
    return cp.arange(n, dtype=cp.int64), p.astype(cp.int32), (p + 1).astype(cp.int32)


def count_features(
    bam: str | Path,
    features: IntervalIndex | str,
    cells: CellIndex | list[str] | str | Path,
    *,
    cell_tag: str = "CB",
    umi_tag: str | None = "UB",
    positions: str = "blocks",
    strand: str | None = None,
    assign: str = "unique",
    min_overlap: int = 1,
    read_filter: ReadFilter | None = None,
    where: Callable[[RecordBatch], object] | None = None,
    keep_chroms: list[str] | None = None,
    drop_chroms: tuple[str, ...] = (),
    chunk_bytes: int = 256 << 20,
) -> FeatureCounts:
    """Count reads or UMIs per cell and feature.

    Parameters
    ----------
    bam
        A coordinate-sorted BAM file.
    features
        An :class:`~cubam.intervals.IntervalIndex`, or the name of a tag
        that names the feature of each read (``"GX"`` in a Cell Ranger BAM).
    cells
        The cell barcodes (a :class:`~cubam.cells.CellIndex`, a list or a
        file). Reads of other barcodes are skipped.
    umi_tag
        UMI tag; ``None`` counts reads.
    positions
        What of a read is matched to features: ``"blocks"`` (aligned blocks,
        split at introns), ``"five_prime"`` or ``"three_prime"`` (the
        aligned base at that end).
    strand
        ``"sense"`` keeps matches where the read and the feature are on the
        same strand (a 10x 3' library), ``"antisense"`` the opposite,
        ``None`` both. Features without a strand always match.
    assign
        ``"unique"``, ``"any"`` or ``"largest"`` (see the module docs).
    min_overlap
        Minimum number of bases of a read on a feature.
    read_filter
        Which reads count. Default: primary alignments with MAPQ of at least
        30, and no duplicate flag when there is no UMI.
    where
        An extra test on each batch, returning a boolean cupy array per
        record (e.g. ``lambda b: (b.tag("xf").value & 8) != 0``).
    """
    import cudf
    import cupy as cp

    from cubam import memory_pool

    t0 = time.perf_counter()
    header = BamHeader.read(bam)
    header.require_coordinate_sorted(bam)
    if positions not in ("blocks", "five_prime", "three_prime"):
        raise ValueError(f"positions must be 'blocks', 'five_prime' or 'three_prime', not {positions!r}")
    if strand not in (None, "sense", "antisense"):
        raise ValueError(f"strand must be 'sense', 'antisense' or None, not {strand!r}")
    if assign not in ("unique", "any", "largest"):
        raise ValueError(f"assign must be 'unique', 'any' or 'largest', not {assign!r}")
    by_tag = isinstance(features, str)
    if by_tag and assign != "unique":
        raise ValueError("Features from a tag are counted with assign='unique' only")
    if read_filter is None:
        read_filter = ReadFilter(min_mapq=30, exclude=_PRIMARY | (0 if umi_tag else DUPLICATE))
    names = header.names
    keep = keep_mask(names, keep_chroms, drop_chroms)
    stats = dict.fromkeys(("records", "reads", "no_feature", "ambiguous", "assigned"), 0)
    tag_names: dict[int, str] = {}  # tag value hash -> value
    with memory_pool():
        if not isinstance(cells, CellIndex):
            cells = CellIndex(cells) if isinstance(cells, list) else CellIndex.from_file(cells)
        keep_ref = cp.asarray(keep)
        multi_gpu = cp.empty(0, cp.uint64)  # hashes of tag values that list several features
        if not by_tag:
            fstrand = features.strand_codes()
            features.segments(names)  # build once, before the first batch
        parts = []
        for batch in read_bam(bam, chunk_bytes):
            stats["records"] += len(batch)
            ok = read_filter.mask(batch, keep_ref)
            if where is not None:
                ok &= cp.asarray(where(batch), cp.bool_)
            sub = batch.take(ok)
            del batch
            cell = cells.lookup(sub.tag(cell_tag))
            has = cell >= 0
            if umi_tag:
                ut = sub.tag(umi_tag)
                has &= ut.type == ord("Z")
            k = cp.flatnonzero(has)
            sub, cell = sub.take(k), cell[k]
            umi = ut.hash()[k] if umi_tag else None
            n = len(sub)
            stats["reads"] += n
            if n == 0:
                continue
            if by_tag:
                t = sub.tag(features)
                present = cp.flatnonzero(t.type == ord("Z"))
                fh = t.hash()[present]
                # names of new tag values
                u, first = np.unique(fh.get(), return_index=True)
                new = np.flatnonzero([int(x) not in tag_names for x in u])
                if new.size:
                    vals = t.strings().take(present[cp.asarray(first[new])]).to_arrow().to_pylist()
                    tag_names.update(zip((int(x) for x in u[new]), vals, strict=True))
                    multi = np.array([h for h, v in tag_names.items() if ";" in v], np.uint64)
                    multi_gpu = cp.asarray(multi)
                stats["no_feature"] += n - int(present.size)
                amb = cp.isin(fh, multi_gpu) if multi_gpu.size else cp.zeros(fh.size, cp.bool_)
                stats["ambiguous"] += int(amb.sum())
                read, feat = present[~amb], fh[~amb]
            else:
                rec, start, end = _positions(sub, positions)
                q, feat, bases = features.overlaps(names, sub.ref[rec], start, end)
                read = rec[q]
                # one row per (read, feature): positions of a read can meet a feature more than once
                key = (read << 32) | feat.astype(cp.int64)
                order = cp.argsort(key)
                key, bases = key[order], bases[order]
                last = cp.ones(key.size, cp.bool_)
                if key.size:
                    last[:-1] = key[1:] != key[:-1]
                cs = cp.cumsum(bases)
                at = cp.flatnonzero(last)
                bases = cs[at] - cp.concatenate([cp.zeros(1, cs.dtype), cs[at][:-1]])
                read, feat = key[at] >> 32, (key[at] & 0xFFFFFFFF).astype(cp.int32)
                good = bases >= min_overlap
                if strand is not None:
                    rs = cp.where((sub.flag[read] & REVERSE) != 0, -1, 1).astype(cp.int8)
                    fs = fstrand[feat]
                    good &= (fs == 0) | (rs * fs == (1 if strand == "sense" else -1))
                g = cp.flatnonzero(good)
                read, feat, bases = read[g], feat[g], bases[g]
                pick = _per_read_features(read, feat, bases, assign, stats)
                matched = int(cp.unique(read).size)
                read, feat = read[pick], feat[pick]
                stats["no_feature"] += n - matched
            df = cudf.DataFrame({"cell": cell[read], "feature": feat})
            if umi_tag:
                df["umi"] = umi[read]
                df = df.drop_duplicates(["cell", "umi", "feature"])
            else:
                df = df.groupby(["cell", "feature"], sort=False).size().rename("count").reset_index()
            parts.append(df)
            del sub
        if parts:
            df = cudf.concat(parts, ignore_index=True)
            del parts
            if umi_tag:
                df = df.drop_duplicates(["cell", "umi", "feature"])
                df = df.groupby(["cell", "feature"], sort=False).size().rename("count").reset_index()
            else:
                df = df.groupby(["cell", "feature"], sort=False).agg({"count": "sum"}).reset_index()
        else:
            df = cudf.DataFrame(
                {k: cudf.Series([], dtype=t) for k, t in (("cell", "int32"), ("feature", "uint64"), ("count", "int64"))}
            )
        if by_tag:
            # feature rows in order of tag value, for the values that name a single feature
            single = sorted((v, h) for h, v in tag_names.items() if ";" not in v)
            ftable = pd.DataFrame({"name": [v for v, _ in single]})
            hashes = np.array([h for _, h in single], np.uint64)
            order = np.argsort(hashes)
            keys = cp.asarray(hashes[order])
            pos = cp.searchsorted(keys, df["feature"].values)
            df["feature"] = cp.asarray(order.astype(np.int32))[pos] if len(df) else cp.empty(0, cp.int32)
        else:
            ftable = features.features.copy()
        df["count"] = df["count"].astype("int32")
        df = df.sort_values(["cell", "feature"]).reset_index(drop=True)
        stats["assigned"] = stats["reads"] - stats["no_feature"] - stats["ambiguous"]
    stats["entries"] = len(df)
    return FeatureCounts(df, cells.barcodes, ftable, stats, time.perf_counter() - t0)
