"""
Genomic intervals (GTF, BED) cut into disjoint segments, for overlap queries on the GPU.

Features can overlap (genes in genes, exons of the transcripts of a gene).
The genome is cut at every start and end of an interval, so each segment
between two cuts holds a fixed set of features (featureCounts and HTSeq do
the same). The cuts are sorted ``int64`` keys ``ref << 32 | pos``, and a
CSR array maps each segment to its features. A query interval is then two
binary searches; an interval that crosses several segments gives one row per
segment.

A feature is a named group of intervals: a gene made of its exons
(``from_gtf(feature="exon", group_by="gene_id")``), a gene body
(``feature="gene"``), a peak or a repeat (``from_bed``).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cubam.contigs import ref_ids


def _read_text(path: Path) -> io.BytesIO | str:
    """A plain path, or the inflated bytes of a gzip or BGZF file (cuDF reads only the first gzip member)."""
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic != b"\x1f\x8b":
        return str(path)
    import gzip

    try:
        import cupy as cp

        from cubam.bgzf import inflate_chunks

        return io.BytesIO(cp.concatenate(list(inflate_chunks(path))).get().tobytes())
    except ValueError:  # gzip, not BGZF
        with gzip.open(path, "rb") as fh:
            return io.BytesIO(fh.read())


@dataclass
class _Segments:
    """Segments of an index for the contigs of one BAM header (cupy arrays)."""

    bounds: object  # sorted unique keys ref << 32 | pos, the cuts
    ptr: object  # CSR offsets: segment i = [bounds[i], bounds[i+1]) has features feat[ptr[i]:ptr[i+1]]
    feat: object


class IntervalIndex:
    """Intervals grouped into named features, ready for overlap queries against a BAM header.

    Parameters
    ----------
    chrom, start, end
        The intervals: contig names (a sequence or a ``pd.Categorical``, the
        fast way for millions of intervals), 0-based starts and exclusive ends.
    feature
        The feature of each interval (any hashable label). Features are
        numbered in order of first appearance; ``features`` lists them.
    strand
        ``"+"``, ``"-"`` or ``"."`` per interval; a feature takes the strand
        of its first interval. ``None`` for unstranded features.
    attrs
        Extra columns per feature (e.g. ``gene_name``), a DataFrame indexed
        like ``features["name"]``.
    """

    def __init__(self, chrom, start, end, feature, strand=None, attrs: pd.DataFrame | None = None):
        if not isinstance(chrom, pd.Categorical):
            chrom = pd.Categorical(np.asarray(chrom, dtype=object))
        start = np.asarray(start, np.int64)
        end = np.asarray(end, np.int64)
        feature = np.asarray(feature)
        if not len(chrom) == len(start) == len(end) == len(feature):
            raise ValueError("chrom, start, end and feature must have the same length")
        if (end < start).any() or (start < 0).any():
            raise ValueError("Intervals need 0 <= start <= end")
        if feature.dtype.kind not in "iub":
            feature = feature.astype(object)
        if feature.dtype.kind in "iu" and np.array_equal(feature, np.arange(len(feature))):
            codes, names = feature, feature  # one feature per interval, already numbered
        else:
            codes, names = pd.factorize(feature, sort=False)
        if (codes < 0).any():
            raise ValueError("Features cannot be missing")
        self._chrom = chrom
        self._start = start
        self._end = end
        self._feat = codes.astype(np.int32)
        # first interval of each feature: codes number features in order of first appearance
        first = np.arange(len(codes)) if len(names) == len(codes) else np.unique(codes, return_index=True)[1]
        table = pd.DataFrame({"name": names if names.dtype.kind in "iub" else np.asarray(names, dtype=object)})
        table["chrom"] = pd.Categorical.from_codes(chrom.codes[first], categories=chrom.categories)
        if strand is not None:
            table["strand"] = np.asarray(strand, dtype=object)[first]
        if attrs is not None:
            table = table.join(attrs, on="name")
        self.features = table
        self._cache: dict = {}

    def __len__(self) -> int:
        return len(self.features)

    @property
    def n_intervals(self) -> int:
        return len(self._start)

    @classmethod
    def from_gtf(
        cls,
        path: str | Path,
        feature: str = "exon",
        group_by: str = "gene_id",
        attrs: tuple[str, ...] = ("gene_name", "gene_type"),
    ) -> IntervalIndex:
        """Rows of one type of a GTF file (``exon``, ``gene``...), grouped into features by an attribute.

        GTF coordinates are 1-based and inclusive; they become 0-based and
        end-exclusive. ``attrs`` are attributes kept per feature when present.
        """
        import cudf

        cols = ["chrom", "source", "type", "start", "end", "score", "strand", "frame", "attr"]
        df = cudf.read_csv(
            _read_text(Path(path)),
            sep="\t",
            comment="#",
            header=None,
            names=cols,
            usecols=["chrom", "type", "start", "end", "strand", "attr"],
            dtype={"chrom": "str", "type": "str", "start": "int64", "end": "int64", "strand": "str", "attr": "str"},
        )
        df = df[df["type"] == feature]
        if len(df) == 0:
            raise ValueError(f"{path}: no rows of type {feature!r}")
        key = df["attr"].str.extract(f'{group_by} "([^"]*)"')[0]
        if key.isna().any():
            raise ValueError(f"{path}: {int(key.isna().sum())} {feature} rows without {group_by}")
        extra = {a: df["attr"].str.extract(f'{a} "([^"]*)"')[0] for a in attrs}
        names = key.to_pandas().to_numpy()
        table = None
        if extra:
            table = pd.DataFrame({"name": names, **{a: v.to_pandas().to_numpy() for a, v in extra.items()}})
            table = table.dropna(axis=1, how="all").drop_duplicates("name").set_index("name")
        return cls(
            df["chrom"].to_pandas().to_numpy(),
            df["start"].to_pandas().to_numpy() - 1,
            df["end"].to_pandas().to_numpy(),
            names,
            strand=df["strand"].to_pandas().to_numpy(),
            attrs=table,
        )

    @classmethod
    def from_bed(cls, path: str | Path, name: bool | None = None) -> IntervalIndex:
        """Intervals of a BED file. The 4th column names features when present (``name=False`` ignores it);
        otherwise each interval is its own feature, named ``chrom:start-end``. The 6th column gives strands."""
        src = _read_text(Path(path))

        def header_lines(fh) -> int:
            n = 0
            for line in fh:
                if not line.startswith((b"track", b"browser", b"#")):
                    break
                n += 1
            return n

        if isinstance(src, str):
            with open(src, "rb") as fh:
                skip = header_lines(fh)
        else:
            skip = header_lines(src)
            src.seek(0)
        df = pd.read_csv(src, sep="\t", header=None, skiprows=skip, dtype={0: str})
        chrom, start, end = df[0].astype(str), df[1].astype(np.int64), df[2].astype(np.int64)
        if name is None:
            name = df.shape[1] > 3
        feature = (
            df[3].astype(str) if name and df.shape[1] > 3 else chrom + ":" + start.astype(str) + "-" + end.astype(str)
        )
        strand = df[5].astype(str).to_numpy() if df.shape[1] > 5 else None
        return cls(chrom.to_numpy(), start.to_numpy(), end.to_numpy(), feature.to_numpy(), strand=strand)

    def strand_codes(self):
        """Strand of each feature as a cupy int8 array: 1 for ``+``, -1 for ``-``, 0 otherwise."""
        import cupy as cp

        if "strand" not in self.features:
            return cp.zeros(len(self), cp.int8)
        s = self.features["strand"].to_numpy()
        return cp.asarray(np.where(s == "+", 1, np.where(s == "-", -1, 0)).astype(np.int8))

    def segments(self, names: list[str]) -> _Segments:
        """The segments for the contigs ``names`` of a BAM header (cached). Intervals on other contigs are dropped."""
        import cupy as cp

        key = tuple(names)
        if key in self._cache:
            return self._cache[key]
        ref = ref_ids(pd.Series(self._chrom.categories.astype(str)), names)[self._chrom.codes]
        ref[self._chrom.codes < 0] = -1
        ok = ref >= 0
        r = cp.asarray(ref[ok]) << 32
        ks = r | cp.asarray(self._start[ok])
        ke = r | cp.asarray(self._end[ok])
        feat = cp.asarray(self._feat[ok])
        nonempty = ke > ks
        ks, ke, feat = ks[nonempty], ke[nonempty], feat[nonempty]
        bounds = cp.unique(cp.concatenate([ks, ke]))
        lo = cp.searchsorted(bounds, ks)
        hi = cp.searchsorted(bounds, ke)
        seg, which = _expand(lo, hi - lo)
        pair = cp.unique((seg.astype(cp.int64) << 32) | feat[which].astype(cp.int64))
        pseg = (pair >> 32).astype(cp.int64)
        pfeat = (pair & 0xFFFFFFFF).astype(cp.int32)
        ptr = cp.searchsorted(pseg, cp.arange(bounds.size + 1, dtype=cp.int64))
        s = _Segments(bounds, ptr, pfeat)
        self._cache[key] = s
        return s

    def overlaps(self, names: list[str], ref, start, end):
        """Features overlapping the query intervals ``[start, end)`` on ``ref`` (ids into ``names``).

        Returns cupy arrays ``(query, feature, bases)``: one row per query and
        feature it overlaps, with the number of overlapping bases, ordered by query.
        """
        import cupy as cp

        s = self.segments(names)
        nb = int(s.bounds.size)
        empty = (cp.empty(0, cp.int64), cp.empty(0, cp.int32), cp.empty(0, cp.int64))
        if nb == 0 or ref.size == 0:
            return empty
        r = cp.asarray(ref).astype(cp.int64) << 32
        ks = r | cp.asarray(start).astype(cp.int64)
        ke = r | cp.asarray(end).astype(cp.int64)
        a = cp.maximum(cp.searchsorted(s.bounds, ks, side="right") - 1, 0)
        b = cp.searchsorted(s.bounds, ke, side="left") - 1  # the segment holding end - 1
        n = cp.where(ke > ks, b - a + 1, 0)
        seg, q = _expand(a, cp.maximum(n, 0))
        nf = s.ptr[seg + 1] - s.ptr[seg]
        keep = cp.flatnonzero(nf > 0)
        seg, q, nf = seg[keep], q[keep], nf[keep]
        segend = s.bounds[cp.minimum(seg + 1, nb - 1)]
        bases = cp.minimum(ke[q], segend) - cp.maximum(ks[q], s.bounds[seg])
        j, row = _expand(s.ptr[seg], nf)
        feat = s.feat[j]
        q, bases = q[row], bases[row]
        # one row per (query, feature): a query can meet a feature in several segments
        key = (q.astype(cp.int64) << 32) | feat.astype(cp.int64)
        order = cp.argsort(key)
        key, bases = key[order], bases[order]
        last = cp.ones(key.size, cp.bool_)
        if key.size:
            last[:-1] = key[1:] != key[:-1]
        cs = cp.cumsum(bases)
        at = cp.flatnonzero(last)
        tot = cs[at] - cp.concatenate([cp.zeros(1, cs.dtype), cs[at][:-1]])
        k = key[at]
        return (k >> 32).astype(cp.int64), (k & 0xFFFFFFFF).astype(cp.int32), tot.astype(cp.int64)


def _expand(first, count):
    """For each i, the values first[i] .. first[i] + count[i] - 1: returns (values, i)."""
    import cupy as cp

    count = count.astype(cp.int64)
    ends = cp.cumsum(count)
    total = int(ends[-1]) if ends.size else 0
    j = cp.arange(total, dtype=cp.int64)
    i = cp.searchsorted(ends, j, side="right")
    return first[i].astype(cp.int64) + (j - (ends[i] - count[i])), i
