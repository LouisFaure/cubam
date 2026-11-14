"""
ATAC fragments from a coordinate-sorted, paired-end BAM, with the rules of Cell Ranger.

The rules come from the source of Cell Ranger ATAC 1.2
(``lib/python/tools/peaks.py`` and the ``MARK_DUPLICATES`` stage), and match
the ``atac_fragments.tsv.gz`` of Cell Ranger ARC 2.0.2 on every row and count:

- The fragment runs from the 5' end of the forward read to the 5' end of the
  reverse read, soft clips included, shifted by +4 and -5 bp (Tn5).
- Both reads need a MAPQ above 30.
- Chimeric pairs are dropped: mates on different chromosomes or the same
  strand, facing outward (5' ends compared), or with alignment starts more
  than 5,000 bp apart.
- Contigs outside ``keep_chroms`` (the primary contigs of the reference) are
  dropped. A fragment whose start lies before the contig start is dropped,
  and the end is clipped to the contig length.
- Pairs with the same barcode, chromosome, start and end are one fragment,
  with ``count`` pairs. The filters apply to the representative of a group,
  the pair without the duplicate flag: when it fails, the group is dropped.

Mates are joined on the hash of the read name (the BAM files of Cell Ranger
have no ``MQ`` tag, so the left mate alone cannot tell whether the right one
passes the MAPQ filter). Duplicates are summed by a streaming group-by. Rows
that start more than ``max_mate_distance`` plus 64 kb behind the last
position read are final, so the output comes sorted by position. Final rows
move to host memory as they come, so GPU memory does not grow with the
number of fragments.
"""

from __future__ import annotations

import io
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cubam.bam import (
    DUPLICATE,
    MATE_UNMAPPED,
    PAIRED,
    PROPER_PAIR,
    QC_FAIL,
    REVERSE,
    SECONDARY,
    SUPPLEMENTARY,
    UNMAPPED,
    BamHeader,
    read_bam,
    unpack_acgt,
)
from cubam.cells import packed_barcodes
from cubam.contigs import keep_mask
from cubam.filters import ReadFilter

_READ_COLS = ("ref", "pos", "flag", "nref", "npos", "hash", "bc", "five")
_KEY = ["ref", "start", "end", "bc"]
_DTYPES = {"ref": np.int32, "start": np.int32, "end": np.int32, "bc": np.uint64, "count": np.int32}


@dataclass
class Fragments:
    """Fragments called from a BAM file.

    ``table`` is a pandas DataFrame, in host memory, of ``ref`` (index into
    ``header.names``), ``start``, ``end``, ``bc`` (packed barcode, see
    :func:`cubam.bam.unpack_acgt`) and ``count``, sorted by position.
    """

    table: object
    header: BamHeader
    stats: dict = field(default_factory=dict)
    seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.table)

    def to_frame(self, lo: int = 0, hi: int | None = None):
        """Rows ``lo:hi`` as a cuDF DataFrame with ``chrom`` and ``barcode`` as strings (Cell Ranger columns).

        All rows of a whole sample can take tens of GB of GPU memory as
        strings: :meth:`write_tsv` writes them a slice at a time.
        """
        import cudf

        t = cudf.from_pandas(self.table.iloc[lo:hi])
        codes, uniq = t["bc"].factorize()
        names = unpack_acgt(uniq.values)
        chroms = cudf.Series(self.header.names)
        return cudf.DataFrame(
            {
                "chrom": chroms.take(t["ref"].values).reset_index(drop=True),
                "start": t["start"].reset_index(drop=True),
                "end": t["end"].reset_index(drop=True),
                "barcode": names.take(codes).reset_index(drop=True),
                "count": t["count"].reset_index(drop=True),
            }
        )

    def write_tsv(self, path: str | Path, rows_per_slice: int = 10_000_000) -> None:
        """Write the fragments as Cell Ranger does (chrom, start, end, barcode, count), BGZF when ``path`` ends in .gz.

        The file has no header lines. Index a .gz file with ``tabix -p bed``.
        """
        from cubam.bgzf import BgzfWriter

        path = Path(path)
        gz = path.suffix == ".gz"
        with BgzfWriter(path) if gz else open(path, "wb") as out:
            for lo in range(0, len(self.table), rows_per_slice):
                buf = io.BytesIO()  # cuDF writes bytes faster than a str
                self.to_frame(lo, lo + rows_per_slice).to_csv(buf, sep="\t", header=False, index=False)
                out.write(buf.getbuffer())


def _concat(a, b):
    import cupy as cp

    if a is None:
        return b
    return {k: cp.concatenate([a[k], b[k]]) for k in _READ_COLS}


def _take(cols, idx):
    return {k: cols[k][idx] for k in _READ_COLS}


def _pair(reads, last_ref: int, last_pos: int, final: bool, acc):
    """Join mates on the name hash; returns the pairs (left, right) and the reads still waiting."""
    import cupy as cp

    n = int(reads["hash"].size)
    if n < 2:
        return None, reads
    order = cp.argsort(reads["hash"])
    h = reads["hash"][order]
    same = h[1:] == h[:-1]
    first = cp.zeros(n, cp.bool_)
    first[:-1] = same
    first[1:-1] &= ~same[:-1]  # a triple would pair twice: keep the first two
    i = cp.flatnonzero(first)
    a, b = order[i], order[i + 1]
    paired = cp.zeros(n, cp.bool_)
    paired[a] = True
    paired[b] = True
    rest = cp.flatnonzero(~paired)
    # A read waits if its mate lies at or after the last position read
    nr, np_ = reads["nref"][rest], reads["npos"][rest]
    wait = (nr > last_ref) | ((nr == last_ref) & (np_ >= last_pos))
    if final:
        wait[:] = False
    acc["mate_dropped"] += (~wait).sum()
    return (_take(reads, a), _take(reads, b)), _take(reads, rest[wait])


def _fragments(left, right, ref_len, shift, max_mate_distance, acc):
    import cupy as cp

    lrev = (left["flag"] & REVERSE) != 0
    rrev = (right["flag"] & REVERSE) != 0
    fwd5 = cp.where(lrev, right["five"], left["five"])
    rev5 = cp.where(lrev, left["five"], right["five"])
    ok = (left["bc"] == right["bc"]) & (left["ref"] == right["ref"]) & (lrev != rrev)
    ok &= fwd5 <= rev5  # outward-facing: ARC 2 compares 5' ends (ATAC 1.2 compared alignment starts)
    if max_mate_distance is not None:
        ok &= cp.abs(left["pos"] - right["pos"]) <= max_mate_distance
    ok &= fwd5 >= 0
    acc["dropped_pairs"] += (~ok).sum()
    k = cp.flatnonzero(ok)
    ref = left["ref"][k]
    return {
        "ref": ref,
        "start": fwd5[k] + shift[0],
        "end": cp.minimum(rev5[k] + shift[1], ref_len[ref]),
        "bc": left["bc"][k],
        "count": cp.ones(k.size, cp.int32),
        # the representative of a group of duplicates has no duplicate flag
        "rep": ((left["flag"][k] & DUPLICATE) == 0).astype(cp.int32),
    }


class _Dedup:
    """Streaming sum of duplicates; rows too far behind the last position read are final."""

    def __init__(self, max_mate_distance, require_representative, acc):
        self.margin = None if max_mate_distance is None else max_mate_distance + (1 << 16)
        self.rep = require_representative
        self.acc = acc
        self.pending = None

    def add(self, frag, last_ref: int, last_pos: int, final: bool):
        """Add fragments; returns the final ones (a cuDF DataFrame sorted by position, or None) and their bound.

        No later fragment has a key ``ref << 32 | start`` below the bound
        (``None`` when ``final``).
        """
        import cudf

        parts = [p for p in (self.pending, None if frag is None else cudf.DataFrame(frag)) if p is not None]
        if not parts:
            return None, None
        df = cudf.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        df = df.groupby(_KEY, sort=False).agg({"count": "sum", "rep": "sum"}).reset_index()
        if final:
            fin = bound = None
        elif self.margin is None:
            self.pending = df
            return None, None
        else:
            fin = (df["ref"] < last_ref) | ((df["ref"] == last_ref) & (df["start"] < last_pos - self.margin))
            bound = (last_ref << 32) + last_pos - self.margin
        out = df if fin is None else df[fin]
        self.pending = None if fin is None else df[~fin]
        if self.rep:
            norep = out["rep"] == 0
            self.acc["no_representative"] += norep.values.sum()
            out = out[~norep]
        return out.drop(columns="rep").sort_values(_KEY), bound


def call_fragments(
    path: str | Path,
    *,
    keep_chroms: list[str] | None = None,
    drop_chroms: tuple[str, ...] = (),
    barcode_tag: str = "CB",
    min_mapq: int = 31,
    shift: tuple[int, int] = (4, -5),
    max_mate_distance: int | None = 5000,
    require_proper_pair: bool = False,
    require_representative: bool = True,
    min_length: int = 1,
    chunk_bytes: int | None = None,
) -> Fragments:
    """Call ATAC fragments from a coordinate-sorted, paired-end BAM file on the GPU.

    Parameters
    ----------
    keep_chroms
        Contigs to keep, e.g. the primary contigs of a Cell Ranger reference
        (``contig-defs.json``, or the ``# primary_contig=`` lines of a Cell
        Ranger fragments file). The BAM header does not list them. ``None``
        keeps every contig.
    drop_chroms
        Contigs to drop, e.g. ``("chrM",)``.
    barcode_tag
        Tag of the cell barcode. Values must be ``[ACGT]+`` with an optional
        ``-N`` suffix.
    min_mapq
        Minimum MAPQ of both reads (Cell Ranger keeps MAPQ above 30).
    max_mate_distance
        Maximum distance between the alignment starts of the mates (Cell
        Ranger ``READ_MATE_FAR_DIST``). ``None`` for no limit, which keeps
        every row pending until the end.
    require_representative
        Drop a group of duplicates whose representative (the pair without the
        duplicate flag) fails the filters, as Cell Ranger does.
    """
    from cubam import memory_pool

    t0 = time.perf_counter()
    path = Path(path)
    header = BamHeader.read(path)
    header.require_coordinate_sorted(path)
    stats: dict = {}
    host = {k: [] for k in _DTYPES}
    with memory_pool():
        for part, _ in _fragment_parts(
            path,
            header,
            stats,
            keep_chroms=keep_chroms,
            drop_chroms=drop_chroms,
            barcode_tag=barcode_tag,
            min_mapq=min_mapq,
            shift=shift,
            max_mate_distance=max_mate_distance,
            require_proper_pair=require_proper_pair,
            require_representative=require_representative,
            min_length=min_length,
            chunk_bytes=chunk_bytes,
        ):
            for k in host:
                host[k].append(part[k].values.get())
    cols = {}
    for k in _DTYPES:  # one column at a time, freeing its parts: host memory peaks near the table size
        v = host.pop(k)
        cols[k] = np.concatenate(v) if v else np.empty(0, _DTYPES[k])
        del v
    table = pd.DataFrame(cols, copy=False)
    stats["fragments"] = len(table)
    return Fragments(table, header, stats, time.perf_counter() - t0)


def _fragment_parts(
    path: Path,
    header: BamHeader,
    stats: dict,
    *,
    keep_chroms: list[str] | None = None,
    drop_chroms: tuple[str, ...] = (),
    barcode_tag: str = "CB",
    min_mapq: int = 31,
    shift: tuple[int, int] = (4, -5),
    max_mate_distance: int | None = 5000,
    require_proper_pair: bool = False,
    require_representative: bool = True,
    min_length: int = 1,
    chunk_bytes: int | None = None,
) -> Iterator:
    """Final fragments as they come, as ``(part, bound)``: a cuDF DataFrame sorted by position on the GPU.

    No later fragment has a key ``ref << 32 | start`` below ``bound``
    (``None`` for the last part). ``stats`` is filled at the end. Run it
    inside :func:`cubam.memory_pool`. Arguments as :func:`call_fragments`.
    """
    import cupy as cp

    n_ref = len(header.names)
    keep = keep_mask(header.names, keep_chroms, drop_chroms)
    read_filter = ReadFilter(
        min_mapq=min_mapq,
        require=PAIRED | (PROPER_PAIR if require_proper_pair else 0),
        exclude=UNMAPPED | MATE_UNMAPPED | SECONDARY | QC_FAIL | SUPPLEMENTARY,
        same_contig_as_mate=True,
    )

    keep_ref = cp.asarray(keep)
    ref_len = cp.asarray(header.lengths, cp.int32)
    keys = (
        "records",
        "kept_reads",
        "no_barcode",
        "mate_dropped",
        "dropped_pairs",
        "too_short",
        "no_representative",
    )
    acc = {k: cp.zeros((), cp.int64) for k in keys}
    dedup = _Dedup(max_mate_distance, require_representative, acc)
    waiting = None
    last_ref = last_pos = -1
    for batch in read_bam(path, chunk_bytes):
        acc["records"] += len(batch)
        last_ref, last_pos = (int(x) for x in cp.stack([batch.ref[-1], batch.pos[-1]]).get())
        if last_ref < 0:  # unplaced unmapped reads at the end
            last_ref = n_ref
        sub = batch.take(read_filter.mask(batch, keep_ref))
        del batch
        tag = sub.tag(barcode_tag)
        bc, packable = packed_barcodes(tag)
        acc["no_barcode"] += (~packable).sum()
        sub = sub.take(packable)
        acc["kept_reads"] += len(sub)
        reads = {
            "ref": sub.ref,
            "pos": sub.pos,
            "flag": sub.flag,
            "nref": sub.nref,
            "npos": sub.npos,
            "hash": sub.name_hash(),
            "bc": bc[packable],
            "five": sub.five_prime(),
        }
        del sub, tag
        pairs, waiting = _pair(_concat(waiting, reads), last_ref, last_pos, False, acc)
        if pairs is not None:
            frag = _fragments(*pairs, ref_len, shift, max_mate_distance, acc)
            short = frag["end"] - frag["start"] < min_length
            acc["too_short"] += short.sum()
            out, bound = dedup.add({k: v[~short] for k, v in frag.items()}, last_ref, last_pos, False)
            if out is not None and len(out):
                yield out, bound
    if waiting is not None:  # mates never seen: they failed a filter
        acc["mate_dropped"] += waiting["hash"].size
    out, _ = dedup.add(None, last_ref, last_pos, True)
    if out is not None and len(out):
        yield out, None
    stats.update({k: int(v) for k, v in acc.items()})
