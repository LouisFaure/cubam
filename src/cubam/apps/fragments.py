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
position read are final, so the output comes sorted by position.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

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

_READ_COLS = ("ref", "pos", "flag", "nref", "npos", "hash", "bc", "five")
_KEY = ["ref", "start", "end", "bc"]


@dataclass
class Fragments:
    """Fragments called from a BAM file.

    ``table`` is a cuDF DataFrame of ``ref`` (index into ``header.names``),
    ``start``, ``end``, ``bc`` (packed barcode, see :func:`cubam.bam.unpack_acgt`)
    and ``count``, sorted by position.
    """

    table: object
    header: BamHeader
    stats: dict = field(default_factory=dict)
    seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.table)

    def to_frame(self):
        """A cuDF DataFrame with ``chrom`` and ``barcode`` as strings (Cell Ranger columns)."""
        import cudf

        t = self.table
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
        self.done = []

    def add(self, frag, last_ref: int, last_pos: int, final: bool):
        import cudf

        parts = [p for p in (self.pending, None if frag is None else cudf.DataFrame(frag)) if p is not None]
        if not parts:
            return
        df = cudf.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        df = df.groupby(_KEY, sort=False).agg({"count": "sum", "rep": "sum"}).reset_index()
        if final:
            fin = None
        elif self.margin is None:
            self.pending = df
            return
        else:
            fin = (df["ref"] < last_ref) | ((df["ref"] == last_ref) & (df["start"] < last_pos - self.margin))
        out = df if fin is None else df[fin]
        self.pending = None if fin is None else df[~fin]
        if self.rep:
            norep = out["rep"] == 0
            self.acc["no_representative"] += norep.values.sum()
            out = out[~norep]
        self.done.append(out.drop(columns="rep").sort_values(_KEY))

    def result(self):
        import cudf

        if not self.done:
            return cudf.DataFrame({k: cudf.Series([], dtype=t) for k, t in zip(_KEY + ["count"], "iiiQi", strict=True)})
        return cudf.concat(self.done, ignore_index=True)


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
    chunk_bytes: int = 256 << 20,
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
    import cupy as cp

    from cubam import memory_pool

    t0 = time.perf_counter()
    path = Path(path)
    header = BamHeader.read(path)
    if header.sort_order != "coordinate":
        raise ValueError(f"{path}: not coordinate-sorted (@HD SO:{header.sort_order})")
    names = header.names
    n_ref = len(names)
    keep = np.ones(n_ref, bool)
    if keep_chroms is not None:
        unknown = sorted(set(keep_chroms) - set(names))
        if unknown:
            raise ValueError(f"Contigs not in the BAM header: {unknown[:5]}")
        keep[:] = False
        keep[[names.index(c) for c in keep_chroms]] = True
    for c in drop_chroms:
        if c in names:
            keep[names.index(c)] = False

    with memory_pool():
        keep_ref = cp.asarray(keep)
        ref_len = cp.asarray(header.lengths, cp.int32)
        reject = UNMAPPED | MATE_UNMAPPED | SECONDARY | QC_FAIL | SUPPLEMENTARY
        need = PAIRED | (PROPER_PAIR if require_proper_pair else 0)
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
            ok = batch.has_flags(need, reject) & (batch.mapq >= min_mapq) & (batch.ref >= 0) & (batch.ref == batch.nref)
            ok &= keep_ref[cp.maximum(batch.ref, 0)]
            sub = batch.take(ok)
            del batch
            tag = sub.tag(barcode_tag)
            bc, packable = tag.packed()
            if bool((tag.present & ~packable).any()):
                raise ValueError(f"{barcode_tag} values other than [ACGT]+ with an optional -N are not supported yet")
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
                dedup.add({k: v[~short] for k, v in frag.items()}, last_ref, last_pos, False)
        if waiting is not None:  # mates never seen: they failed a filter
            acc["mate_dropped"] += waiting["hash"].size
        dedup.add(None, last_ref, last_pos, True)
        table = dedup.result()
        stats = {k: int(v) for k, v in acc.items()}
    stats["fragments"] = len(table)
    return Fragments(table, header, stats, time.perf_counter() - t0)
