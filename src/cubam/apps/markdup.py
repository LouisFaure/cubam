"""
Duplicate marking of a coordinate-sorted BAM file on the GPU, as ``samtools markdup``.

Gives the duplicate flags of ``samtools fixmate -m`` followed by ``samtools markdup``
(template mode, 1.22), optionally with ``--barcode-tag`` so that duplicates must share
a cell barcode (10x scATAC). The input needs the ``MC`` tag (BWA-MEM and Parabricks
fq2bam write it) but no ``ms`` tag: the mate score is computed here.

Three passes over the file:

1. Pair scores. The score of a read is the sum of its base qualities of at least 15.
   The scores of the primary reads are summed per read name, which gives the score of
   a read plus that of its mate (``calc_score + ms`` in samtools). A score waits on the
   GPU only until its mate is read; the sums go to host memory (4 bytes per record).
2. Duplicates. Every examined read (primary, mapped, not QC-failed) gets the pair key
   of samtools (barcode, unclipped 5' coordinates of the read and of its mate from
   ``MC``, orientation, leftmost) when it has a mapped mate, and the single key
   (barcode, unclipped 5' coordinate, strand) in any case. In a group of reads with
   the same pair key the read with the highest pair score stays, ties going to the
   smallest read name; the others are duplicates. A read without a mapped mate is a
   duplicate when a read with a mate has the same single key; among such reads only,
   the highest score stays, ties going to the first in the file. A group is decided
   once the reads still to come cannot join it: they start at least the largest
   leading clip of the file (found in pass 1) before the last position read. So the
   GPU holds a batch plus about a read length of reads, and the flags go to host
   memory (1 byte per record).
3. Writing. The duplicate flag is set in the raw records, which are compressed again
   (BGZF, nvCOMP) behind the input header with a ``@PG`` line.

GPU memory does not grow with the file: it fits an 8 GB GPU.

Differences from samtools: read names are compared on their first 32 bytes (ties
between longer names that share them are counted in the statistics); barcodes and
read names are matched by 64-bit hashes; samtools compares reads within a window of
the maximum read length, which only matters for reads with the same unclipped
coordinate but alignments more than that apart; QC-failed reads are not examined
(the samtools default) and the QC-fail precedence of samtools is not applied.
Optical duplicates, ``-S`` (supplementary reads) and ``-r`` are not implemented.
"""

from __future__ import annotations

import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cubam.bam import (
    MATE_REVERSE,
    MATE_UNMAPPED,
    PAIRED,
    QC_FAIL,
    READ1,
    REVERSE,
    SECONDARY,
    SUPPLEMENTARY,
    UNMAPPED,
    BamHeader,
    read_bam,
)
from cubam.bgzf import BgzfWriter

# orientations and leftmost codes of samtools
O_FF, O_RR, O_FR, O_RF = 2, 3, 5, 7
R_LE, R_RI = 11, 13
MIN_QUALITY = 15
NAME_WORDS = 4  # read names compared on 32 bytes

_SOURCE = r"""
__device__ __forceinline__ unsigned int u16(const unsigned char* b, long long o) {
    return b[o] | (b[o + 1] << 8);
}
__device__ __forceinline__ unsigned int u32(const unsigned char* b, long long o) {
    return b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | ((unsigned int)b[o + 3] << 24);
}

// Per record: score (qualities >= 15), leading and trailing clips (S, H), reference length,
// and from the MC string: mate leading clips and mate reference length + trailing clips.
extern "C" __global__ void md_fields(const unsigned char* b, const long long* off, long long n,
                                     const long long* mc_off, const int* mc_len, int min_q,
                                     int* score, int* lclip, int* rclip, int* reflen,
                                     int* m_lclip, int* m_span) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long o = off[i];
    unsigned int lrn = b[o + 12], ncig = u16(b, o + 16);
    int lseq = (int)u32(b, o + 20);
    long long c = o + 36 + lrn;
    int lead = 0, trail = 0, rl = 0, seen = 0;
    for (unsigned int k = 0; k < ncig; k++) {
        unsigned int v = u32(b, c + 4LL * k);
        unsigned int op = v & 15, len = v >> 4;
        bool clip = (op == 4 || op == 5);
        if (!seen) { if (clip) { lead += len; continue; } seen = 1; }
        if (op == 0 || op == 2 || op == 3 || op == 7 || op == 8) rl += len;
    }
    for (int k = (int)ncig - 1; k >= 0; k--) {
        unsigned int v = u32(b, c + 4LL * k);
        unsigned int op = v & 15;
        if (op == 4 || op == 5) trail += v >> 4; else break;
    }
    long long q = c + 4LL * ncig + (lseq + 1) / 2;
    int s = 0;
    if (lseq > 0 && b[q] != 0xff) {
        for (int k = 0; k < lseq; k++) { int x = b[q + k]; if (x >= min_q) s += x; }
    }
    score[i] = s; lclip[i] = lead; rclip[i] = trail; reflen[i] = rl;
    // mate CIGAR (samtools unclipped_other_start / unclipped_other_end)
    int ml = 0, ms = 0;
    int L = mc_len[i];
    if (L > 0) {
        long long p = mc_off[i], e = p + L;
        int skip = 1, lead_done = 0;
        while (p < e && b[p] != '*') {
            long num = 0; int digits = 0;
            while (p < e && b[p] >= '0' && b[p] <= '9') { num = num * 10 + (b[p] - '0'); p++; digits = 1; }
            if (!digits) num = 1;
            if (p >= e) break;
            unsigned char ch = b[p];
            if (!lead_done) { if (ch == 'S' || ch == 'H') ml += num; else lead_done = 1; }
            switch (ch) {
                case 'M': case 'D': case 'N': case '=': case 'X': ms += num; skip = 0; break;
                case 'S': case 'H': if (!skip) ms += num; break;
                default: break;
            }
            p++;
        }
    }
    m_lclip[i] = ml; m_span[i] = ms;
}

// First 32 bytes of the read name as 4 big-endian words (zero padded): word order = strcmp order.
extern "C" __global__ void md_name_words(const unsigned char* b, const long long* off, long long n,
                                         unsigned long long* words, int* long_name) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long o = off[i];
    int len = (int)b[o + 12] - 1;
    for (int w = 0; w < 4; w++) {
        unsigned long long x = 0;
        for (int k = 0; k < 8; k++) {
            int j = w * 8 + k;
            x = (x << 8) | (j < len ? b[o + 36 + j] : 0);
        }
        words[i * 4 + w] = x;
    }
    long_name[i] = len > 32;
}

// Set the duplicate flag (0x400: bit 2 of the high flag byte) of the marked records.
extern "C" __global__ void md_set_dup(unsigned char* b, const long long* off, long long n, const unsigned char* dup) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (dup[i]) b[off[i] + 19] |= 0x04;
}
"""

_module = None


def _launch(name: str, n: int, *args) -> None:
    import cupy as cp

    global _module
    if _module is None:
        _module = cp.RawModule(code=_SOURCE, options=("-std=c++14",))
    if n:
        _module.get_function(name)(((n + 255) // 256,), (256,), args)


@dataclass
class MarkdupResult:
    """Counts as in the ``samtools markdup -s`` report, the run time, and the time of each pass."""

    stats: dict = field(default_factory=dict)
    seconds: float = 0.0
    steps: dict = field(default_factory=dict)


def _fields(batch, barcode_tag: str | None):
    """Columns of the records of a batch needed for the keys."""
    import cupy as cp

    n = len(batch)
    mc = batch.tag("MC")
    is_str = mc.type == ord("Z")
    mc_len = cp.where(is_str, mc.length, 0).astype(cp.int32)
    out = {k: cp.empty(n, cp.int32) for k in ("score", "lclip", "rclip", "reflen", "m_lclip", "m_span")}
    _launch(
        "md_fields",
        n,
        batch.data,
        batch.offset,
        np.int64(n),
        mc.offset,
        mc_len,
        np.int32(MIN_QUALITY),
        out["score"],
        out["lclip"],
        out["rclip"],
        out["reflen"],
        out["m_lclip"],
        out["m_span"],
    )
    out["has_mc"] = is_str
    if barcode_tag:
        out["barcode"] = batch.tag(barcode_tag).hash()
    else:
        out["barcode"] = cp.zeros(n, cp.uint64)
    return out


_NOT_PRIMARY = SECONDARY | SUPPLEMENTARY
_NOT_EXAMINED = SECONDARY | SUPPLEMENTARY | UNMAPPED | QC_FAIL
_PAIR_KEY = ("barcode", "tid", "pair_t", "pair_mtid", "pair_o", "pair_left", "pair_orient")
_SINGLE_KEY = ("barcode", "tid", "single_c", "single_orient")


def _coord_key(ref, pos):
    """(reference, position) as one int64 in file order, unplaced records (reference -1) last."""
    import cupy as cp

    r = ref.astype(cp.int64)
    r = cp.where(r < 0, np.int64(1 << 31), r)
    return (r << 32) | (pos.astype(cp.int64) + 1)


def _cat(a: dict | None, b: dict | None) -> dict | None:
    import cupy as cp

    if a is None or b is None:
        return b if a is None else a
    return {k: cp.concatenate([a[k], b[k]]) for k in a}


def _sum_by_name(rows: dict, last):
    """Per row: the sum of the scores of the rows with its name, and whether that sum is final.

    It is final when the name has all its primary reads (2 for a paired read, 1
    otherwise), or when the file is read past the position of the mate
    (``mate < last``) without finding it. ``last=None``: every sum is final.
    """
    import cupy as cp

    order = cp.argsort(rows["h"])
    hs = rows["h"][order]
    s = rows["s"][order]
    start = cp.ones(hs.size, cp.bool_)
    start[1:] = hs[1:] != hs[:-1]
    gid = cp.cumsum(start) - 1
    first = cp.flatnonzero(start)
    end = cp.concatenate([first[1:], cp.asarray([hs.size], first.dtype)]) - 1
    cs = cp.cumsum(s)
    total = cp.empty_like(s)
    total[order] = (cs[end] - cs[first] + s[first])[gid]
    if last is None:
        return total, cp.ones(hs.size, cp.bool_)
    count = cp.empty(hs.size, cp.int64)
    count[order] = (end - first + 1)[gid]
    done = (count >= rows["need"]) | (rows["mate"] < last)
    return total, done


def _pair_scores(path, chunk_bytes):
    """Pair score of every record (numpy int32, in file order), the number of records, and the
    largest leading clip of the examined reads.

    The pair score of a primary read is the sum of the scores of the primary
    reads of its name: its own plus that of its mate. Each score waits on the
    GPU until the mate is read (or the file is read past the mate's position),
    so only pairs whose mate is still to come are held: mostly mates on later
    references. Other records get 0.
    """
    import cupy as cp

    parts: list[np.ndarray] = []
    late: list[tuple] = []  # (ordinals, scores) of the reads whose mate came in a later batch
    held = None
    max_lclip = 0
    ordinal0 = 0
    for batch in read_bam(path, chunk_bytes):
        n = len(batch)
        out = cp.zeros(n, cp.int32)
        idx = cp.flatnonzero(batch.has_flags(none_of=_NOT_PRIMARY))
        cur = None
        if idx.size:
            prim = batch.take(idx)
            f = _fields(prim, None)
            ex = prim.has_flags(none_of=UNMAPPED | QC_FAIL)
            if bool(ex.any()):
                max_lclip = max(max_lclip, int(f["lclip"][ex].max()))
            cur = {
                "h": prim.name_hash(),
                "s": f["score"].astype(cp.int64),
                "o": idx.astype(cp.int64) + ordinal0,
                "need": cp.where((prim.flag.astype(cp.int32) & PAIRED) != 0, 2, 1).astype(cp.int8),
                "mate": _coord_key(prim.nref, prim.npos),
            }
        rows = _cat(held, cur)
        if rows is not None:
            total, done = _sum_by_name(rows, _coord_key(batch.ref[-1:], batch.pos[-1:]))
            o, s = rows["o"][done], total[done].astype(cp.int32)
            here = o >= ordinal0
            out[o[here] - ordinal0] = s[here]
            if not bool(here.all()):
                late.append((o[~here].get(), s[~here].get()))
            keep = cp.flatnonzero(~done)
            held = {k: v[keep] for k, v in rows.items()} if keep.size else None
        parts.append(out.get())
        ordinal0 += n
    if held is not None:  # mates missing from the file: the sum of what was found
        total, _ = _sum_by_name(held, None)
        late.append((held["o"].get(), total.astype(cp.int32).get()))
    scores = np.concatenate(parts) if parts else np.zeros(0, np.int32)
    for o, s in late:
        scores[o] = s
    return scores, ordinal0, max_lclip


def _keys(batch, ordinal0: int, scores, barcode_tag):
    """Key columns of the examined reads of a batch (a dict of cupy arrays), and the
    (reference, position) of the last one; ``None`` when there is none."""
    import cupy as cp

    ex = batch.has_flags(none_of=_NOT_EXAMINED)
    idx = cp.flatnonzero(ex)
    sub = batch.take(idx)
    n = len(sub)
    if n == 0:
        return None
    f = _fields(sub, barcode_tag)
    flag = sub.flag.astype(cp.int32)
    rev = (flag & REVERSE) != 0
    mrev = (flag & MATE_REVERSE) != 0
    r1 = (flag & READ1) != 0
    tid = sub.ref.astype(cp.int64)
    mtid = sub.nref.astype(cp.int64)
    pos = sub.pos.astype(cp.int64)
    mpos = sub.npos.astype(cp.int64)
    has_mate = ((flag & PAIRED) != 0) & ((flag & MATE_UNMAPPED) == 0) & ~((mtid == -1) & (mpos == -1))
    if bool((has_mate & ~f["has_mc"]).any()):
        raise ValueError("a paired read has no MC tag (run samtools fixmate, or align with BWA-MEM >= 0.7.16)")
    # samtools coordinates are 1-based; bam_endpos = pos + reference length (pos + 1 when 0)
    endpos = pos + cp.maximum(f["reflen"], 1).astype(cp.int64)
    this_coord = pos - f["lclip"] + 1
    this_end = endpos + f["rclip"]
    other_coord = mpos - f["m_lclip"] + 1
    other_end = mpos + f["m_span"]

    same = rev == mrev
    lm_same = cp.where(~rev, this_coord <= other_coord, this_end <= other_end)
    lm_diff = cp.where(rev, this_end <= other_coord, this_coord <= other_end)
    leftmost = cp.where(tid != mtid, tid < mtid, cp.where(same, lm_same, lm_diff))

    # pair key coordinates and orientation (samtools make_pair_key, template mode)
    t = this_coord.copy()
    o = other_coord.copy()
    orient = cp.zeros(n, cp.int8)
    a = leftmost & same
    o = cp.where(a, other_end, o)
    orient = cp.where(a, cp.where(~rev, cp.where(r1, O_FF, O_RR), cp.where(r1, O_RR, O_FF)), orient)
    a = leftmost & ~same
    o = cp.where(a & ~rev, other_end, o)
    t = cp.where(a & rev, this_end, t)
    orient = cp.where(a, cp.where(~rev, O_FR, O_RF), orient)
    a = ~leftmost & same
    t = cp.where(a, this_end, t)
    orient = cp.where(a, cp.where(~rev, cp.where(r1, O_RR, O_FF), cp.where(r1, O_FF, O_RR)), orient)
    a = ~leftmost & ~same
    o = cp.where(a & ~rev, other_end, o)
    t = cp.where(a & rev, this_end, t)
    orient = cp.where(a, cp.where(~rev, O_RF, O_FR), orient)
    left_read = cp.where(leftmost, R_LE, R_RI).astype(cp.int8)

    words = cp.empty(n * 4, cp.uint64)
    long_name = cp.empty(n, cp.int32)
    _launch("md_name_words", n, sub.data, sub.offset, np.int64(n), words, long_name)

    cols = {
        "ordinal": idx.astype(cp.int64) + ordinal0,
        "tid": tid.astype(cp.int32),
        "barcode": f["barcode"],
        "has_mate": has_mate,
        "pair_t": t,
        "pair_mtid": mtid.astype(cp.int32),
        "pair_o": o,
        "pair_left": left_read,
        "pair_orient": orient,
        "single_c": cp.where(rev, this_end, this_coord),
        "single_orient": cp.where(rev, O_RR, O_FF).astype(cp.int8),
        "score": f["score"].astype(cp.int64),
        "pair_score": scores[idx].astype(cp.int64),  # read + mate (pass 1)
        "w0": words[0::4].copy(),
        "w1": words[1::4].copy(),
        "w2": words[2::4].copy(),
        "w3": words[3::4].copy(),
        "long_name": long_name.astype(cp.bool_),
    }
    return cols, (int(tid[-1]), int(pos[-1]))


def _group_ids(cols: list):
    """Group number of each row for the key columns (cudf hash groupby)."""
    import cudf

    df = cudf.DataFrame({f"k{i}": c for i, c in enumerate(cols)})
    return df.groupby(list(df.columns), sort=False).ngroup().values


def _first_by(group, order_cols: list):
    """Rows sorted by group then ``order_cols`` (ascending), and whether each is the first of its group."""
    import cudf
    import cupy as cp

    cols = {"g": group}
    cols.update({f"o{i}": c for i, c in enumerate(order_cols)})
    cols["row"] = cp.arange(group.size, dtype=cp.int64)
    df = cudf.DataFrame(cols).sort_values([k for k in cols if k != "row"])
    order = df["row"].values
    g = df["g"].values
    first = cp.ones(g.size, cp.bool_)
    first[1:] = g[1:] != g[:-1]
    return order, first


def _pair_duplicates(t: dict, stats: dict):
    """Ordinals of the duplicates among reads with a mapped mate whose key groups are complete.

    The highest pair score stays, then the smallest name, then the first in the file.
    """
    import cupy as cp

    n = t["ordinal"].size
    g = _group_ids([t[c] for c in _PAIR_KEY])
    neg = -t["pair_score"]
    order, first = _first_by(g, [neg, t["w0"], t["w1"], t["w2"], t["w3"], t["ordinal"]])
    d = cp.zeros(n, cp.bool_)
    d[order[~first]] = True
    # ties on the 32-byte name prefix between long names (the order of samtools may differ)
    if n > 1:
        sg = g[order]
        eq = (sg[1:] == sg[:-1]) & (neg[order][1:] == neg[order][:-1])
        for w in ("w0", "w1", "w2", "w3"):
            ww = t[w][order]
            eq &= ww[1:] == ww[:-1]
        stats["NAME PREFIX TIES"] += int((eq & t["long_name"][order][1:]).sum())
    stats["DUPLICATE PAIR"] += int(d.sum())
    return t["ordinal"][d]


def _single_duplicates(t: dict, stats: dict):
    """Ordinals of the duplicates among reads without a mapped mate whose key groups are complete.

    ``t`` holds every examined read (``has_mate`` tells which have one). A
    read without a mapped mate is a duplicate when a read with one has its
    single key; among reads without one only, the highest score stays, then
    the first in the file.
    """
    import cupy as cp

    pm = t["has_mate"]
    si = cp.flatnonzero(~pm)
    if si.size == 0:
        return cp.zeros(0, cp.int64)
    g = _group_ids([t[c] for c in _SINGLE_KEY])
    has_pair = cp.zeros(int(g.max()) + 1, cp.bool_)
    has_pair[g[pm]] = True
    gs = g[si]
    d = has_pair[gs]
    alone = cp.flatnonzero(~d)
    if alone.size:
        order, first = _first_by(gs[alone], [-t["score"][si][alone], t["ordinal"][si][alone]])
        da = cp.zeros(alone.size, cp.bool_)
        da[order[~first]] = True
        d[alone[da]] = True
    stats["DUPLICATE SINGLE"] += int(d.sum())
    return t["ordinal"][si[d]]


class _Window:
    """Rows waiting until no read still to come can share their key.

    The file is sorted by (reference, position), and a read at position ``p``
    has an unclipped 5' coordinate of at least ``p + 1 - max_lclip``. Once the
    reads up to (``tid``, ``p``) are in, the rows of earlier references, and
    those whose key coordinate (column ``coord``) is below ``p + 1 - max_lclip``,
    have their whole group: ``decide`` then marks them.
    """

    def __init__(self, decide, columns: tuple, coord: str, max_lclip: int):
        self.decide, self.columns, self.coord, self.max_lclip = decide, columns, coord, max_lclip
        self.rows = None

    def add(self, rows: dict | None, last: tuple | None, stats: dict, dup: np.ndarray) -> None:
        """Add ``rows``, the examined reads up to ``last`` (reference, position) being read;
        ``last=None`` at the end of the file."""
        import cupy as cp

        if rows is not None:
            rows = {k: rows[k] for k in self.columns}
        rows = _cat(self.rows, rows)
        if rows is None:
            return
        if last is None:
            final = cp.ones(rows["tid"].size, cp.bool_)
        else:
            tid, pos = last
            final = (rows["tid"] < tid) | (rows[self.coord] < pos + 1 - self.max_lclip)
        idx = cp.flatnonzero(final)
        if idx.size == final.size:
            done, self.rows = rows, None
        else:
            keep = cp.flatnonzero(~final)
            done = {k: v[idx] for k, v in rows.items()} if idx.size else None
            self.rows = {k: v[keep] for k, v in rows.items()}
        if done is not None:
            dup[self.decide(done, stats).get()] = True


_PAIR_COLUMNS = _PAIR_KEY + ("ordinal", "pair_score", "w0", "w1", "w2", "w3", "long_name")
_SINGLE_COLUMNS = _SINGLE_KEY + ("ordinal", "has_mate", "score")

_STATS = (
    "READ",
    "EXAMINED",
    "PAIRED",
    "SINGLE",
    "DUPLICATE PAIR",
    "DUPLICATE SINGLE",
    "DUPLICATE TOTAL",
    "NAME PREFIX TIES",
)


def _duplicate_mask(path, barcode_tag, chunk_bytes, stats: dict, seconds: dict) -> np.ndarray:
    """Duplicate flag of every record of ``path`` (numpy bool, in file order), in an RMM pool."""
    import cupy as cp

    t0 = time.perf_counter()
    scores, n_records, max_lclip = _pair_scores(path, chunk_bytes)
    seconds["pair_scores"] = time.perf_counter() - t0
    stats["READ"] = n_records

    t0 = time.perf_counter()
    dup = np.zeros(n_records, np.bool_)
    pairs = _Window(_pair_duplicates, _PAIR_COLUMNS, "pair_t", max_lclip)
    singles = _Window(_single_duplicates, _SINGLE_COLUMNS, "single_c", max_lclip)
    ordinal0 = 0
    for batch in read_bam(path, chunk_bytes):
        n = len(batch)
        k = _keys(batch, ordinal0, cp.asarray(scores[ordinal0 : ordinal0 + n]), barcode_tag)
        ordinal0 += n
        if k is None:
            continue
        cols, last = k
        pm = cols["has_mate"]
        n_pair = int(pm.sum())
        stats["EXAMINED"] += int(pm.size)
        stats["PAIRED"] += n_pair
        stats["SINGLE"] += int(pm.size) - n_pair
        pi = cp.flatnonzero(pm)
        pairs.add({c: cols[c][pi] for c in _PAIR_COLUMNS} if pi.size else None, last, stats, dup)
        singles.add(cols, last, stats, dup)
    pairs.add(None, None, stats, dup)
    singles.add(None, None, stats, dup)
    seconds["duplicates"] = time.perf_counter() - t0
    stats["DUPLICATE TOTAL"] = stats["DUPLICATE PAIR"] + stats["DUPLICATE SINGLE"]
    return dup


def duplicate_mask(
    path: str | Path, barcode_tag: str | None = None, chunk_bytes: int | None = None
) -> tuple[np.ndarray, MarkdupResult]:
    """Which records of a coordinate-sorted BAM file are duplicates, without writing a file.

    Returns a boolean array with one value per record, in file order, and the
    counts. The duplicate flags already in the file are ignored, so the result
    can be compared with them (for example on a file marked by samtools).
    """
    from cubam._memory import auto_chunk_bytes, memory_pool

    t0 = time.perf_counter()
    BamHeader.read(path).require_coordinate_sorted(path)
    chunk_bytes = chunk_bytes or auto_chunk_bytes()
    stats, steps = dict.fromkeys(_STATS, 0), {}
    with memory_pool():
        mask = _duplicate_mask(path, barcode_tag, chunk_bytes, stats, steps)
    return mask, MarkdupResult(stats, time.perf_counter() - t0, steps)


def mark_duplicates(
    path: str | Path,
    output: str | Path,
    barcode_tag: str | None = None,
    chunk_bytes: int | None = None,
    level: int = 2,
    command_line: str | None = None,
) -> MarkdupResult:
    """Mark the duplicates of a coordinate-sorted BAM file into ``output`` (see the module documentation).

    ``barcode_tag``: duplicates must share this tag (``CB`` for 10x data), as
    ``samtools markdup --barcode-tag``. ``level``: compression level of nvCOMP Deflate.
    ``chunk_bytes``: compressed bytes read at a time (default: :func:`cubam.auto_chunk_bytes`).
    The duplicate flags already in the input are kept.
    """
    import cupy as cp

    from cubam._memory import auto_chunk_bytes, memory_pool

    t0 = time.perf_counter()
    output = Path(output)
    if output.exists() and output.samefile(path):
        raise ValueError(f"Output path refers to the input BAM: {output}")
    header = BamHeader.read(path)
    header.require_coordinate_sorted(path)
    chunk_bytes = chunk_bytes or auto_chunk_bytes()
    stats, steps = dict.fromkeys(_STATS, 0), {}
    with memory_pool():
        dup = _duplicate_mask(path, barcode_tag, chunk_bytes, stats, steps)

        t1 = time.perf_counter()
        cl = command_line or shlex.join([Path(sys.argv[0]).name or "python", *sys.argv[1:]])
        bc = f" on {barcode_tag}" if barcode_tag else ""
        text = header.text
        if text and not text.endswith("\n"):
            text += "\n"
        text += f"@PG\tID:cubam.markdup\tPN:cubam\tCL:{cl}\tDS:duplicates as samtools markdup{bc}\n"
        ordinal0 = 0
        with BgzfWriter(output, level=level) as out:
            out.write(header.to_bytes(text))
            for batch in read_bam(path, chunk_bytes):
                n = len(batch)
                d = cp.asarray(dup[ordinal0 : ordinal0 + n].view(np.uint8))
                _launch("md_set_dup", n, batch.data, batch.offset, np.int64(n), d)
                ordinal0 += n
                end = int(batch.offset[-1] + batch.record_sizes()[-1])
                out.write(batch.data[int(batch.offset[0]) : end])
        steps["write"] = time.perf_counter() - t1
    return MarkdupResult(stats, time.perf_counter() - t0, steps)
