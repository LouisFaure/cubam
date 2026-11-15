"""
Duplicate marking of a coordinate-sorted BAM file on the GPU, as ``samtools markdup``.

Gives the duplicate flags of ``samtools fixmate -m`` followed by ``samtools markdup``
(template mode, 1.22), optionally with ``--barcode-tag`` so that duplicates must share
a cell barcode (10x scATAC). The input needs the ``MC`` tag (BWA-MEM and Parabricks
fq2bam write it) but no ``ms`` tag: the mate score is computed here.

Three passes over the file:

1. Pair scores. The score of a read is the sum of its base qualities of at least 15.
   The scores of the primary reads are summed per read name, which gives the score of
   a read plus that of its mate (``calc_score + ms`` in samtools).
2. Duplicates, one reference at a time (duplicates never cross references). Every
   examined read (primary, mapped, not QC-failed) gets the pair key of samtools
   (barcode, unclipped 5' coordinates of the read and of its mate from ``MC``,
   orientation, leftmost) when it has a mapped mate, and the single key (barcode,
   unclipped 5' coordinate, strand) in any case. In a group of reads with the same
   pair key the read with the highest pair score stays, ties going to the smallest
   read name; the others are duplicates. A read without a mapped mate is a duplicate
   when a read with a mate has the same single key; among such reads only, the
   highest score stays, ties going to the first in the file.
3. Writing. The duplicate flag is set in the raw records, which are compressed again
   (BGZF, nvCOMP) behind the input header with a ``@PG`` line.

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
    """Counts as in the ``samtools markdup -s`` report, and timings."""

    stats: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)


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


def _pair_scores(path, chunk_bytes):
    """Sorted unique read-name hashes of primary reads and the sum of their scores (read + mate)."""
    import cudf
    import cupy as cp

    hashes, scores = [], []
    n_records = 0
    for batch in read_bam(path, chunk_bytes):
        n_records += len(batch)
        prim = batch.take(batch.has_flags(none_of=SECONDARY | SUPPLEMENTARY))
        if len(prim) == 0:
            continue
        f = _fields(prim, None)
        df = cudf.DataFrame({"h": prim.name_hash(), "s": f["score"].astype(cp.int64)})
        # reduce inside the batch first: most mates are in the same batch
        red = df.groupby("h", sort=False).s.sum()
        hashes.append(red.index.values)
        scores.append(red.values)
    df = cudf.DataFrame({"h": cp.concatenate(hashes), "s": cp.concatenate(scores)})
    del hashes, scores
    red = df.groupby("h", sort=True).s.sum()
    del df
    uh, us = red.index.values, red.values.astype(cp.int64)
    return uh, us, n_records


def _keys(batch, ordinal0: int, uh, us, barcode_tag):
    """Key columns of the examined reads of a batch (a dict of cupy arrays)."""
    import cupy as cp

    ex = batch.has_flags(none_of=SECONDARY | SUPPLEMENTARY | UNMAPPED | QC_FAIL)
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

    # pair score: read + mate (sum over the primary reads of the name)
    h = sub.name_hash()
    k = cp.minimum(cp.searchsorted(uh, h), uh.size - 1)
    pair_score = cp.where(uh[k] == h, us[k], f["score"].astype(cp.int64))

    words = cp.empty(n * 4, cp.uint64)
    long_name = cp.empty(n, cp.int32)
    _launch("md_name_words", n, sub.data, sub.offset, np.int64(n), words, long_name)

    return {
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
        "pair_score": pair_score,
        "w0": words[0::4].copy(),
        "w1": words[1::4].copy(),
        "w2": words[2::4].copy(),
        "w3": words[3::4].copy(),
        "long_name": long_name.astype(cp.bool_),
    }


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


def _duplicates(t: dict, stats: dict):
    """Ordinals of the duplicates among the examined reads of one reference."""
    import cupy as cp

    n = t["ordinal"].size
    dup = cp.zeros(n, cp.bool_)
    pm = t["has_mate"]
    # pairs: highest pair score, then smallest name (then first in the file) stays
    pi = cp.flatnonzero(pm)
    if pi.size:
        g = _group_ids([t[c][pi] for c in ("barcode", "pair_t", "pair_mtid", "pair_o", "pair_left", "pair_orient")])
        neg = -t["pair_score"][pi]
        order, first = _first_by(g, [neg, t["w0"][pi], t["w1"][pi], t["w2"][pi], t["w3"][pi], t["ordinal"][pi]])
        d = cp.zeros(pi.size, cp.bool_)
        d[order[~first]] = True
        dup[pi[d]] = True
        # ties on the 32-byte name prefix between long names (the order of samtools may differ)
        sg = g[order]
        same_prefix = cp.zeros(pi.size, cp.bool_)
        if pi.size > 1:
            eq = (sg[1:] == sg[:-1]) & (neg[order][1:] == neg[order][:-1])
            for w in ("w0", "w1", "w2", "w3"):
                ww = t[w][pi][order]
                eq &= ww[1:] == ww[:-1]
            same_prefix[1:] = eq & t["long_name"][pi][order][1:]
        stats["NAME PREFIX TIES"] += int(same_prefix.sum())
        stats["DUPLICATE PAIR"] += int(d.sum())
    # singles
    g = _group_ids([t[c] for c in ("barcode", "tid", "single_c", "single_orient")])
    ng = int(g.max()) + 1 if n else 0
    group_has_pair = cp.zeros(ng, cp.int32)
    group_has_pair[g[pm]] = 1
    si = cp.flatnonzero(~pm)
    if si.size:
        gs = g[si]
        hp = group_has_pair[gs] > 0
        d = hp.copy()
        alone = cp.flatnonzero(~hp)
        if alone.size:
            order, first = _first_by(gs[alone], [-t["score"][si][alone], t["ordinal"][si][alone]])
            da = cp.zeros(alone.size, cp.bool_)
            da[order[~first]] = True
            d[alone[da]] = True
        dup[si[d]] = True
        stats["DUPLICATE SINGLE"] += int(d.sum())
    stats["PAIRED"] += int(pm.sum())
    stats["SINGLE"] += int((~pm).sum())
    return t["ordinal"][dup]


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


def _duplicate_mask(path, barcode_tag, chunk_bytes, stats: dict, seconds: dict):
    """Duplicate flag of every record of ``path`` (cupy uint8, in file order), in an RMM pool."""
    import cupy as cp

    t0 = time.perf_counter()
    uh, us, n_records = _pair_scores(path, chunk_bytes)
    seconds["pair_scores"] = time.perf_counter() - t0
    stats["READ"] = n_records

    t0 = time.perf_counter()
    dup_mask = cp.zeros(n_records, cp.uint8)
    pending: list[dict] = []
    cur_tid = None
    ordinal0 = 0

    def finish(parts):
        if not parts:
            return
        t = {k: cp.concatenate([p[k] for p in parts]) for k in parts[0]}
        stats["EXAMINED"] += int(t["ordinal"].size)
        dup_mask[_duplicates(t, stats)] = 1

    for batch in read_bam(path, chunk_bytes):
        k = _keys(batch, ordinal0, uh, us, barcode_tag)
        ordinal0 += len(batch)
        if k is None:
            continue
        tids = k["tid"]
        # split the batch by reference; earlier references are complete
        bounds = cp.flatnonzero(tids[1:] != tids[:-1]) + 1
        starts = [0] + bounds.get().tolist()
        ends = starts[1:] + [int(tids.size)]
        for s, e in zip(starts, ends, strict=True):
            tid = int(tids[s])
            part = {c: v[s:e] for c, v in k.items()}
            if cur_tid is not None and tid != cur_tid:
                finish(pending)
                pending = []
            cur_tid = tid
            pending.append(part)
    finish(pending)
    seconds["duplicates"] = time.perf_counter() - t0
    stats["DUPLICATE TOTAL"] = stats["DUPLICATE PAIR"] + stats["DUPLICATE SINGLE"]
    return dup_mask


def duplicate_mask(
    path: str | Path, barcode_tag: str | None = None, chunk_bytes: int | None = None
) -> tuple[np.ndarray, MarkdupResult]:
    """Which records of a coordinate-sorted BAM file are duplicates, without writing a file.

    Returns a boolean array with one value per record, in file order, and the
    counts. The duplicate flags already in the file are ignored, so the result
    can be compared with them (for example on a file marked by samtools).
    """
    from cubam._memory import memory_pool

    BamHeader.read(path).require_coordinate_sorted(path)
    stats, seconds = dict.fromkeys(_STATS, 0), {}
    with memory_pool():
        mask = _duplicate_mask(path, barcode_tag, chunk_bytes, stats, seconds).get().astype(bool)
    return mask, MarkdupResult(stats=stats, seconds=seconds)


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
    The duplicate flags already in the input are kept.
    """
    from cubam._memory import memory_pool

    header = BamHeader.read(path)
    header.require_coordinate_sorted(path)
    stats, seconds = dict.fromkeys(_STATS, 0), {}
    with memory_pool():
        dup_mask = _duplicate_mask(path, barcode_tag, chunk_bytes, stats, seconds)

        t0 = time.perf_counter()
        cl = command_line or " ".join(shlex.quote(a) for a in sys.argv)
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
                _launch("md_set_dup", n, batch.data, batch.offset, np.int64(n), dup_mask[ordinal0 : ordinal0 + n])
                ordinal0 += n
                end = int(batch.offset[-1] + batch.record_sizes()[-1])
                out.write(batch.data[int(batch.offset[0]) : end])
        seconds["write"] = time.perf_counter() - t0
    return MarkdupResult(stats=stats, seconds=seconds)
