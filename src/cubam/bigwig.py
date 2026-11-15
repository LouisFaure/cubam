"""
bigWig files written on the GPU.

A bigWig file (UCSC, version 4) holds the runs of a track in compressed
sections, an R-tree over the sections, and zoom levels: summaries (bases
covered, min, max, sum, sum of squares) over bins of increasing size, each
with its own sections and R-tree. :func:`write_bigwig` builds the sections
and the zoom summaries on the GPU and compresses them with the Deflate
compressor of nvCOMP (zlib streams, with Adler-32 from a kernel). The trees,
a few kilobytes, are built on the host.

The layout follows libBigWig (pyBigWig): header, zoom headers, total
summary, contig B+ tree (one leaf), data (an 8-byte section count, then the
sections), R-tree, then for each zoom level its data (a 4-byte section
count, then the sections) and its R-tree.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from cubam._kernels import copy_spans, launch
from cubam.bgzf import _adler32_spans, _deflate_spans

_BW_MAGIC = 0x888FFC26
_BPT_MAGIC = 0x78CA8C91
_CIR_MAGIC = 0x2468ACE0
ITEMS_PER_SLOT = 1024
_RTREE_BLOCK = 256
_ZOOM_STEP = 4


def _sections(ref):
    """Sections of at most ITEMS_PER_SLOT items within a contig: (section of each item, first item, count)."""
    import cupy as cp

    n = int(ref.size)
    i = cp.arange(n, dtype=cp.int64)
    first_of_ref = cp.searchsorted(ref, ref, side="left")
    new = ((i - first_of_ref) % ITEMS_PER_SLOT) == 0
    sec = cp.cumsum(new) - 1
    first = cp.flatnonzero(new)
    count = cp.diff(cp.concatenate([first, cp.asarray([n])]))
    return sec, first, count


def _zlib(buf, off, size, level: int):
    """zlib streams (header, Deflate, Adler-32) of the spans of ``buf``: (bytes on the host, offsets, sizes)."""
    import cupy as cp

    raw, raw_off, raw_size = _deflate_spans(buf, off, size, level)
    adler = _adler32_spans(buf, off, size)
    zsize = raw_size + 6
    zoff = cp.cumsum(zsize) - zsize
    out = cp.empty(int(zoff[-1] + zsize[-1]), cp.uint8)
    out[zoff] = 0x78
    out[zoff + 1] = 0x9C
    copy_spans(raw, out, raw_off, zoff + 2, raw_size)
    end = zoff + zsize
    for k in range(4):  # Adler-32, big-endian
        out[end - 4 + k] = ((adler >> (8 * (3 - k))) & 0xFF).astype(cp.uint8)
    return out.get(), zoff.get(), zsize.get()


def _rtree(sec_ref, sec_start, sec_end, offsets, sizes, data_end: int, at: int) -> bytes:
    """A cirTree index over sections (one leaf item per section), to be written at file offset ``at``."""
    n = len(offsets)
    b = _RTREE_BLOCK
    header = struct.pack(
        "<IIQIIIIQII",
        _CIR_MAGIC,
        b,
        n,
        int(sec_ref[0]) if n else 0,
        int(sec_start[0]) if n else 0,
        int(sec_ref[-1]) if n else 0,
        int(sec_end[-1]) if n else 0,
        data_end,
        ITEMS_PER_SLOT,
        0,
    )
    # levels from the leaves up: bounds of each node as (start ref, start, end ref, end)
    leaf_bounds = list(zip(sec_ref, sec_start, sec_ref, sec_end, strict=True))
    levels = [leaf_bounds]
    while len(levels[-1]) > b:
        child = levels[-1]
        parent = []
        for k in range(0, len(child), b):
            g = child[k : k + b]
            parent.append((g[0][0], g[0][1], g[-1][2], g[-1][3]))
        levels.append(parent)
    # nodes per level (top first), each padded to b items, so offsets follow from sizes
    tops = levels[::-1]
    n_nodes = [max(1, -(-len(lv) // b)) for lv in tops]  # nodes holding the items of each level
    leaf_level = len(tops) - 1
    node_size = [4 + b * (32 if d == leaf_level else 24) for d in range(len(tops))]
    level_off = []
    pos = at + len(header)
    for d in range(len(tops)):
        level_off.append(pos)
        pos += n_nodes[d] * node_size[d]
    out = [header]
    for d, items in enumerate(tops):
        leaf = d == leaf_level
        for node in range(n_nodes[d]):
            chunk = items[node * b : (node + 1) * b]
            parts = [struct.pack("<BBH", 1 if leaf else 0, 0, len(chunk))]
            for j, (r0, s0, r1, e1) in enumerate(chunk):
                k = node * b + j
                if leaf:
                    parts.append(struct.pack("<IIIIQQ", r0, s0, r1, e1, int(offsets[k]), int(sizes[k])))
                else:  # child k of the next level down
                    parts.append(struct.pack("<IIIIQ", r0, s0, r1, e1, level_off[d + 1] + k * node_size[d + 1]))
            body = b"".join(parts)
            out.append(body + b"\0" * (node_size[d] - len(body)))
    return b"".join(out)


def _chrom_tree(names: list[str], lengths: list[int]) -> bytes:
    """A B+ tree of the contigs with a single leaf, keys sorted."""
    key_size = max((len(n.encode()) for n in names), default=1)
    order = sorted(range(len(names)), key=lambda i: names[i].encode())
    n = len(names)
    out = [struct.pack("<IIIIQQ", _BPT_MAGIC, max(n, 1), key_size, 8, n, 0), struct.pack("<BBH", 1, 0, n)]
    for i in order:
        out.append(names[i].encode().ljust(key_size, b"\0") + struct.pack("<II", i, lengths[i]))
    return b"".join(out)


def _zoom(ref, start, end, value, reduction: int, ref_len):
    """Zoom records over bins of ``reduction`` bases, from runs: dict of cupy columns."""
    import cudf
    import cupy as cp

    s, e = start.astype(cp.int64), end.astype(cp.int64)
    b0, b1 = s // reduction, (e - 1) // reduction
    nb = b1 - b0 + 1
    ends = cp.cumsum(nb)
    j = cp.arange(int(ends[-1]), dtype=cp.int64)
    rec = cp.searchsorted(ends, j, side="right")
    b = b0[rec] + (j - (ends[rec] - nb[rec]))
    w = (cp.minimum(e[rec], (b + 1) * reduction) - cp.maximum(s[rec], b * reduction)).astype(cp.float64)
    v = value[rec].astype(cp.float64)
    df = cudf.DataFrame({"ref": ref[rec], "bin": b, "n": w, "mn": v, "mx": v, "sum": v * w, "ssq": v * v * w})
    return (
        df.groupby(["ref", "bin"], sort=True)
        .agg({"n": "sum", "mn": "min", "mx": "max", "sum": "sum", "ssq": "sum"})
        .reset_index()
    )


def _coarser(z, factor: int):
    """Zoom records of bins ``factor`` times larger, from finer ones."""
    z = z.assign(bin=z["bin"] // factor)
    return (
        z.groupby(["ref", "bin"], sort=True)
        .agg({"n": "sum", "mn": "min", "mx": "max", "sum": "sum", "ssq": "sum"})
        .reset_index()
    )


def write_bigwig(
    path: str | Path,
    names: list[str],
    lengths: list[int],
    ref,
    start,
    end,
    value,
    *,
    level: int = 2,
    max_zooms: int = 10,
) -> None:
    """Write runs ``[start, end)`` with ``value`` on contig ``ref`` (indices into ``names``) as a bigWig.

    The runs (cupy or numpy arrays) must be sorted by contig and start, and
    must not overlap. All contigs of ``names`` are listed in the file.
    ``level`` is the nvCOMP Deflate level, 0 (entropy coding only) to 5.
    """
    import cupy as cp

    ref = cp.asarray(ref, cp.int32)
    start = cp.asarray(start, cp.int32)
    end = cp.asarray(end, cp.int32)
    value = cp.asarray(value, cp.float32)
    n = int(ref.size)
    ref_len = cp.asarray(lengths, cp.int64)

    # Total summary, in float64 as libBigWig
    w = (end - start).astype(cp.float64)
    v = value.astype(cp.float64)
    if n:
        covered, vmin, vmax, vsum, vssq = (
            float(x) for x in cp.stack([w.sum(), v.min(), v.max(), (v * w).sum(), (v * v * w).sum()]).get()
        )
    else:
        covered, vmin, vmax, vsum, vssq = 0.0, 0.0, 0.0, 0.0, 0.0
    total = struct.pack("<Qdddd", int(covered), vmin, vmax, vsum, vssq)

    # Data sections
    if n:
        sec, first, count = _sections(ref)
        sec_len = 24 + 12 * count
        sec_off = cp.cumsum(sec_len) - sec_len
        buf = cp.empty(int(sec_off[-1] + sec_len[-1]), cp.uint8)
        launch("bw_data_sections", n, ref, start, end, value, sec, first, count, sec_off, np.int64(n), buf)
        data_blocks, data_off, data_size = _zlib(buf, sec_off, sec_len, level)
        last = first + count - 1
        data_ref, data_start, data_end = ref[first].get(), start[first].get(), end[last].get()
        del buf
    else:
        data_blocks, data_off, data_size = np.empty(0, np.uint8), np.empty(0, np.int64), np.empty(0, np.int64)
        data_ref = data_start = data_end = np.empty(0, np.int64)

    # Zoom levels: bins of 10 times the mean run, then 4 times larger each
    zooms = []
    if n and max_zooms > 0:
        reduction = max(int(10 * covered / n), 1)
        z = _zoom(ref, start, end, value, reduction, ref_len)
        longest = max(lengths)
        while len(zooms) < max_zooms and reduction <= longest:
            zooms.append((reduction, z))
            if len(z) <= 1:
                break
            reduction *= _ZOOM_STEP
            z = _coarser(z, _ZOOM_STEP)
    zoom_parts = []
    for reduction, z in zooms:
        zr = z["ref"].values.astype(cp.int32)
        zs = (z["bin"].values * reduction).astype(cp.int64)
        ze = cp.minimum(zs + reduction, ref_len[zr])
        m = int(zr.size)
        sec, first, count = _sections(zr)
        sec_len = 32 * count
        sec_off = cp.cumsum(sec_len) - sec_len
        buf = cp.empty(int(sec_off[-1] + sec_len[-1]), cp.uint8)
        launch(
            "bw_zoom_records",
            m,
            zr,
            zs.astype(cp.int32),
            ze.astype(cp.int32),
            z["n"].values.astype(cp.uint32),
            z["mn"].values.astype(cp.float32),
            z["mx"].values.astype(cp.float32),
            z["sum"].values.astype(cp.float32),
            z["ssq"].values.astype(cp.float32),
            sec,
            first,
            sec_off,
            np.int64(m),
            buf,
        )
        blocks, off, size = _zlib(buf, sec_off, sec_len, level)
        last = first + count - 1
        zoom_parts.append((reduction, blocks, off, size, zr[first].get(), zs[first].get(), ze[last].get()))
        del buf

    # Layout
    nz = len(zoom_parts)
    chrom_tree = _chrom_tree(names, lengths)
    total_off = 64 + 24 * nz
    tree_off = total_off + len(total)
    data_at = tree_off + len(chrom_tree)
    data_payload = data_at + 8
    data_end_off = data_payload + int(data_blocks.size)
    index = _rtree(data_ref, data_start, data_end, data_off + data_payload, data_size, data_end_off, data_end_off)
    pos = data_end_off + len(index)
    zoom_headers, zoom_bytes = [], []
    for reduction, blocks, off, size, zr0, zs0, ze0 in zoom_parts:
        zdata = pos
        payload = zdata + 4
        zend = payload + int(blocks.size)
        zindex = _rtree(zr0, zs0, ze0, off + payload, size, zend, zend)
        zoom_headers.append(struct.pack("<IIQQ", reduction, 0, zdata, zend))
        zoom_bytes += [struct.pack("<I", len(off)), blocks.tobytes(), zindex]
        pos = zend + len(zindex)
    header = struct.pack(
        "<IHHQQQHHQQIQ",
        _BW_MAGIC,
        4,
        nz,
        tree_off,
        data_at,
        data_end_off,
        0,
        0,
        0,
        total_off,
        max(24 + 12 * ITEMS_PER_SLOT, 32 * ITEMS_PER_SLOT),
        0,
    )
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(b"".join(zoom_headers))
        fh.write(total)
        fh.write(chrom_tree)
        fh.write(struct.pack("<Q", len(data_off)))
        fh.write(data_blocks.tobytes())
        fh.write(index)
        for part in zoom_bytes:
            fh.write(part)
