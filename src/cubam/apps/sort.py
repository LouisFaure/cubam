"""
A BAM file sorted by coordinate or by read name, in the order of ``samtools sort``, on the GPU.

The order is that of samtools 1.24:

- by coordinate (the default): reference id, unmapped reads (id -1) last,
  then position, then the forward strand before the reverse strand;
- by name (``samtools sort -n``): read names in natural order (``a7b``
  before ``a12b``), or in byte order with ``natural=False``
  (``samtools sort -N``), then READ1, READ2, supplementary and secondary
  records in this order.

Records that tie keep the order of the input, as in samtools, so the
output holds the same records in the same order.

The input is sorted in two passes. The first reads it in chunks and sorts
a run of records at a time on the GPU (8 GB of records on a GPU of 24 GB
or more); it writes each run to a temporary BAM file. When the input fits
in one run, the sorted records go straight to the output. Otherwise, the
second pass merges the runs: it reads each run a piece at a time, and
writes the records up to the last record read from any run.
The record bytes are copied as they are, and the output gets the header of
the input with its ``@HD`` line updated and a ``@PG`` line.
"""

from __future__ import annotations

import shlex
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cubam._kernels import launch
from cubam.bam import REVERSE, BamHeader, RecordBatch, _string_series, _strings_at, read_bam
from cubam.bgzf import BLOCK_DATA, BgzfWriter

#: Bytes of records gathered and compressed at a time
_PIECE = 4096 * BLOCK_DATA
#: nvCOMP Deflate level of the runs
_RUN_LEVEL = 2


@dataclass
class SortResult:
    """The sorted file, its number of records and the number of runs merged (1: no merge)."""

    path: Path
    records: int
    runs: int
    stats: dict = field(default_factory=dict)
    seconds: float = 0.0


class _Keys:
    """Sort keys of records, as columns of a cuDF frame, with where the records are (``src``, ``off``).

    In the merge, frames also have the run, the index in the run (``seq``)
    and the last record read from the run (``last``).
    """

    def __init__(self, by: str, natural: bool):
        self.by, self.natural = by, natural
        self.columns = ["k"] if by == "coordinate" else ["name", "fk"]
        self.order = [*self.columns, "run", "seq"]

    def frame(self, batch: RecordBatch, src: int, run: int | None = None, seq0: int = 0):
        import cudf
        import cupy as cp

        n = len(batch)
        cols = {}
        if self.by == "coordinate":
            # as the radix sort of samtools: tid (-1 as the largest), pos + 1, then the strand
            tid = batch.ref.astype(cp.uint32).astype(cp.uint64)
            pos = (batch.pos.astype(cp.int64) + 1).astype(cp.uint64)
            rev = ((batch.flag & REVERSE) != 0).astype(cp.uint64)
            cols["k"] = (tid << np.uint64(32)) | (pos << np.uint64(1)) | rev
        else:
            cols["name"] = self._names(batch)
            f = batch.flag.astype(cp.int32)
            # READ1, READ2, then primary, supplementary, secondary (bam1_cmp_core)
            cols["fk"] = ((f & 0xC0) << 8) | ((f & 0x100) << 3) | ((f & 0x800) >> 3)
        cols["src"] = cp.full(n, src, cp.int32)
        cols["off"] = batch.offset
        if run is not None:
            cols["run"] = cp.full(n, run, cp.int32)
            cols["seq"] = cp.arange(seq0, seq0 + n, dtype=cp.int64)
            cols["last"] = cp.zeros(n, cp.bool_)  # the merge cuts after the first of them
            cols["last"][-1] = True
        return cudf.DataFrame(cols)

    def _names(self, batch: RecordBatch):
        import cupy as cp

        n = len(batch)
        if not self.natural:
            lrn = batch.data[batch.offset + 12].astype(cp.int64)
            return _strings_at(batch.data, batch.offset + 36, lrn - 1)
        length = cp.empty(n, cp.int64)
        dummy = cp.empty(1, cp.int64)
        args = (batch.data, batch.offset, np.int64(n))
        launch("natural_name", n, *args, np.int32(0), dummy, length, cp.empty(1, cp.uint8))
        offsets = cp.zeros(n + 1, cp.int64)
        cp.cumsum(length, out=offsets[1:])
        chars = cp.empty(max(int(offsets[-1]), 1), cp.uint8)
        launch("natural_name", n, *args, np.int32(1), offsets[:-1], length, chars)
        return _string_series(chars, offsets)

    def sort(self, frames: list):
        """``(src, off)`` of the rows of ``frames`` (in input order), sorted by key: ties keep their order."""
        import cudf
        import cupy as cp
        import pylibcudf as plc

        df = cudf.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        frames.clear()  # free the parts
        keys, _ = df[self.columns].to_pylibcudf()
        n = len(self.columns)
        order = plc.sorting.stable_sorted_order(keys, [plc.types.Order.ASCENDING] * n, [plc.types.NullOrder.AFTER] * n)
        del keys
        order = cp.asarray(cudf.Series.from_pylibcudf(order).values)
        return df["src"].values[order], df["off"].values[order]

    def merge(self, frames: list):
        """Frames sorted by key, ``run`` and ``seq``, merged into one."""
        import cudf
        import pylibcudf as plc

        full = [f for f in frames if len(f)]
        if len(full) <= 1:
            return full[0] if full else frames[0]
        frames = full
        tables = [f.to_pylibcudf() for f in frames]
        names = list(frames[0].columns)
        idx = [names.index(c) for c in self.order]
        n = len(idx)
        out = plc.merge.merge(
            [t for t, _ in tables], idx, [plc.types.Order.ASCENDING] * n, [plc.types.NullOrder.AFTER] * n
        )
        return cudf.DataFrame.from_pylibcudf(out, {"columns": frames[0].columns})


def _write_sorted(src, off, sources: dict, writer: BgzfWriter) -> None:
    """Gather the records at ``off`` in ``sources[src]``, in this order, and write them."""
    import cupy as cp

    n = int(src.size)
    if n == 0:
        return
    ptr = np.zeros(max(sources) + 1, np.uint64)
    for i, data in sources.items():
        ptr[i] = data.data.ptr
    ptr = cp.asarray(ptr)
    size = cp.empty(n, cp.int64)
    launch("record_sizes_at", n, ptr, src, off, np.int64(n), size)
    end = cp.cumsum(size)
    total = int(end[-1])
    cuts = [0, *cp.searchsorted(end, cp.arange(_PIECE, total, _PIECE), side="right").get().tolist(), n]
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        if a == b:
            continue
        base = int(end[a - 1]) if a else 0
        dst = end[a:b] - size[a:b] - base
        buf = cp.empty(int(end[b - 1]) - base, cp.uint8)
        launch(
            "gather_spans", b - a, ptr, src[a:b], off[a:b], buf, dst, size[a:b], np.int64(b - a), threads_per_item=32
        )
        writer.write(buf)


def _hd_line(text: str, by: str, natural: bool) -> str:
    """``text`` with the ``@HD`` line of a file sorted by ``by``, as samtools sort writes it."""
    so = "coordinate" if by == "coordinate" else "queryname"
    ss = None if by == "coordinate" else ("queryname:natural" if natural else "queryname:lexicographical")
    lines = text.split("\n")
    if lines and lines[0].startswith("@HD\t"):
        fields = lines[0].split("\t")
        new = {"SO": so, "SS": ss, "GO": None}
        out = []
        for f in fields:
            tag = f[:2] if f[2:3] == ":" else None
            if tag in new:
                if new[tag] is not None:
                    out.append(f"{tag}:{new.pop(tag)}")
                else:
                    new.pop(tag)
            else:
                out.append(f)
        out += [f"{t}:{v}" for t, v in new.items() if v is not None]
        lines[0] = "\t".join(out)
    else:
        lines.insert(0, f"@HD\tVN:1.6\tSO:{so}" + (f"\tSS:{ss}" if ss else ""))
    return "\n".join(lines)


def _pg_line(text: str, by: str, natural: bool) -> str:
    """A ``@PG`` line for the sort, chained to the last ``@PG`` line of ``text``."""
    ids = [f[3:] for line in text.split("\n") if line.startswith("@PG\t") for f in line.split("\t") if f[:3] == "ID:"]
    pg_id, k = "cubam.sort_bam", 1
    while pg_id in ids:
        pg_id, k = f"cubam.sort_bam.{k}", k + 1
    cl = shlex.join([Path(sys.argv[0]).name or "python", *sys.argv[1:]])
    order = by if by == "coordinate" else f"name ({'natural' if natural else 'lexicographical'})"
    pp = f"\tPP:{ids[-1]}" if ids else ""
    return f"@PG\tID:{pg_id}\tPN:cubam{pp}\tCL:{cl}\tDS:sort by {order}\n"


def _run_bytes(chunk_bytes: int) -> int:
    """Bytes of records sorted per run: 32 chunks (8 GB with chunks of 256 MB), at most half the free GPU memory.

    Larger runs save little time: the sort is bound by the file system and
    the Deflate compression, not by the number of runs merged.
    """
    import cupy as cp

    free, _ = cp.cuda.runtime.memGetInfo()
    return max(8 * chunk_bytes, min(32 * chunk_bytes, (free - 24 * chunk_bytes) // 2))


def sort_bam(
    bam: str | Path,
    output: str | Path,
    *,
    by: str = "coordinate",
    natural: bool = True,
    level: int = 2,
    tmp_dir: str | Path | None = None,
    run_bytes: int | None = None,
    chunk_bytes: int | None = None,
) -> SortResult:
    """Write the records of ``bam`` to ``output``, sorted as ``samtools sort`` sorts them.

    Parameters
    ----------
    by
        ``"coordinate"`` (as ``samtools sort``) or ``"name"`` (as
        ``samtools sort -n``).
    natural
        With ``by="name"``: compare the numbers in read names by value, as
        ``samtools sort -n``; ``False`` compares bytes, as ``samtools sort -N``.
    level
        nvCOMP Deflate level of the output, 0 to 5 (see
        :func:`cubam.apps.split.split_bam`). Runs are written at level 2.
    tmp_dir
        Where to write the runs (default: the directory of ``output``).
        They take about twice the size of the input.
    run_bytes
        Bytes of records sorted at a time on the GPU (default: 32 times
        ``chunk_bytes``, 8 GB on a GPU of 24 GB or more). The GPU memory in
        use peaks at about this plus 10 GB.
    chunk_bytes
        Compressed bytes read at a time (default: :func:`cubam.auto_chunk_bytes`).
    """

    from cubam import auto_chunk_bytes, memory_pool

    t0 = time.perf_counter()
    if by not in ("coordinate", "name"):
        raise ValueError(f"by must be 'coordinate' or 'name', not {by!r}")
    output = Path(output)
    if output.exists() and output.samefile(bam):
        raise ValueError(f"The output is the input BAM: {output}")
    header = BamHeader.read(bam)
    text = header.text if header.text.endswith("\n") or not header.text else header.text + "\n"
    out_head = header.to_bytes(_hd_line(text, by, natural) + _pg_line(text, by, natural))
    run_head = header.to_bytes()
    keys = _Keys(by, natural)
    stats = {"records": 0, "runs": 0}

    with memory_pool():
        chunk_bytes = chunk_bytes or auto_chunk_bytes()
        run_bytes = run_bytes or _run_bytes(chunk_bytes)
        tmp = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=tmp_dir or output.parent))
        try:
            runs: list[Path] = []
            frames, sources, held = [], {}, 0

            def write(path: Path, head: bytes, lvl: int) -> None:
                with BgzfWriter(path, lvl, flush_bytes=_PIECE) as w:
                    w.write(head)
                    w.flush()  # the header in blocks of its own, as samtools
                    if frames:
                        _write_sorted(*keys.sort(frames), sources, w)

            # pass 1: sorted runs
            for batch in read_bam(bam, chunk_bytes):
                if held >= run_bytes:
                    runs.append(tmp / f"run{len(runs):04d}.bam")
                    write(runs[-1], run_head, _RUN_LEVEL)
                    frames, sources, held = [], {}, 0
                sources[len(sources)] = batch.data
                frames.append(keys.frame(batch, len(sources) - 1))
                stats["records"] += len(batch)
                held += int(batch.data.size)
                del batch
            if not runs:
                write(output, out_head, level)
                stats["runs"] = 1
                return SortResult(output, stats["records"], 1, stats, time.perf_counter() - t0)
            runs.append(tmp / f"run{len(runs):04d}.bam")
            write(runs[-1], run_head, _RUN_LEVEL)
            frames, sources = [], {}
            stats["runs"] = len(runs)

            # pass 2: merge the runs
            k = len(runs)
            piece = int(min(chunk_bytes, max(1 << 20, run_bytes // (8 * k))))
            readers = [read_bam(p, piece, n_buffers=2, n_threads=2) for p in runs]
            try:
                with BgzfWriter(output, level, flush_bytes=_PIECE) as w:
                    w.write(out_head)
                    w.flush()
                    _merge(readers, keys, w)
            finally:
                for r in readers:
                    r.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return SortResult(output, stats["records"], stats["runs"], stats, time.perf_counter() - t0)


def _merge(readers: list, keys: _Keys, writer: BgzfWriter) -> None:
    """Merge sorted runs: write, again and again, the records up to the last one read from any run.

    The pool holds the records read and not yet written, sorted. A run is
    read further only once none of its records is left in the pool, so the
    last record read from each unfinished run is in the pool. The records
    not read yet sort after it, so every record up to the first of these
    "last" records is final.
    """
    import cupy as cp

    k = len(readers)
    pool = None
    sources: dict = {}
    active = set(range(k))
    seen = [0] * k
    next_src = 0
    pull = list(range(k))
    while True:
        frames = [] if pool is None else [pool]
        for r in pull:
            batch = next(readers[r], None)
            if batch is None:
                active.discard(r)
                continue
            sources[next_src] = batch.data
            frames.append(keys.frame(batch, next_src, r, seen[r]))
            seen[r] += len(batch)
            next_src += 1
            del batch
        df = keys.merge(frames)
        if not active:
            _write_sorted(df["src"].values, df["off"].values, sources, writer)
            return
        cut = int(cp.argmax(df["last"].values)) + 1
        _write_sorted(df["src"].values[:cut], df["off"].values[:cut], sources, writer)
        pool = df.iloc[cut:].reset_index(drop=True)
        left = cp.bincount(pool["run"].values, minlength=k).get() if len(pool) else np.zeros(k, np.int64)
        pull = [r for r in sorted(active) if left[r] == 0]
        used = set(cp.unique(pool["src"].values).get().tolist()) if len(pool) else set()
        for s in [s for s in sources if s not in used]:
            del sources[s]
