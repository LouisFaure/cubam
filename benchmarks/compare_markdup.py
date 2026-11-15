"""cubam markdup against samtools markdup on one coordinate-sorted BAM: time, memory, and the flags of every read.

    uv run --extra rapids-cu12 python benchmarks/compare_markdup.py sorted.bam WORKDIR \
        [--barcode-tag CB] [--threads 6] [--skip-samtools]

The input needs MC tags (BWA-MEM, Parabricks fq2bam) and no duplicate flags. samtools
starts from the same file, as it must: ``samtools collate | fixmate -m | sort |
markdup``, with ``--threads`` threads (``samtools`` on the PATH). The peak host memory
of each tool is that of its largest process; the peak GPU memory of cubam is sampled with
nvidia-smi. Then the duplicate flag of every primary read is compared, by read name
and mate: both outputs hold the same records, in an order that may differ at equal
coordinates: single reads with an unmapped mate and the same score keep the first in the
file, so they can differ. Last, cubam decides on the file samtools marked, in its record
order, and every flag is compared. Writes WORKDIR/markdup_benchmark.json.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# runs a command and prints the peak RSS (kB) of its largest process, from getrusage of the children
_RUSAGE = (
    "import resource, subprocess, sys\n"
    "r = subprocess.run(['bash', '-o', 'pipefail', '-c', sys.argv[1]])\n"
    "print(f'MAXRSS_KB {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}', file=sys.stderr)\n"
    "sys.exit(r.returncode)\n"
)


def timed(cmd: str, log: Path) -> dict:
    """Run ``cmd`` in bash: wall seconds and peak RSS of its largest process."""
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-c", _RUSAGE, cmd], capture_output=True, text=True)
    sec = time.perf_counter() - t0
    log.write_text(p.stdout + p.stderr)
    if p.returncode:
        sys.exit(f"failed ({p.returncode}): {cmd}\n{p.stderr[-2000:]}")
    rss = re.search(r"MAXRSS_KB (\d+)", p.stderr)
    return {"seconds": round(sec, 1), "max_rss_gb": round(int(rss.group(1)) / 2**20, 1) if rss else None}


class GpuPeak:
    """Peak GPU memory in use (MiB), sampled every 0.2 s with nvidia-smi."""

    def __init__(self):
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            ).stdout.split()
            if out:
                self.peak = max(self.peak, int(out[0]))
            time.sleep(0.2)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def dup_keys(path: Path):
    """Sorted keys (read-name hash, read 1 or 2) of all primary reads and of the primary duplicates."""
    import cupy as cp

    from cubam import read_bam
    from cubam._memory import memory_pool
    from cubam.bam import DUPLICATE, READ1, SECONDARY, SUPPLEMENTARY

    every, dups = [], []
    with memory_pool():
        for b in read_bam(path):
            primary = b.has_flags(none_of=SECONDARY | SUPPLEMENTARY)
            key = (b.name_hash() << 1) | ((b.flag & READ1) != 0).astype(cp.uint64)
            every.append(key[primary])
            dups.append(key[primary & ((b.flag & DUPLICATE) != 0)])
        return cp.sort(cp.concatenate(every)), cp.sort(cp.concatenate(dups))


def dup_flags(path: Path):
    """Duplicate flag of every record, in file order (numpy bool)."""
    import numpy as np

    from cubam import read_bam
    from cubam._memory import memory_pool
    from cubam.bam import DUPLICATE

    with memory_pool():
        return np.concatenate([((b.flag & DUPLICATE) != 0).get() for b in read_bam(path)])


def main() -> None:
    from cubam.apps.markdup import duplicate_mask

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bam", type=Path)
    ap.add_argument("workdir", type=Path)
    ap.add_argument("--barcode-tag", default="CB")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--skip-samtools", action="store_true", help="reuse WORKDIR/samtools.bam")
    args = ap.parse_args()
    w = args.workdir
    w.mkdir(parents=True, exist_ok=True)
    t = args.threads
    bc = f"--barcode-tag {args.barcode_tag}" if args.barcode_tag else ""
    res = {"input": str(args.bam), "threads": t}

    with GpuPeak() as gpu:
        res["cubam"] = timed(
            f"{sys.executable} -m cubam.cli markdup {args.bam} {bc} -o {w}/cubam.bam -f {w}/cubam_stats.json",
            w / "cubam.log",
        )
    res["cubam"]["gpu_peak_gb"] = round(gpu.peak / 1024, 1)

    if not args.skip_samtools:
        res["samtools"] = timed(
            f"samtools collate -@ {t} -O -u -T {w}/collate {args.bam} | samtools fixmate -m -u -@ {t} - - "
            f"| samtools sort -u -@ {t} -m 2G -T {w}/sort - "
            f"| samtools markdup {bc} -@ {t} -f {w}/samtools_stats.txt - {w}/samtools.bam",
            w / "samtools.log",
        )
    if res.get("samtools"):
        res["speedup"] = round(res["samtools"]["seconds"] / res["cubam"]["seconds"], 1)

    import cupy as cp

    a_all, a_dup = dup_keys(w / "cubam.bam")
    b_all, b_dup = dup_keys(w / "samtools.bam")
    res["primary_reads"] = int(a_all.size)
    res["same_primary_reads"] = bool(a_all.size == b_all.size and (a_all == b_all).all())
    res["duplicates_cubam"] = int(a_dup.size)
    res["duplicates_samtools"] = int(b_dup.size)
    res["only_cubam"] = int((~cp.isin(a_dup, b_dup)).sum())
    res["only_samtools"] = int((~cp.isin(b_dup, a_dup)).sum())
    # the same rules on the same record order: cubam's decision on the file samtools marked
    mask, _ = duplicate_mask(w / "samtools.bam", barcode_tag=args.barcode_tag)
    theirs = dup_flags(w / "samtools.bam")
    res["same_order_records"] = int(mask.size)
    res["same_order_differences"] = int((mask != theirs).sum())
    (w / "markdup_benchmark.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
