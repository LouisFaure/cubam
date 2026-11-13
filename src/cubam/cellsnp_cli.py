"""
``cubam-cellsnp``: a stand-in for ``cellsnp-lite`` in mode 1a, on the GPU.

It takes the arguments of cellsnp-lite. Mode 1a (one BAM with ``-s``, a
barcode list with ``-b``, sites with ``-R``) runs on the GPU with
:func:`cubam.apps.alleles.count_alleles` and writes the same files. Any other
mode, or an option that cubam does not implement, runs the real cellsnp-lite:
the one named by ``CUBAM_CELLSNP_FALLBACK``, or else the first
``cellsnp-lite`` on ``PATH`` that is not this command.

Put a ``cellsnp-lite`` link to it first on ``PATH`` to speed up tools that
call cellsnp-lite, such as the ``pileup_and_phase.R`` script of numbat.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

# Options of cellsnp-lite that mode 1a on the GPU implements; any other one goes to cellsnp-lite
_VALUE_OPTS = {
    "-s": "bam",
    "--samFile": "bam",
    "-b": "barcodes",
    "--barcodeFile": "barcodes",
    "-O": "outdir",
    "--outDir": "outdir",
    "-R": "sites",
    "--regionsVCF": "sites",
    "-p": "nproc",
    "--nproc": "nproc",
    "--minMAF": "min_maf",
    "--minCOUNT": "min_count",
    "--UMItag": "umi_tag",
    "--cellTAG": "cell_tag",
    "--minMAPQ": "min_mapq",
    "--minLEN": "min_len",
    "--exclFLAG": "excl_flag",
}
_FLAG_OPTS = {"--countORPHAN": "count_orphan"}


def _parse(argv: list[str]) -> dict | None:
    """The options of a mode 1a run, or None when cubam does not implement the call."""
    out: dict = {}
    i = 0
    while i < len(argv):
        a = argv[i]
        key, value = (a.split("=", 1) + [None])[:2] if a.startswith("--") and "=" in a else (a, None)
        if key in _FLAG_OPTS:
            out[_FLAG_OPTS[key]] = True
            i += 1
            continue
        if key not in _VALUE_OPTS:
            return None
        if value is None:
            if i + 1 >= len(argv):
                return None
            value = argv[i + 1]
            i += 1
        out[_VALUE_OPTS[key]] = value
        i += 1
    if not all(k in out for k in ("bam", "barcodes", "outdir", "sites")) or "," in out["bam"]:
        return None
    if out.get("cell_tag", "CB") in ("None", "none"):
        return None
    return out


def _fallback(argv: list[str]) -> int:
    exe = os.environ.get("CUBAM_CELLSNP_FALLBACK")
    if not exe:
        me = Path(sys.argv[0]).resolve()
        for d in os.environ.get("PATH", "").split(os.pathsep):
            cand = shutil.which("cellsnp-lite", path=d)
            if cand and Path(cand).resolve() != me:
                exe = cand
                break
    if not exe:
        print("[E::cubam-cellsnp] mode not supported on the GPU, and no cellsnp-lite to fall back to", file=sys.stderr)
        return 1
    os.execv(exe, [exe, *argv])
    return 1  # not reached


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    opts = _parse(argv)
    if opts is None:
        return _fallback(argv)
    from cubam.apps.alleles import count_alleles

    umi = opts.get("umi_tag", "Auto")
    # cellsnp-lite: Auto is UB with a barcode list
    umi = "UB" if umi == "Auto" else None if umi in ("None", "none") else umi
    kw = {
        "cell_tag": opts.get("cell_tag", "CB"),
        "umi_tag": umi,
        "min_count": int(opts.get("min_count", 20)),
        "min_maf": float(opts.get("min_maf", 0.0)),
        "min_mapq": int(opts.get("min_mapq", 20)),
        "min_len": int(opts.get("min_len", 30)),
        "no_orphan": not opts.get("count_orphan", False),
    }
    if "excl_flag" in opts:
        kw["exclude_flags"] = int(opts["excl_flag"], 0)
    t0 = time.time()
    print(f"[I::main] cubam-cellsnp: {opts['bam']} (cellsnp-lite mode 1a on the GPU)", file=sys.stderr)
    res = count_alleles(opts["bam"], opts["sites"], opts["barcodes"], **kw)
    res.write_cellsnp(opts["outdir"])
    print(
        f"[I::main] {len(res.sites)} sites, {len(res.counts)} (site, cell) entries; "
        f"time spent: {time.time() - t0:.0f} seconds.",
        file=sys.stderr,
    )
    print("[I::main] All Done!", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
