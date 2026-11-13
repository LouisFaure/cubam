"""Write one bigwig from runs saved by :meth:`cubam.apps.coverage.Coverage.write_bigwig`.

``python -m cubam._bigwig RUNS.npz OUT.bw`` runs in a subprocess, so groups are
written in parallel without re-running the caller's script (as the ``spawn``
start method of multiprocessing would).
"""

from __future__ import annotations

import json
import sys

import numpy as np


def write(path: str, contigs: list, ref, start, end, value) -> None:
    import pyBigWig

    bw = pyBigWig.open(path, "w")
    if bw is None:
        raise OSError(f"Cannot open {path} for writing")
    try:
        bw.addHeader([(str(n), int(ln)) for n, ln in contigs], maxZooms=10)
        bounds = np.flatnonzero(np.diff(ref)) + 1
        for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(ref)], strict=True):
            if hi > lo:
                bw.addEntries(
                    [contigs[ref[lo]][0]] * int(hi - lo),
                    start[lo:hi].astype(np.int64),
                    ends=end[lo:hi].astype(np.int64),
                    values=value[lo:hi].astype(np.float64),
                )
    finally:
        bw.close()


def main(argv: list[str]) -> int:
    runs, out = argv
    d = np.load(runs)
    contigs = json.loads(str(d["contigs"]))
    write(out, contigs, d["ref"], d["start"], d["end"], d["value"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
