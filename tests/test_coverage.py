import random

import numpy as np
import pandas as pd
import pytest
from bamwriter import Read, parse_cigar, sort_key, write_bam

from cubam import read_bam
from cubam.apps.coverage import coverage, fragment_coverage
from cubam.apps.fragments import Fragments
from cubam.bam import DUPLICATE, REVERSE, BamHeader
from cubam.cells import CellIndex, _pack_acgt

REFS = [("chr1", 20_000), ("chr2", 5_000), ("chrM", 3_000)]
CELLS = ["AAAACCCCGGGGTTTT-1", "CCCCGGGGTTTTAAAA-1", "GGGGTTTTAAAACCCC-1", "TTTTAAAACCCCGGGG-1"]
LABELS = ["T", "B", "T", None]  # the last cell has no group


def read(name, ref, pos, cigar, *, cb=CELLS[0], flag=0, mapq=60):
    tags = [("CB", "Z", cb)] if cb else []
    return Read(name, flag, ref, pos, mapq, cigar, -1, -1, 0, tags=tags)


def blocks_of(r, split_deletions=False):
    out, x, s = [], r.pos, r.pos
    for op, n in parse_cigar(r.cigar):
        if op in "M=X" or (op == "D" and not split_deletions):
            x += n
        elif op == "N" or op == "D":
            if x > s:
                out.append((s, x))
            x += n
            s = x
    if x > s:
        out.append((s, x))
    return out


def runs_of(cov):
    """Maximal runs of equal non-zero value of per-base arrays: {(ref, start, end): value}."""
    out = {}
    for ref, a in enumerate(cov):
        change = np.flatnonzero(np.diff(np.r_[0, a, 0])) if a.size else []
        for s, e in zip(change[:-1], change[1:], strict=False):
            if a[s]:
                out[(ref, int(s), int(e))] = int(a[s])
    return out


def got_runs(c, group):
    t = c.runs
    t = t[t["group"] == c.groups.index(group)]
    return {(int(r.ref), int(r.start), int(r.end)): int(r.value) for r in t.itertuples()}


def write(tmp_path, reads, block=4096):
    return write_bam(tmp_path / "c.bam", REFS, sorted(reads, key=sort_key), block=block)


def test_blocks(tmp_path):
    cigars = ["50M", "10S20M100N30M5S", "10M2D10M3I10M", "5M10N5M10N5M", "3H20M", "10M5D"]
    reads = [read(f"r{i}", 0, 100 + 300 * i, c) for i, c in enumerate(cigars)]
    b = next(read_bam(write(tmp_path, reads)))
    for split in (False, True):
        rec, s, e = (x.get().tolist() for x in b.blocks(split))
        expected = [(i, bs, be) for i, r in enumerate(reads) for bs, be in blocks_of(r, split)]
        assert list(zip(rec, s, e, strict=True)) == expected


@pytest.mark.parametrize(("block", "chunk"), [(4096, 1 << 20), (300, 2000)])
def test_read_coverage_matches_per_base(tmp_path, block, chunk):
    rng = random.Random(3)
    reads = []
    for i in range(2500):
        ref = rng.choice([0, 0, 1, 2])
        ln = REFS[ref][1]
        parts = [f"{rng.randint(5, 40)}M"]
        for _ in range(rng.randint(0, 2)):
            parts.append(rng.choice([f"{rng.randint(50, 400)}N", f"{rng.randint(1, 4)}D", f"{rng.randint(1, 3)}I"]))
            parts.append(f"{rng.randint(5, 40)}M")
        cigar = rng.choice(["", "4S"]) + "".join(parts)
        pos = rng.randrange(0, ln - 1000)
        flag = rng.choice([0, 0, 0, REVERSE, DUPLICATE])
        cb = rng.choice(CELLS + ["ACGTACGTACGTACGT-1", None])
        reads.append(read(f"r{i}", ref, pos, cigar, cb=cb, flag=flag, mapq=rng.choice([60, 60, 255, 3])))
    p = write(tmp_path, reads, block)
    cells = CellIndex(CELLS, labels=LABELS)
    for strand in (None, "-"):
        c = coverage(p, cells, strand=strand, chunk_bytes=chunk)
        assert c.groups == ["T", "B"]
        for g in c.groups:
            cov = [np.zeros(ln, np.int64) for _, ln in REFS]
            n = 0
            for r in reads:
                cb = dict((t[0], t[2]) for t in r.tags).get("CB")
                if cb not in CELLS or LABELS[CELLS.index(cb)] != g:
                    continue
                if r.mapq < 30 or r.flag & DUPLICATE:
                    continue
                if strand == "-" and not r.flag & REVERSE:
                    continue
                n += 1
                for s, e in blocks_of(r):
                    cov[r.ref][s:e] += 1
            assert got_runs(c, g) == runs_of(cov), (strand, g)
            assert c.units[c.groups.index(g)] == n
        assert c.n_cells.tolist() == [2, 1]


def fragments_table(rows):
    """A Fragments object from (ref, start, end, barcode) rows, sorted by position."""
    rows = sorted(rows)
    t = pd.DataFrame(
        {
            "ref": np.array([r[0] for r in rows], np.int32),
            "start": np.array([r[1] for r in rows], np.int32),
            "end": np.array([r[2] for r in rows], np.int32),
            "bc": _pack_acgt([r[3] for r in rows]),
            "count": np.ones(len(rows), np.int32),
        }
    )
    return Fragments(t, BamHeader("", [n for n, _ in REFS], [ln for _, ln in REFS], 0))


@pytest.mark.parametrize("width", [1, 5])
def test_fragment_coverage_matches_per_base(width):
    rng = random.Random(5)
    rows = []
    for _ in range(3000):
        ref = rng.choice([0, 0, 1, 2])
        s = rng.randrange(0, REFS[ref][1] - 10)
        e = min(s + rng.randint(1, 600), REFS[ref][1])
        rows.append((ref, s, e, rng.choice(CELLS + ["ACGTACGTACGTACGT-1"])))
    frags = fragments_table(rows)
    cells = CellIndex(CELLS, labels=LABELS)
    ins = fragment_coverage(frags, cells, mode="insertions", insertion_width=width)
    whole = fragment_coverage(frags, cells, mode="fragments")
    for g in ins.groups:
        mine = [r for r in rows if r[3] in CELLS and LABELS[CELLS.index(r[3])] == g]
        cov_i = [np.zeros(ln, np.int64) for _, ln in REFS]
        cov_f = [np.zeros(ln, np.int64) for _, ln in REFS]
        for ref, s, e, _ in mine:
            cov_f[ref][s:e] += 1
            for p in (s, e - 1):
                lo = max(p - width // 2, 0)
                cov_i[ref][lo : max(p - width // 2 + width, 0)] += 1
        assert got_runs(ins, g) == runs_of(cov_i)
        assert got_runs(whole, g) == runs_of(cov_f)
        assert ins.units[ins.groups.index(g)] == 2 * len(mine)
        assert whole.units[whole.groups.index(g)] == len(mine)


def test_normalise_and_bins():
    rows = [(0, 0, 100, CELLS[0]), (0, 50, 150, CELLS[0]), (0, 19_990, 20_000, CELLS[1])]
    c = fragment_coverage(fragments_table(rows), CellIndex(CELLS, labels=LABELS), mode="fragments")
    t = c.to_frame("T", normalize="cpm").to_pandas()
    assert t[["start", "end"]].values.tolist() == [[0, 50], [50, 100], [100, 150]]
    assert np.allclose(t["value"], np.array([1, 2, 1]) * 1e6 / 2)
    t = c.to_frame("T", normalize="cells", bin_size=100).to_pandas()
    assert t[["start", "end"]].values.tolist() == [[0, 100], [100, 200]]
    assert np.allclose(t["value"], [1.5 / 2, 0.5 / 2])
    t = c.to_frame("B", bin_size=64).to_pandas()  # the last bin of chr1 is 20000 - 19968 = 32 bases
    assert t[["start", "end"]].values.tolist() == [[19_968, 20_000]]
    assert np.allclose(t["value"], [10 / 32])
    with pytest.raises(ValueError, match="normalize"):
        c.to_frame("T", normalize="rpkm")


def test_write_bigwig(tmp_path):
    pyBigWig = pytest.importorskip("pyBigWig")
    rows = [(0, 0, 100, CELLS[0]), (0, 50, 150, CELLS[0]), (1, 10, 20, CELLS[0]), (2, 5, 9, CELLS[1])]
    c = fragment_coverage(fragments_table(rows), CellIndex(CELLS, labels=["T/1", "B", "T/1", None]), mode="fragments")
    paths = c.write_bigwig(tmp_path / "bw", normalize=None, prefix="s1_")
    assert sorted(p.name for p in paths.values()) == ["s1_B.bw", "s1_T_1.bw"]
    bw = pyBigWig.open(str(paths["T/1"]))
    assert bw.chroms() == dict(REFS)
    assert bw.intervals("chr1") == ((0, 50, 1.0), (50, 100, 2.0), (100, 150, 1.0))
    assert bw.intervals("chr2") == ((10, 20, 1.0),)
    bw.close()
