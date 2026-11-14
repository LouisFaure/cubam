import numpy as np
import pytest

import cubam.bigwig as bwmod
from cubam.bigwig import write_bigwig

pyBigWig = pytest.importorskip("pyBigWig")

NAMES = ["chr1", "chr10", "chr2", "chrUn_KI270302v1"]
LENGTHS = [300_000, 120_000, 200_000, 2_274]


def random_runs(rng, n):
    rows = []
    for r, length in enumerate(LENGTHS[:3]):
        cuts = np.unique(rng.integers(0, length, 2 * n))
        k = len(cuts) // 2
        s, e = cuts[0::2][:k], cuts[1::2][:k]
        rows.append((np.full(k, r), s, e, rng.normal(3, 2, k).astype(np.float32)))
    return (np.concatenate(x) for x in zip(*rows, strict=True))


@pytest.mark.parametrize(("slot", "block"), [(1024, 256), (3, 2)])
def test_bigwig_reads_back(tmp_path, monkeypatch, slot, block):
    # small sections and tree nodes give deep R-trees and many zoom sections on little data
    monkeypatch.setattr(bwmod, "ITEMS_PER_SLOT", slot)
    monkeypatch.setattr(bwmod, "_RTREE_BLOCK", block)
    ref, s, e, v = random_runs(np.random.default_rng(slot), 3000)
    p = tmp_path / "t.bw"
    write_bigwig(p, NAMES, LENGTHS, ref, s, e, v)
    bw = pyBigWig.open(str(p))
    assert bw.chroms() == dict(zip(NAMES, LENGTHS, strict=True))
    h = bw.header()
    w = (e - s).astype(np.float64)
    assert h["nBasesCovered"] == int(w.sum())
    assert h["nLevels"] >= 1
    for r, name in enumerate(NAMES):
        m = ref == r
        want = tuple(zip(s[m].tolist(), e[m].tolist(), v[m].astype(float).tolist(), strict=True))
        assert tuple(bw.intervals(name) or ()) == want
        if m.any():
            # whole-contig summaries from the zoom levels equal the exact ones
            for t in ("mean", "max", "min", "coverage"):
                z = bw.stats(name, 0, LENGTHS[r], type=t)[0]
                x = bw.stats(name, 0, LENGTHS[r], type=t, exact=True)[0]
                assert z == pytest.approx(x, rel=1e-5)
    # a query inside a contig goes through the R-tree
    got = bw.intervals("chr2", 50_000, 60_000)
    m = (ref == 2) & (e > 50_000) & (s < 60_000)
    assert [g[:2] for g in got] == list(zip(s[m].tolist(), e[m].tolist(), strict=True))
    bw.close()
