import gzip
import random
from collections import Counter

import pytest
from bamwriter import Read, sort_key, write_bam
from test_coverage import blocks_of

from cubam.apps.features import count_features
from cubam.bam import DUPLICATE, REVERSE, SECONDARY
from cubam.intervals import IntervalIndex

REFS = [("chr1", 20_000), ("chr2", 8_000)]
CELLS = ["AAAACCCCGGGGTTTT-1", "CCCCGGGGTTTTAAAA-1", "GGGGTTTTAAAACCCC-1"]


def make_index(rng):
    rows = []
    for g in range(40):
        c = rng.choice(["chr1", "chr1", "chr2"])
        s = rng.randrange(0, REFS[0 if c == "chr1" else 1][1] - 3000)
        st = rng.choice("+-")
        for _ in range(rng.randint(1, 4)):  # exons, possibly overlapping
            es = s + rng.randrange(0, 2500)
            rows.append((c, es, es + rng.randint(20, 400), f"G{g}", st))
    return rows, IntervalIndex(*zip(*rows, strict=True))


def make_reads(rng, n=3000):
    reads = []
    for i in range(n):
        ref = rng.choice([0, 0, 1])
        parts = [f"{rng.randint(10, 60)}M"]
        if rng.random() < 0.3:
            parts += [f"{rng.randint(50, 800)}N", f"{rng.randint(10, 60)}M"]
        cigar = rng.choice(["", "5S"]) + "".join(parts)
        pos = rng.randrange(0, REFS[ref][1] - 1000)
        tags = []
        cb = rng.choice(CELLS + ["TTTTTTTTTTTTTTTT-1", None])
        if cb:
            tags.append(("CB", "Z", cb))
        if rng.random() < 0.95:
            tags.append(("UB", "Z", "".join(rng.choices("AC", k=4))))  # few UMIs: many collisions
        flag = rng.choice([0, 0, REVERSE, REVERSE, DUPLICATE, SECONDARY])
        reads.append(Read(f"r{i}", flag, ref, pos, rng.choice([255, 255, 60, 1]), cigar, -1, -1, 0, tags=tags))
    return reads


def expected(reads, rows, *, umi=True, strand=None, assign="unique", min_overlap=1, positions="blocks"):
    fstrand = {f: st for *_, f, st in rows}
    counts = Counter()
    seen = set()
    for r in reads:
        tags = {t[0]: t[2] for t in r.tags}
        if r.mapq < 30 or r.flag & SECONDARY or (not umi and r.flag & DUPLICATE):
            continue
        if tags.get("CB") not in CELLS or (umi and "UB" not in tags):
            continue
        rev = bool(r.flag & REVERSE)
        if positions == "blocks":
            spans = blocks_of(r)
        else:
            p = (r.end - 1 if rev else r.pos) if positions == "five_prime" else (r.pos if rev else r.end - 1)
            spans = [(p, p + 1)]
        ov = {}
        for c, s, e, f, _ in rows:
            if c != REFS[r.ref][0]:
                continue
            for bs, be in spans:
                ov.setdefault(f, set()).update(range(max(s, bs), min(e, be)))
        hits = {f: len(b) for f, b in ov.items() if len(b) >= max(min_overlap, 1)}
        if strand == "sense":
            hits = {f: b for f, b in hits.items() if (fstrand[f] == "-") == rev}
        elif strand == "antisense":
            hits = {f: b for f, b in hits.items() if (fstrand[f] == "-") != rev}
        if assign == "unique":
            picked = list(hits) if len(hits) == 1 else []
        elif assign == "any":
            picked = list(hits)
        else:
            m = max(hits.values(), default=0)
            top = [f for f, b in hits.items() if b == m]
            picked = top if len(top) == 1 else []
        for f in picked:
            if umi:
                key = (tags["CB"], tags["UB"], f)
                if key in seen:
                    continue
                seen.add(key)
            counts[(tags["CB"], f)] += 1
    return dict(counts)


def got(res):
    c = res.counts.to_pandas()
    names = res.features["name"].tolist()
    return {(res.barcodes[int(r.cell)], names[int(r.feature)]): int(r.count) for r in c.itertuples()}


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    rng = random.Random(11)
    rows, index = make_index(rng)
    reads = make_reads(rng)
    p = write_bam(tmp_path_factory.mktemp("f") / "a.bam", REFS, sorted(reads, key=sort_key), block=2000)
    return rows, index, reads, p


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"strand": "antisense", "umi": False},
        {"assign": "any"},
        {"assign": "largest"},
        {"min_overlap": 25},
        {"positions": "five_prime"},
        {"positions": "three_prime", "strand": "sense"},
    ],
)
def test_counts_match_reference(data, kw):
    rows, index, reads, p = data
    umi = kw.pop("umi", True)
    res = count_features(p, index, CELLS, umi_tag="UB" if umi else None, chunk_bytes=6000, **kw)
    assert got(res) == expected(reads, rows, umi=umi, **kw)
    s = res.stats
    assert s["assigned"] + s["ambiguous"] + s["no_feature"] == s["reads"]


def test_features_from_tag(tmp_path):
    reads = []
    genes = ["G1", "G2", "G1;G2", None]
    rng = random.Random(2)
    exp = Counter()
    seen = set()
    for i in range(500):
        cb = rng.choice(CELLS)
        ub = rng.choice(["AAAA", "CCCC", "GGGG"])
        gx = rng.choice(genes)
        xf = rng.choice([25, 17, 0])
        tags = [("CB", "Z", cb), ("UB", "Z", ub), ("xf", "i", xf)] + ([("GX", "Z", gx)] if gx else [])
        reads.append(Read(f"r{i}", 0, 0, rng.randrange(0, 10_000), 255, "50M", -1, -1, 0, tags=tags))
        if gx and ";" not in gx and xf & 8 and (cb, ub, gx) not in seen:
            seen.add((cb, ub, gx))
            exp[(cb, gx)] += 1
    p = write_bam(tmp_path / "t.bam", REFS, sorted(reads, key=sort_key))
    res = count_features(p, "GX", CELLS, where=lambda b: (b.tag("xf").value & 8) != 0, chunk_bytes=3000)
    assert res.features["name"].tolist() == ["G1", "G2"]
    assert got(res) == dict(exp)
    assert res.stats["ambiguous"] > 0
    with pytest.raises(ValueError, match="unique"):
        count_features(p, "GX", CELLS, assign="any")


def test_write_mtx_and_sparse(data, tmp_path):
    rows, index, reads, p = data
    res = count_features(p, index, CELLS)
    m = res.sparse()
    assert m.shape == (3, len(index))
    assert int(m.sum()) == sum(got(res).values())
    res.write_mtx(tmp_path / "mtx")
    with gzip.open(tmp_path / "mtx" / "matrix.mtx.gz", "rt") as fh:
        lines = fh.read().splitlines()
    nf, nc, nnz = map(int, lines[2].split())
    assert (nf, nc, nnz) == (len(index), 3, len(res.counts))
    with gzip.open(tmp_path / "mtx" / "features.tsv.gz", "rt") as fh:
        feats = fh.read().splitlines()
    assert len(feats) == len(index) and feats[0].endswith("\tGene Expression")
    pytest.importorskip("anndata")
    a = res.to_anndata()
    assert a.shape == (3, len(index)) and int(a.X.sum()) == int(m.sum())
    assert list(a.var_names) == res.features["name"].astype(str).tolist()
