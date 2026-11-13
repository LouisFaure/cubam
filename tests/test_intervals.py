import random

import cupy as cp
import pytest

from cubam.intervals import IntervalIndex

NAMES = ["chr1", "chr2", "chr3"]


def brute(ivs, queries):
    """{(query, feature): bases} by base-by-base overlap."""
    out = {}
    for qi, (qr, qs, qe) in enumerate(queries):
        for f in {iv[3] for iv in ivs}:
            cov = set()
            for c, s, e, ff in ivs:
                if ff == f and NAMES.index(c) == qr:
                    cov.update(range(max(s, qs), min(e, qe)))
            if cov:
                out[(qi, f)] = len(cov)
    return out


def run(index, queries):
    ref = cp.asarray([q[0] for q in queries], cp.int32)
    s = cp.asarray([q[1] for q in queries], cp.int32)
    e = cp.asarray([q[2] for q in queries], cp.int32)
    q, f, b = index.overlaps(NAMES, ref, s, e)
    names = index.features["name"].tolist()
    return {(int(a), names[int(c)]): int(d) for a, c, d in zip(q.get(), f.get(), b.get(), strict=True)}


def test_overlaps_match_brute_force():
    rng = random.Random(7)
    ivs = []
    for i in range(300):
        c = rng.choice(["chr1", "chr1", "chr2", "1"])  # "1" is chr1 without the prefix
        s = rng.randrange(0, 5000)
        ivs.append((c, s, s + rng.randint(0, 400), f"g{i % 60}"))
    ivs = [(("chr1" if c == "1" else c), s, e, f) for c, s, e, f in ivs]
    queries = []
    for _ in range(400):
        r = rng.choice([0, 0, 1, 2])
        s = rng.randrange(0, 5500)
        queries.append((r, s, s + rng.randint(0, 300)))
    index = IntervalIndex([c for c, *_ in ivs], [x[1] for x in ivs], [x[2] for x in ivs], [x[3] for x in ivs])
    assert run(index, queries) == brute(ivs, queries)


def test_nested_and_edges():
    # A contains B; C touches A at its end (no overlap at a shared boundary)
    index = IntervalIndex(["chr1"] * 3, [100, 150, 300], [300, 200, 400], ["A", "B", "C"])
    got = run(index, [(0, 0, 100), (0, 99, 101), (0, 160, 170), (0, 290, 310), (0, 300, 300), (0, 500, 600)])
    assert got == {(1, "A"): 1, (2, "A"): 10, (2, "B"): 10, (3, "A"): 10, (3, "C"): 10}


def test_from_gtf_and_bed(tmp_path):
    gtf = tmp_path / "g.gtf"
    rows = [
        ("chr1", "gene", 101, 300, "+", 'gene_id "G1"; gene_name "ONE"; gene_type "protein_coding";'),
        ("chr1", "exon", 101, 150, "+", 'gene_id "G1"; transcript_id "T1"; gene_name "ONE";'),
        ("chr1", "exon", 141, 200, "+", 'gene_id "G1"; transcript_id "T2"; gene_name "ONE";'),
        ("chr1", "exon", 251, 300, "+", 'gene_id "G1"; transcript_id "T1"; gene_name "ONE";'),
        ("chr2", "exon", 11, 20, "-", 'gene_id "G2"; transcript_id "T3"; gene_name "TWO";'),
    ]
    lines = [f"{c}\tsrc\t{t}\t{s}\t{e}\t.\t{st}\t.\t{a}\n" for c, t, s, e, st, a in rows]
    gtf.write_text("#!genome-build x\n" + "".join(lines))
    ex = IntervalIndex.from_gtf(gtf)
    assert ex.features["name"].tolist() == ["G1", "G2"]
    assert ex.features["gene_name"].tolist() == ["ONE", "TWO"]
    assert ex.features["strand"].tolist() == ["+", "-"]
    assert ex.n_intervals == 4
    # exons overlap within G1: bases 100-200 and 250-300, counted once
    assert run(ex, [(0, 0, 1000), (1, 0, 1000)]) == {(0, "G1"): 150, (1, "G2"): 10}
    genes = IntervalIndex.from_gtf(gtf, feature="gene")
    assert run(genes, [(0, 0, 1000)]) == {(0, "G1"): 200}
    with pytest.raises(ValueError, match="no rows"):
        IntervalIndex.from_gtf(gtf, feature="CDS")

    bed = tmp_path / "p.bed"
    bed.write_text("track name=x\nchr1\t10\t20\tpeakA\t0\t+\nchr2\t5\t8\tpeakB\t0\t-\n")
    b = IntervalIndex.from_bed(bed)
    assert b.features["name"].tolist() == ["peakA", "peakB"]
    assert b.strand_codes().get().tolist() == [1, -1]
    import gzip

    with gzip.open(tmp_path / "p.bed.gz", "wt") as fh:
        fh.write("# comment\nchr1\t10\t20\tpeakA\n")
    assert IntervalIndex.from_bed(tmp_path / "p.bed.gz").features["name"].tolist() == ["peakA"]
    bed.write_text("chr1\t10\t20\n")
    assert IntervalIndex.from_bed(bed).features["name"].tolist() == ["chr1:10-20"]


def test_bad_intervals():
    with pytest.raises(ValueError, match="start <= end"):
        IntervalIndex(["chr1"], [10], [5], ["x"])
    with pytest.raises(ValueError, match="same length"):
        IntervalIndex(["chr1"], [10], [20], ["x", "y"])


def test_contig_missing_from_header():
    index = IntervalIndex(["chrX", "chr1"], [0, 0], [10, 10], ["x", "a"])
    assert run(index, [(0, 0, 100)]) == {(0, "a"): 10}
