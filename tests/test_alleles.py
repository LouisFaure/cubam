import random

import pandas as pd
import pytest
from bamwriter import Read, sort_key, write_bam

from cubam.apps.alleles import count_alleles

REFS = [("chr1", 100_000), ("chr2", 100_000)]
CELLS = ["AAAACCCCGGGGTTTT-1", "CCCCGGGGTTTTAAAA-1"]


def read(name, pos, seq, *, cell=CELLS[0], umi="ACGTACGTACGT", flag=0, mapq=60, cigar=None, ref=0):
    tags = []
    if cell:
        tags.append(("CB", "Z", cell))
    if umi:
        tags.append(("UB", "Z", umi))
    return Read(name, flag, ref, pos, mapq, cigar or f"{len(seq)}M", -1, -1, 0, seq=seq, tags=tags)


def seq_with(base_at: int, base: str, n: int = 50, fill: str = "C") -> str:
    """A read sequence of n bases with ``base`` at offset ``base_at``."""
    return fill * base_at + base + fill * (n - base_at - 1)


def sites(*rows):
    return pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt"])


def run(tmp_path, reads, site_df, **kw):
    p = write_bam(tmp_path / "a.bam", REFS, sorted(reads, key=sort_key), block=kw.pop("block", 4096))
    kw.setdefault("min_count", 1)
    res = count_alleles(p, site_df, CELLS, chunk_bytes=kw.pop("chunk_bytes", 1 << 20), **kw)
    c = res.counts.to_pandas()
    got = {(int(r.site), int(r.cell)): (int(r.AD), int(r.DP), int(r.OTH)) for r in c.itertuples()}
    return res, got


def test_counts_per_cell(tmp_path):
    # site chr1:1011 (1-based) = 0-based 1010; reads start at 1000, so offset 10
    reads = [
        read("r1", 1000, seq_with(10, "A"), umi="AAAAAAAAAAAA"),
        read("r2", 1000, seq_with(10, "G"), umi="CCCCCCCCCCCC"),
        read("r3", 1000, seq_with(10, "T"), umi="GGGGGGGGGGGG"),
        read("r4", 1001, seq_with(9, "G"), cell=CELLS[1]),
    ]
    res, got = run(tmp_path, reads, sites(("1", 1011, "A", "G")))
    assert got == {(0, 0): (1, 2, 1), (0, 1): (1, 1, 0)}
    assert res.sites[["AD", "DP", "OTH"]].values.tolist() == [[2, 3, 1]]


def test_umi_first_read_in_file_order(tmp_path):
    reads = [
        read("r1", 1000, seq_with(10, "A"), umi="AAAAAAAAAAAA"),
        read("r2", 1005, seq_with(5, "G"), umi="AAAAAAAAAAAA"),  # same UMI, later: ignored
        read("r3", 1005, seq_with(5, "G"), umi="TTTTTTTTTTTT"),
    ]
    _, got = run(tmp_path, reads, sites(("1", 1011, "A", "G")))
    assert got == {(0, 0): (1, 2, 0)}


def test_no_umi_counts_every_read(tmp_path):
    reads = [
        read("p1", 1000, seq_with(10, "G"), umi=None, flag=0x1 | 0x2 | 0x40),
        read("p1", 1005, seq_with(5, "G"), umi=None, flag=0x1 | 0x2 | 0x10 | 0x80),  # its mate, same site
        read("d", 1000, seq_with(10, "G"), umi=None, flag=0x400),  # duplicate: skipped
        read("o", 1000, seq_with(10, "G"), umi=None, flag=0x1),  # orphan: skipped
    ]
    _, got = run(tmp_path, reads, sites(("1", 1011, "A", "G")), umi_tag=None)
    assert got == {(0, 0): (2, 2, 0)}


def test_read_filters(tmp_path):
    reads = [
        read("ok", 1000, seq_with(10, "G"), umi="A" * 12),
        read("lowq", 1000, seq_with(10, "G"), umi="C" * 12, mapq=19),
        read("short", 1000, seq_with(10, "G", n=29), umi="G" * 12),
        read("del", 1000, "C" * 50, umi="T" * 12, cigar="10M2D40M"),  # deletion over the site
        read("skip", 1000, "C" * 50, umi="AC" * 6, cigar="5M100N45M"),  # intron over the site
        read("clip", 1011, "G" * 50, umi="AG" * 6, cigar="10S40M"),  # the site is soft-clipped
        read("other", 1000, seq_with(10, "G"), umi="AT" * 6, cell="GGGGGGGGGGGGGGGG-1"),
        read("noumi", 1000, seq_with(10, "G"), umi=None),
        read("nocell", 1000, seq_with(10, "G"), umi="CA" * 6, cell=None),
        read("sec", 1000, seq_with(10, "G"), umi="CG" * 6, flag=0x100),
    ]
    res, got = run(tmp_path, reads, sites(("1", 1011, "A", "G")))
    assert got == {(0, 0): (1, 1, 0)}
    assert res.stats["reads"] == 4  # ok, del, skip and clip pass the read filters; only ok has a base there


def test_site_filters_and_order(tmp_path):
    reads = [read(f"r{i}", 1000 + i, "G" * 50, umi=f"{i:012d}".replace("0", "A").replace("1", "C")) for i in range(3)]
    site_df = sites(
        ("chr1", 1031, "A", "G"),  # chr prefix in the sites
        ("1", 1021, "AT", "G"),  # REF longer than one base: skipped
        ("1", 1021, "A", "G,T"),  # more than two alleles: skipped
        ("1", 1011, "A", "G"),
        ("2", 1011, "A", "G"),  # no read
        ("3", 1011, "A", "G"),  # not in the BAM header
    )
    res, got = run(tmp_path, reads, site_df)
    assert res.sites[["chrom", "pos"]].values.tolist() == [["chr1", 1031], ["1", 1011]]
    assert got == {(0, 0): (3, 3, 0), (1, 0): (3, 3, 0)}
    assert res.stats["sites_skipped"] == 2


def test_min_count(tmp_path):
    reads = [read("r1", 1000, seq_with(10, "G")), read("r2", 2000, seq_with(10, "G"), umi="C" * 12)]
    reads.append(read("r3", 2000, seq_with(10, "A"), umi="G" * 12))
    res, _ = run(tmp_path, reads, sites(("1", 1011, "A", "G"), ("1", 2011, "A", "G")), min_count=2)
    assert res.sites["pos"].tolist() == [2011]


def test_write_cellsnp(tmp_path):
    reads = [read("r1", 1000, seq_with(10, "G")), read("r2", 1000, seq_with(10, "A"), umi="C" * 12, cell=CELLS[1])]
    res, _ = run(tmp_path, reads, sites(("1", 1011, "A", "G")))
    res.write_cellsnp(tmp_path / "out")
    vcf = (tmp_path / "out" / "cellSNP.base.vcf").read_text().splitlines()
    assert vcf[-1] == "1\t1011\t.\tA\tG\t.\tPASS\tAD=1;DP=2;OTH=0"
    dp = (tmp_path / "out" / "cellSNP.tag.DP.mtx").read_text().splitlines()
    assert dp[2:] == ["1\t2\t2", "1\t1\t1", "1\t2\t1"]
    assert (tmp_path / "out" / "cellSNP.samples.tsv").read_text().split() == CELLS


@pytest.mark.parametrize(("block", "chunk"), [(4096, 1 << 20), (600, 4000)])
def test_many_chunks_same_counts(tmp_path, block, chunk):
    rng = random.Random(0)
    reads = []
    for i in range(2000):
        pos = rng.randrange(0, 20_000)
        seq = "".join(rng.choices("ACGT", k=60))
        umi = "".join(rng.choices("ACGT", k=12))
        reads.append(read(f"r{i}", pos, seq, umi=umi, cell=rng.choice(CELLS)))
    site_df = sites(*[("1", p, "A", "G") for p in range(100, 20_000, 37)])
    _, small = run(tmp_path, reads, site_df, block=block, chunk_bytes=chunk)
    _, big = run(tmp_path, reads, site_df)
    assert small == big
    # reference: a direct count on the host
    exp = {}
    kept = sorted({p for p in range(100, 20_000, 37) if any(r.pos < p <= r.pos + 60 for r in reads)})
    index = {p: i for i, p in enumerate(kept)}
    for r in reads:
        for p in range(r.pos + 1, r.pos + 61):
            if p in index:
                b = r.seq[p - 1 - r.pos]
                cell = CELLS.index(dict((t[0], t[2]) for t in r.tags)["CB"])
                ad, dp, oth = exp.get((index[p], cell), (0, 0, 0))
                exp[(index[p], cell)] = (ad + (b == "G"), dp + (b in "AG"), oth + (b not in "AG"))
    exp = {k: v for k, v in exp.items() if any(v)}
    assert big == exp
