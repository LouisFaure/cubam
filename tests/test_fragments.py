import random

import pytest
from bamwriter import Read, sort_key, write_bam

from cubam.apps.fragments import call_fragments

REFS = [("chr1", 1_000_000), ("chrM", 16_569), ("chrUn", 50_000)]
BC = "ACGTACGTACGTACGT-1"


def pair(
    name,
    fwd_pos,
    rev_pos,
    *,
    bc=BC,
    ref=0,
    fwd_cigar="50M",
    rev_cigar="50M",
    mapq=(60, 60),
    dup=False,
    rev_first=False,
    same_strand=False,
    mate_ref=None,
):
    """A read pair: forward read at ``fwd_pos``, reverse read at ``rev_pos``."""
    d = 0x400 if dup else 0
    tags = [("CB", "Z", bc)] if bc else []
    mref = ref if mate_ref is None else mate_ref
    f = Read(name, 0x1 | 0x20 | 0x40 | d, ref, fwd_pos, mapq[0], fwd_cigar, mref, rev_pos, 0, tags=tags)
    r = Read(
        name,
        0x1 | (0 if same_strand else 0x10) | 0x80 | d,
        mref,
        rev_pos,
        mapq[1],
        rev_cigar,
        ref,
        fwd_pos,
        0,
        tags=tags,
    )
    return [r, f] if rev_first else [f, r]


def call(tmp_path, reads, **kw):
    reads = sorted(reads, key=sort_key)
    p = write_bam(tmp_path / "f.bam", REFS, reads, block=kw.pop("block", 4096))
    fr = call_fragments(p, chunk_bytes=kw.pop("chunk_bytes", 1 << 20), **kw)
    df = fr.to_frame().to_pandas()
    return fr, {(r.chrom, r.start, r.end, r.barcode): r.count for r in df.itertuples()}


def test_basic_and_soft_clips(tmp_path):
    reads = pair("a", 1000, 1150) + pair("b", 2003, 2150, fwd_cigar="3S47M", rev_cigar="47M3S")
    _, got = call(tmp_path, reads)
    # forward 5' end + 4, reverse 5' end - 5 (soft clips included)
    assert got == {("chr1", 1004, 1195, BC): 1, ("chr1", 2004, 2195, BC): 1}


def test_mapq_above_30(tmp_path):
    reads = pair("a", 1000, 1150, mapq=(60, 30)) + pair("b", 3000, 3150, mapq=(31, 31))
    _, got = call(tmp_path, reads)
    assert got == {("chr1", 3004, 3195, BC): 1}


def test_mate_distance(tmp_path):
    reads = pair("a", 10_000, 15_000) + pair("b", 30_000, 35_001)
    _, got = call(tmp_path, reads)
    assert got == {("chr1", 10_004, 15_045, BC): 1}
    _, got = call(tmp_path, reads, max_mate_distance=None)
    assert len(got) == 2


def test_outward_and_same_strand(tmp_path):
    reads = (
        # outward: the 5' end of the reverse read lies before the 5' end of the forward read
        pair("out", 20_100, 20_000, rev_first=True)
        # reverse read aligned 1 bp left of the forward read: kept (Cell Ranger ARC 2)
        + pair("short", 30_001, 30_000, rev_first=True)
        + pair("same", 40_000, 40_150, same_strand=True)
        + pair("chim", 50_000, 100, mate_ref=2)
    )
    _, got = call(tmp_path, reads)
    assert got == {("chr1", 30_005, 30_045, BC): 1}


def test_duplicates_and_representative(tmp_path):
    reads = (
        pair("a1", 1000, 1150)
        + pair("a2", 1000, 1150, dup=True)
        + pair("a3", 1000, 1150, bc="TTTTACGTACGTACGT-1")  # its own group
        # representative fails MAPQ, the duplicate passes: the group is dropped
        + pair("b1", 5000, 5150, mapq=(60, 30))
        + pair("b2", 5000, 5150, dup=True)
    )
    fr, got = call(tmp_path, reads)
    assert got == {("chr1", 1004, 1195, BC): 2, ("chr1", 1004, 1195, "TTTTACGTACGTACGT-1"): 1}
    assert fr.stats["no_representative"] == 1


def test_contigs_and_edges(tmp_path):
    reads = (
        pair("m", 100, 250, ref=1)
        + pair("u", 100, 250, ref=2)
        + pair("start", 0, 150, fwd_cigar="2S48M")  # 5' end before the contig start
        + pair("end", 999_800, 999_950, rev_cigar="50M10S")  # end past the contig end
    )
    _, got = call(tmp_path, reads, keep_chroms=["chr1", "chrUn"], drop_chroms=("chrUn",))
    assert got == {("chr1", 999_804, 1_000_000, BC): 1}


def test_no_barcode(tmp_path):
    reads = pair("a", 1000, 1150, bc=None) + pair("b", 2000, 2150)
    fr, got = call(tmp_path, reads)
    assert got == {("chr1", 2004, 2195, BC): 1}
    assert fr.stats["no_barcode"] == 2


def test_not_coordinate_sorted(tmp_path):
    p = write_bam(tmp_path / "u.bam", REFS, [], text="@HD\tVN:1.6\tSO:queryname\n")
    with pytest.raises(ValueError, match="coordinate"):
        call_fragments(p)


@pytest.mark.parametrize(("block", "chunk"), [(4096, 1 << 20), (500, 3000)])
def test_many_chunks(tmp_path, block, chunk):
    rng = random.Random(1)
    reads, expected = [], {}
    barcodes = ["".join(rng.choices("ACGT", k=16)) + "-1" for _ in range(20)]
    for i in range(3000):
        start = rng.randrange(0, 900_000)
        length = rng.randint(30, 800)
        bc = rng.choice(barcodes)
        dups = rng.choice([1, 1, 1, 2, 3])
        for d in range(dups):
            reads += pair(f"p{i}_{d}", start, start + length - 50, bc=bc, dup=d > 0)
        key = ("chr1", start + 4, start + length - 5, bc)
        expected[key] = expected.get(key, 0) + dups
    fr, got = call(tmp_path, reads, block=block, chunk_bytes=chunk)
    assert got == expected
    t = fr.table.to_pandas()
    assert (t["start"].diff().dropna() >= 0).all()
