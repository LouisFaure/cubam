import gzip
import random
import struct

import pytest
from bamwriter import EOF_BLOCK, Read, sort_key, write_bam

from cubam import read_bam
from cubam.apps.split import split_bam
from cubam.bam import BamHeader
from cubam.cells import CellIndex
from cubam.filters import ReadFilter

REFS = [("chr1", 100_000), ("chr2", 50_000)]
CELLS = ["AAAACCCCGGGGTTTT-1", "CCCCGGGGTTTTAAAA-1", "GGGGTTTTAAAACCCC-1", "TTTTAAAACCCCGGGG-1"]
LABELS = ["T cells", "B/plasma", "T cells", None]


def records_of(path) -> tuple[str, bytes]:
    """SAM header text and record bytes of a BAM file, read with gzip."""
    raw = gzip.decompress(path.read_bytes())
    (lt,) = struct.unpack("<i", raw[4:8])
    text = raw[8 : 8 + lt].decode()
    o = 8 + lt
    (nref,) = struct.unpack("<i", raw[o : o + 4])
    o += 4
    for _ in range(nref):
        (ln,) = struct.unpack("<i", raw[o : o + 4])
        o += 8 + ln
    return text, raw[o:]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    rng = random.Random(4)
    reads = []
    for i in range(4000):
        ref = rng.choice([0, 0, 1, -1])
        cb = rng.choice(CELLS + ["ACGTACGTACGTACGT-1", None])
        tags = [("CB", "Z", cb)] if cb else []
        if ref < 0:
            reads.append(Read(f"u{i}", 4, -1, -1, 0, "", -1, -1, 0, seq="ACGT" * 10, tags=tags))
        else:
            pos = rng.randrange(0, REFS[ref][1] - 200)
            cigar = f"{rng.randint(20, 150)}M"
            reads.append(Read(f"r{i}", 0, ref, pos, rng.choice([0, 60]), cigar, -1, -1, 0, tags=tags))
    reads = sorted(reads, key=sort_key)
    p = write_bam(tmp_path_factory.mktemp("s") / "in.bam", REFS, reads, block=3000)
    return reads, p


def expect(reads, keep):
    return b"".join(r.encode() for r in reads if keep(r))


def cb(r):
    return dict((t[0], t[2]) for t in r.tags).get("CB")


@pytest.mark.parametrize("flush", [1 << 30, 50_000])
def test_split_by_group(data, tmp_path, flush):
    reads, p = data
    res = split_bam(p, CellIndex(CELLS, labels=LABELS), tmp_path / "out", flush_bytes=flush, chunk_bytes=20_000)
    assert sorted(x.name for x in res.paths.values()) == ["B_plasma.bam", "T_cells.bam"]
    for label in ("T cells", "B/plasma"):
        mine = {c for c, lab in zip(CELLS, LABELS, strict=True) if lab == label}
        text, recs = records_of(res.paths[label])
        assert recs == expect(reads, lambda r, mine=mine: cb(r) in mine)
        assert text.startswith("@HD\tVN:1.6\tSO:coordinate\n") and "@PG\tID:cubam.split_bam" in text
        assert res.records[label] == sum(cb(r) in mine for r in reads)
        # cubam reads its own output, with the header of the input
        h = BamHeader.read(res.paths[label])
        assert h.names == [n for n, _ in REFS]
        assert sum(len(b) for b in read_bam(res.paths[label])) == res.records[label]
        assert res.paths[label].read_bytes().endswith(EOF_BLOCK)


def test_split_by_cell_with_filter(data, tmp_path):
    reads, p = data
    res = split_bam(p, CellIndex(CELLS), tmp_path / "cells", read_filter=ReadFilter(min_mapq=30), prefix="s1_")
    assert len(res.paths) == 4
    for c in CELLS:
        _, recs = records_of(res.paths[c])
        assert recs == expect(reads, lambda r, c=c: cb(r) == c and r.mapq >= 30 and r.ref >= 0)


def test_empty_group_is_a_valid_bam(tmp_path):
    p = write_bam(tmp_path / "in.bam", REFS, [])
    res = split_bam(p, CellIndex(CELLS, labels=LABELS), tmp_path / "o")
    for path in res.paths.values():
        text, recs = records_of(path)
        assert recs == b"" and "@PG" in text
