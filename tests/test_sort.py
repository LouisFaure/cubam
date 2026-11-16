import functools
import random

import pytest
from bamwriter import EOF_BLOCK, Read, write_bam
from test_split import records_of

from cubam import read_bam
from cubam.apps.sort import _hd_line, sort_bam

REFS = [("chr1", 100_000), ("chr2", 50_000), ("chrM", 16_569)]


def strnum_cmp(a: str, b: str) -> int:
    """strnum_cmp of samtools 1.24 (bam_sort.c), line for line."""
    a, b = a.encode() + b"\0", b.encode() + b"\0"
    i = j = 0

    def dig(c):
        return 48 <= c <= 57

    while a[i] and b[j]:
        if not dig(a[i]) or not dig(b[j]):
            if a[i] != b[j]:
                return a[i] - b[j]
            i, j = i + 1, j + 1
        else:
            while a[i] == 48:
                i += 1
            while b[j] == 48:
                j += 1
            while dig(a[i]) and a[i] == b[j]:
                i, j = i + 1, j + 1
            diff = a[i] - b[j]
            while dig(a[i]) and dig(b[j]):
                i, j = i + 1, j + 1
            if dig(a[i]):
                return 1
            if dig(b[j]):
                return -1
            if diff:
                return diff
    return 1 if a[i] else -1 if b[j] else 0


def coordinate_key(r: Read):
    return (r.ref & 0xFFFFFFFF, r.pos + 1, (r.flag >> 4) & 1)


def flag_key(r: Read) -> int:
    f = r.flag
    return ((f & 0xC0) << 8) | ((f & 0x100) << 3) | ((f & 0x800) >> 3)


def name_cmp(natural: bool):
    def cmp(x: Read, y: Read) -> int:
        t = strnum_cmp(x.name, y.name) if natural else (x.name > y.name) - (x.name < y.name)
        return t or flag_key(x) - flag_key(y)

    return functools.cmp_to_key(cmp)


NAMES = ["a7b", "a12b", "a007b", "a0", "a00", "a", "a00x", "a0b", "r1:10", "r1:9", "r01:9", "x9", "x10", "B2", "b2"]


@pytest.fixture(scope="module")
def reads():
    """Unsorted reads with many ties: few positions, both strands, unmapped and placed-unmapped reads."""
    rng = random.Random(7)
    out = []
    for i in range(3000):
        name = rng.choice(NAMES) if rng.random() < 0.3 else f"q{rng.randrange(40)}:{rng.randrange(0, 120, 7):03d}"
        flag = rng.choice([0x1 | 0x40, 0x1 | 0x80, 0, 0x100, 0x800]) | rng.choice([0, 0x10])
        tags = [("XI", "i", i)]  # so that records that tie still differ
        kind = rng.random()
        if kind < 0.1:
            out.append(Read(name, flag | 0x4, -1, -1, 0, "", seq="ACGT", tags=tags))
        else:
            ref = rng.randrange(len(REFS))
            pos = rng.choice([0, 5, 99, 100, rng.randrange(15_000)])
            if kind < 0.15:  # unmapped, placed at its mate
                out.append(Read(name, flag | 0x4, ref, pos, 0, "", seq="ACGT", tags=tags))
            else:
                out.append(Read(name, flag, ref, pos, 60, f"{rng.randint(10, 60)}M", tags=tags))
    return out


@pytest.fixture(scope="module")
def unsorted(reads, tmp_path_factory):
    text = "@HD\tVN:1.6\tSO:unsorted\tGO:query\n" + "".join(f"@SQ\tSN:{n}\tLN:{ln}\n" for n, ln in REFS)
    text += "@PG\tID:bwa\tPN:bwa\n"
    return write_bam(tmp_path_factory.mktemp("sort") / "in.bam", REFS, reads, block=3000, text=text)


@pytest.mark.parametrize("small", [False, True], ids=["one_run", "merge"])
@pytest.mark.parametrize(
    "by,natural,key",
    [
        ("coordinate", True, coordinate_key),
        ("name", True, name_cmp(True)),
        ("name", False, name_cmp(False)),
    ],
    ids=["coordinate", "name", "name_ascii"],
)
def test_sort_order(reads, unsorted, tmp_path, by, natural, key, small):
    kw = {"run_bytes": 40_000, "chunk_bytes": 8_000} if small else {}
    res = sort_bam(unsorted, tmp_path / "out.bam", by=by, natural=natural, **kw)
    text, recs = records_of(res.path)
    assert recs == b"".join(r.encode() for r in sorted(reads, key=key))
    assert res.records == len(reads)
    assert (res.runs > 3) == small
    assert res.path.read_bytes().endswith(EOF_BLOCK)
    assert sum(len(b) for b in read_bam(res.path)) == len(reads)
    hd = text.split("\n")[0]
    if by == "coordinate":
        assert hd == "@HD\tVN:1.6\tSO:coordinate"
    else:
        assert hd == f"@HD\tVN:1.6\tSO:queryname\tSS:queryname:{'natural' if natural else 'lexicographical'}"
    assert "\n@PG\tID:cubam.sort_bam\tPN:cubam\tPP:bwa\tCL:" in text
    assert not list(tmp_path.glob(".out.bam.*"))  # the runs are removed


def test_natural_names(tmp_path):
    names = [*NAMES, "a1", "a01", "a1b", "a10", "a9", "0", "00", "1", "z", "a99999999999999999999", "a100000000000"]
    names = sorted(names, key=functools.cmp_to_key(strnum_cmp), reverse=True)
    reads = [Read(n, 0, 0, 10, 60, "10M", tags=[("XI", "i", i)]) for i, n in enumerate(names)]
    p = write_bam(tmp_path / "n.bam", REFS, reads)
    res = sort_bam(p, tmp_path / "out.bam", by="name")
    _, recs = records_of(res.path)
    assert recs == b"".join(r.encode() for r in sorted(reads, key=name_cmp(True)))


def test_empty_bam(tmp_path):
    p = write_bam(tmp_path / "e.bam", REFS, [])
    res = sort_bam(p, tmp_path / "out.bam")
    text, recs = records_of(res.path)
    assert recs == b"" and text.startswith("@HD\tVN:1.6\tSO:coordinate\n@SQ")


def test_rejects_input_as_output(unsorted):
    with pytest.raises(ValueError, match="input BAM"):
        sort_bam(unsorted, unsorted)


def test_hd_line():
    assert _hd_line("@SQ\tSN:a\tLN:1\n", "coordinate", True) == "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:a\tLN:1\n"
    got = _hd_line("@HD\tVN:1.4\tGO:query\tSO:unsorted\tXY:z\n", "name", False)
    assert got == "@HD\tVN:1.4\tSO:queryname\tXY:z\tSS:queryname:lexicographical\n"
    got = _hd_line("@HD\tVN:1.6\tSO:queryname\tSS:queryname:natural\n", "coordinate", True)
    assert got == "@HD\tVN:1.6\tSO:coordinate\n"
