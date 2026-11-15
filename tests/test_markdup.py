import importlib.util
import random
import re
import shutil
import subprocess

import pytest
from bamwriter import Read, sort_key, write_bam

pytestmark = pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="needs cupy and a GPU")

REFS = [("chr1", 100_000), ("chr2", 50_000)]
DUP = 0x400


def flags_of(path):
    """(name, flag) of every record, in file order."""
    from cubam import read_bam

    out = []
    for b in read_bam(path):
        names = b.names().to_arrow().to_pylist()
        out += list(zip(names, b.flag.get().tolist(), strict=True))
    return out


def pair(name, ref, pos1, cig1, pos2, cig2, qual=30, cb=None, rev1=False, mref=None):
    """Read 1 at pos1 and read 2 at pos2, FR orientation by default, with MC tags."""
    mref = ref if mref is None else mref
    f1 = 0x1 | 0x40 | (0x10 if rev1 else 0x20)
    f2 = 0x1 | 0x80 | (0x20 if rev1 else 0x10)
    tags1 = [("MC", "Z", cig2)] + ([("CB", "Z", cb)] if cb else [])
    tags2 = [("MC", "Z", cig1)] + ([("CB", "Z", cb)] if cb else [])
    n1 = sum(int(x) for x in re.findall(r"(\d+)[MIS=X]", cig1))
    n2 = sum(int(x) for x in re.findall(r"(\d+)[MIS=X]", cig2))
    q1 = bytes([qual] * n1) if isinstance(qual, int) else qual[0]
    q2 = bytes([qual] * n2) if isinstance(qual, int) else qual[1]
    r1 = Read(name, f1, ref, pos1, 60, cig1, mref, pos2, 0, qual=q1, tags=tags1)
    r2 = Read(name, f2, mref, pos2, 60, cig2, ref, pos1, 0, qual=q2, tags=tags2)
    return [r1, r2]


def run(tmp_path, reads, barcode_tag="CB"):
    from cubam.apps.markdup import mark_duplicates

    src = write_bam(tmp_path / "in.bam", REFS, sorted(reads, key=sort_key), block=2000)
    res = mark_duplicates(src, tmp_path / "out.bam", barcode_tag=barcode_tag)
    return res, {(n, f & 0xC0): f & DUP for n, f in flags_of(tmp_path / "out.bam")}


def test_best_pair_stays(tmp_path):
    reads = (
        pair("a", 0, 1000, "50M", 1200, "50M", qual=30, cb="AAAA-1")
        + pair("b", 0, 1000, "50M", 1200, "50M", qual=35, cb="AAAA-1")  # highest score
        + pair("c", 0, 1000, "50M", 1200, "50M", qual=20, cb="AAAA-1")
    )
    res, f = run(tmp_path, reads)
    assert {n for (n, _), d in f.items() if d} == {"a", "c"}
    assert res.stats["DUPLICATE PAIR"] == 4
    assert res.stats["DUPLICATE TOTAL"] == 4


def test_ties_go_to_the_smallest_name(tmp_path):
    reads = pair("y", 0, 1000, "50M", 1200, "50M") + pair("x", 0, 1000, "50M", 1200, "50M")
    _, f = run(tmp_path, reads)
    assert {n for (n, _), d in f.items() if d} == {"y"}


def test_barcodes_separate_duplicates(tmp_path):
    reads = pair("a", 0, 1000, "50M", 1200, "50M", cb="AAAA-1") + pair("b", 0, 1000, "50M", 1200, "50M", cb="CCCC-1")
    _, f = run(tmp_path, reads)
    assert not any(f.values())
    _, f = run(tmp_path, reads, barcode_tag=None)
    assert sum(1 for d in f.values() if d) == 2


def test_soft_clips_use_unclipped_coordinates(tmp_path):
    # same unclipped 5' ends: 1000 - 0 and 1005 - 5; the mates end at 1250 either way
    reads = pair("a", 0, 1000, "50M", 1200, "50M", qual=30) + pair("b", 0, 1005, "5S45M", 1200, "50M", qual=20)
    _, f = run(tmp_path, reads)
    assert {n for (n, _), d in f.items() if d} == {"b"}


def test_single_loses_to_a_pair(tmp_path):
    single = Read("s", 0x1 | 0x8 | 0x40, 0, 1000, 60, "50M", 0, 1000, 0, qual=bytes([40] * 50))
    mate = Read("s", 0x1 | 0x4 | 0x80, 0, 1000, 0, "", 0, 1000, 0, seq="A" * 50, qual=bytes([40] * 50))
    reads = pair("p", 0, 1000, "50M", 1200, "50M", qual=10) + [single, mate]
    res, f = run(tmp_path, reads, barcode_tag=None)
    assert f[("s", 0x40)] and not f[("p", 0x40)] and not f[("p", 0x80)]
    assert res.stats["DUPLICATE SINGLE"] == 1


def test_output_is_not_the_input(tmp_path):
    from cubam.apps.markdup import mark_duplicates

    src = write_bam(tmp_path / "in.bam", REFS, sorted(pair("a", 0, 1000, "50M", 1200, "50M"), key=sort_key))
    size = src.stat().st_size
    with pytest.raises(ValueError, match="refers to the input"):
        mark_duplicates(src, src)
    assert src.stat().st_size == size


def test_long_clip_reaches_back_across_batches(tmp_path):
    from cubam.apps.markdup import mark_duplicates

    # "b" starts 100 bases after "a" but has the same unclipped 5' end; the reads
    # between them fill several batches
    reads = pair("a", 0, 1000, "50M", 1300, "50M", qual=20) + pair("b", 0, 1100, "100S50M", 1300, "50M", qual=30)
    for i in range(400):
        reads += pair(f"f{i:03d}", 0, 1001 + i // 4, "50M", 2000 + i * 60, "50M")
    src = write_bam(tmp_path / "in.bam", REFS, sorted(reads, key=sort_key), block=2000)
    res = mark_duplicates(src, tmp_path / "out.bam", barcode_tag=None, chunk_bytes=4000)
    dups = {n for n, f in flags_of(tmp_path / "out.bam") if f & DUP}
    assert "a" in dups and "b" not in dups
    assert res.stats["DUPLICATE PAIR"] == 2  # both reads of "a"


def _random_reads(seed=7, n=3000):
    rng = random.Random(seed)
    cells = ["AAAACCCCGGGGTTTT-1", "CCCCGGGGTTTTAAAA-1", None]
    starts = [rng.randrange(1000, 90_000) for _ in range(120)]
    cigars = ["50M", "3S47M", "47M3S", "2S46M2S", "20M1D30M", "25M2I23M"]
    reads = []
    for i in range(n):
        ref = rng.choice([0, 0, 1])
        p1 = rng.choice(starts) % (REFS[ref][1] - 1000)
        cb = rng.choice(cells)
        c1, c2 = rng.choice(cigars), rng.choice(cigars)
        quals = [bytes(rng.choice([2, 11, 25, 37]) for _ in range(50)) for _ in range(2)]
        kind = rng.random()
        if kind < 0.08:  # mate unmapped
            f = 0x1 | 0x8 | 0x40 | (0x10 if rng.random() < 0.5 else 0)
            tags = [("CB", "Z", cb)] if cb else []
            reads.append(Read(f"m{i}", f, ref, p1, 60, c1, ref, p1, 0, qual=quals[0], tags=tags))
            reads.append(
                Read(f"m{i}", 0x1 | 0x4 | 0x80, ref, p1, 0, "", ref, p1, 0, seq="A" * 50, qual=quals[1], tags=tags)
            )
            continue
        mref = ref if kind < 0.95 else 1 - ref  # some mates on the other reference
        p2 = p1 + rng.choice([100, 150, 300]) if mref == ref else rng.randrange(1000, 40_000)
        rp = pair(f"p{i}", ref, p1, c1, p2, c2, qual=quals, cb=cb, rev1=rng.random() < 0.3, mref=mref)
        if rng.random() < 0.03:  # a secondary alignment
            s = Read(
                f"p{i}",
                rp[0].flag | 0x100,
                ref,
                p1 + 7,
                0,
                c1,
                rp[0].nref,
                rp[0].npos,
                0,
                qual=quals[0],
                tags=rp[0].tags,
            )
            rp.append(s)
        reads += rp
    return reads


def test_cli(tmp_path):
    import json

    from cubam.cli import main

    reads = pair("a", 0, 1000, "50M", 1200, "50M", qual=30, cb="AAAA-1") + pair(
        "b", 0, 1000, "50M", 1200, "50M", cb="AAAA-1", qual=20
    )
    src = write_bam(tmp_path / "in.bam", REFS, sorted(reads, key=sort_key))
    out, stats = tmp_path / "out.bam", tmp_path / "stats.json"
    assert main(["markdup", str(src), "--barcode-tag", "CB", "-o", str(out), "-f", str(stats)]) == 0
    assert json.loads(stats.read_text())["DUPLICATE PAIR"] == 2
    assert {n for n, f in flags_of(out) if f & DUP} == {"b"}
    text = tmp_path / "stats.txt"
    assert main(["markdup", str(src), "-o", str(tmp_path / "out2.bam"), "-f", str(text)]) == 0
    assert "DUPLICATE TOTAL: 2" in text.read_text()


@pytest.mark.skipif(shutil.which("samtools") is None, reason="needs samtools")
@pytest.mark.parametrize("barcode_tag", ["CB", None])
def test_same_flags_as_samtools(tmp_path, barcode_tag):
    from cubam.apps.markdup import duplicate_mask, mark_duplicates

    src = write_bam(tmp_path / "raw.bam", REFS, sorted(_random_reads(), key=sort_key))
    fm = tmp_path / "fixmate.bam"
    subprocess.run(
        f"samtools collate -O -u {src} {tmp_path}/c | samtools fixmate -m -u - - | samtools sort -o {fm} -",
        shell=True,
        check=True,
    )
    bc = ["--barcode-tag", barcode_tag] if barcode_tag else []
    subprocess.run(["samtools", "markdup", *bc, str(fm), str(tmp_path / "samtools.bam")], check=True)
    res = mark_duplicates(fm, tmp_path / "cubam.bam", barcode_tag=barcode_tag)
    ours, theirs = flags_of(tmp_path / "cubam.bam"), flags_of(tmp_path / "samtools.bam")
    assert ours == theirs
    assert res.stats["DUPLICATE TOTAL"] == sum(1 for _, f in theirs if f & DUP) > 100
    # on the file marked by samtools: the flags samtools set, ignoring them, in batches of any size
    for chunk in (None, 3000, 20_000):
        mask, res = duplicate_mask(tmp_path / "samtools.bam", barcode_tag=barcode_tag, chunk_bytes=chunk)
        assert mask.tolist() == [bool(f & DUP) for _, f in theirs]
