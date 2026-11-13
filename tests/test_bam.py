import random

import cupy as cp
import numpy as np
import pytest
from bamwriter import Read, sort_key, write_bam

from cubam import BamHeader, read_bam

REFS = [("chr1", 1_000_000), ("chr2", 500_000), ("chrM", 16_569)]


def _random_reads(n: int, seed: int = 0) -> list[Read]:
    rng = random.Random(seed)
    reads = []
    for i in range(n):
        ref = rng.randrange(len(REFS))
        cl, m, cr = rng.choice([0, 0, 3]), rng.randint(20, 150), rng.choice([0, 0, 7])
        cigar = (f"{cl}S" if cl else "") + f"{m // 2}M2D{m - m // 2}M" + (f"{cr}S" if cr else "")
        tags = [("NM", "i", rng.randint(0, 5)), ("XS", "C", rng.randint(0, 255))]
        if rng.random() < 0.9:
            tags.append(("CB", "Z", "".join(rng.choices("ACGT", k=16)) + "-1"))
        if rng.random() < 0.3:
            tags.insert(0, ("ZB", "B", ("s", [rng.randint(-100, 100) for _ in range(rng.randint(0, 9))])))
        tags.append(("RG", "Z", f"group{rng.randint(0, 3)}"))
        reads.append(
            Read(
                name=f"read{i}:{rng.randint(0, 10**6)}",
                flag=rng.choice([99, 147, 83, 163, 1123, 0, 16]),
                ref=ref,
                pos=rng.randrange(REFS[ref][1] - 500),
                mapq=rng.randint(0, 60),
                cigar=cigar,
                nref=ref,
                npos=rng.randrange(REFS[ref][1]),
                tlen=rng.randint(-1000, 1000),
                tags=tags,
            )
        )
    return sorted(reads, key=sort_key)


def _collect(path, **kw):
    cols, tags = {k: [] for k in ("ref", "pos", "end", "mapq", "flag", "nref", "npos", "tlen", "l_seq")}, []
    names, nm, cb, five, clips = [], [], [], [], []
    for b in read_bam(path, **kw):
        for k in cols:
            cols[k].append(getattr(b, k).get())
        names += b.names().to_arrow().to_pylist()
        t = b.tag("NM")
        nm.append(np.where(t.present.get(), t.value.get(), -1))
        cb += b.tag("CB").strings().to_arrow().to_pylist()
        five.append(b.five_prime().get())
        clips.append(np.stack([b.clip_left.get(), b.clip_right.get()], 1))
        tags.append(b.tag("RG").strings().to_arrow().to_pylist())
    out = {k: np.concatenate(v) for k, v in cols.items()}
    return out, names, np.concatenate(nm), cb, np.concatenate(five), np.concatenate(clips), sum(tags, [])


@pytest.mark.parametrize(("block", "chunk"), [(4096, 1 << 20), (700, 5000), (65280, 64 << 10)])
def test_fields(tmp_path, block, chunk):
    reads = _random_reads(3000)
    p = write_bam(tmp_path / "r.bam", REFS, reads, block=block)
    cols, names, nm, cb, five, clips, rg = _collect(p, chunk_bytes=chunk, piece=chunk // 2 + 1)
    assert len(names) == len(reads)
    assert names == [r.name for r in reads]
    for k in ("ref", "pos", "mapq", "flag", "nref", "npos", "tlen", "end"):
        assert cols[k].tolist() == [getattr(r, k) for r in reads], k
    assert cols["l_seq"].tolist() == [len(r.seq) for r in reads]
    assert nm.tolist() == [dict((t[0], t[2]) for t in r.tags)["NM"] for r in reads]
    assert cb == [dict((t[0], t[2]) for t in r.tags).get("CB", "") for r in reads]
    assert rg == [dict((t[0], t[2]) for t in r.tags)["RG"] for r in reads]
    exp5 = []
    for r in reads:
        ops = [op for op in r.cigar if not op.isdigit()]
        left = int(r.cigar.split("S")[0]) if ops[0] == "S" else 0
        right = int(r.cigar[:-1].rsplit("M", 1)[1]) if ops[-1] == "S" else 0
        assert tuple(clips[len(exp5)]) == (left, right)
        exp5.append(r.end + right if r.flag & 16 else r.pos - left)
    assert five.tolist() == exp5


def test_long_reads_cross_blocks(tmp_path):
    reads = [Read(f"long{i}", 0, 0, 1000 * i, 60, "20000M", -1, -1, 0) for i in range(20)]
    p = write_bam(tmp_path / "l.bam", REFS, reads, block=4096)
    got = [int(x) for b in read_bam(p, chunk_bytes=30_000) for x in b.end.get()]
    assert got == [r.end for r in reads]


def test_header(tmp_path):
    text = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:1000000\n@PG\tID:bwa\tPN:bwa\n"
    p = write_bam(tmp_path / "h.bam", REFS, [], text=text)
    h = BamHeader.read(p)
    assert h.names == [n for n, _ in REFS]
    assert h.lengths == [ln for _, ln in REFS]
    assert h.sort_order == "coordinate"
    assert h.lines("@PG") == ["@PG\tID:bwa\tPN:bwa"]
    assert list(read_bam(p)) == []


def test_mates_share_name_hash(tmp_path):
    reads = [Read(f"pair{i}", 99, 0, 100 * i, 60, "50M", 0, 100 * i + 200, 250) for i in range(100)]
    reads += [Read(f"pair{i}", 147, 0, 100 * i + 200, 60, "50M", 0, 100 * i, -250) for i in range(100)]
    reads.sort(key=sort_key)
    p = write_bam(tmp_path / "p.bam", REFS, reads)
    (b,) = read_bam(p)
    h = b.name_hash().get()
    names = b.names().to_arrow().to_pylist()
    by_name = {}
    for name, x in zip(names, h, strict=True):
        by_name.setdefault(name, set()).add(int(x))
    assert all(len(v) == 1 for v in by_name.values())
    assert len({next(iter(v)) for v in by_name.values()}) == 100


def test_take_and_flags(tmp_path):
    reads = _random_reads(500, seed=3)
    p = write_bam(tmp_path / "t.bam", REFS, reads)
    (b,) = read_bam(p)
    sel = b.has_flags(all_of=1, none_of=0x400)
    sub = b.take(sel)
    exp = [r for r in reads if r.flag & 1 and not r.flag & 0x400]
    assert len(sub) == len(exp)
    assert sub.pos.get().tolist() == [r.pos for r in exp]
    assert sub.tag("NM").value.get().tolist() == [dict((t[0], t[2]) for t in r.tags)["NM"] for r in exp]
    assert isinstance(sub.data, cp.ndarray)
