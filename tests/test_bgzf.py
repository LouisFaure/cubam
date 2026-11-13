import random

import cupy as cp
import pytest
from bamwriter import EOF_BLOCK, bgzf

from cubam import inflate_chunks


def _data(n: int, seed: int = 0) -> bytes:
    rng = random.Random(seed)
    words = [bytes(rng.choices(b"ACGT\t\n0123456789", k=rng.randint(1, 30))) for _ in range(500)]
    out = bytearray()
    while len(out) < n:
        out += rng.choice(words)
    return bytes(out[:n])


def _read(path, **kw) -> bytes:
    return b"".join(c.get().tobytes() for c in inflate_chunks(path, **kw))


@pytest.mark.parametrize(("block", "chunk"), [(1000, 4096), (65280, 1 << 20), (300, 777)])
def test_roundtrip(tmp_path, block, chunk):
    data = _data(300_000)
    p = tmp_path / "x.gz"
    p.write_bytes(bgzf(data, block))
    assert _read(p, chunk_bytes=chunk, piece=chunk // 3 + 1) == data


def test_empty_file(tmp_path):
    p = tmp_path / "e.gz"
    p.write_bytes(EOF_BLOCK)
    assert _read(p, chunk_bytes=4096) == b""


def test_truncated(tmp_path):
    p = tmp_path / "t.gz"
    p.write_bytes(bgzf(_data(50_000), 1000, eof=False)[:-100])
    with pytest.raises(ValueError, match="Truncated"):
        _read(p, chunk_bytes=8192)


@pytest.mark.parametrize("where", ["btype", "isize"])
def test_corrupt_block(tmp_path, where):
    raw = bytearray(bgzf(_data(50_000), 1000))
    bsize = raw[16] + (raw[17] << 8) + 1
    if where == "btype":
        raw[18] |= 0b110  # invalid Deflate block type
    else:
        raw[bsize - 4] ^= 0x01  # uncompressed size in the trailer
    p = tmp_path / "c.gz"
    p.write_bytes(bytes(raw))
    with pytest.raises((ValueError, RuntimeError)):
        _read(p, chunk_bytes=8192)


def test_not_bgzf(tmp_path):
    p = tmp_path / "n.gz"
    p.write_bytes(b"hello world, not compressed" * 100)
    with pytest.raises(ValueError):
        _read(p, chunk_bytes=4096)


def test_chunks_are_on_the_gpu(tmp_path):
    p = tmp_path / "x.gz"
    p.write_bytes(bgzf(_data(10_000), 1000))
    chunks = list(inflate_chunks(p, chunk_bytes=4096))
    assert all(isinstance(c, cp.ndarray) for c in chunks)
