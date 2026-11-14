import gzip
import random
import zlib

import cupy as cp
import numpy as np
import pytest
from bamwriter import EOF_BLOCK, bgzf

from cubam import inflate_chunks
from cubam.bgzf import BgzfWriter, adler32_spans, bgzf_compress, crc32_spans, deflate_spans


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


def _spans(n, step):
    off = np.arange(0, n, step)
    return off, np.minimum(off + step, n) - off


@pytest.mark.parametrize("level", [2, 5])
def test_deflate_crc_adler_match_zlib(level):
    rng = np.random.default_rng(level)
    text = (b"chr1\t12345\tACGTACGT\tCB:Z:AAAC\n" * 20000)[:400_000]
    raw = text + rng.integers(0, 256, 150_000, dtype=np.uint8).tobytes() + b"z"
    d = cp.asarray(np.frombuffer(raw, np.uint8))
    off, size = _spans(len(raw), 30_001)
    out, oo, osz = deflate_spans(d, off, size, level, max_chunks=4)  # several nvCOMP calls
    o, oo, osz = out.get(), oo.get(), osz.get()
    for a, b, s, n in zip(oo, osz, off, size, strict=True):
        assert zlib.decompress(o[a : a + b].tobytes(), -15) == raw[s : s + n]
    spans = list(zip(off, size, strict=True))
    assert crc32_spans(d, off, size).get().tolist() == [zlib.crc32(raw[s : s + n]) for s, n in spans]
    assert adler32_spans(d, off, size).get().tolist() == [zlib.adler32(raw[s : s + n]) for s, n in spans]


def test_bgzf_compress_and_writer(tmp_path):
    rng = np.random.default_rng(1)
    raw = (b"ACGT" * 50_000) + rng.integers(0, 256, 200_000, dtype=np.uint8).tobytes()  # random: stored blocks
    blocks = bgzf_compress(cp.asarray(np.frombuffer(raw, np.uint8))).get().tobytes()
    assert gzip.decompress(blocks + EOF_BLOCK) == raw
    p = tmp_path / "w.gz"
    with BgzfWriter(p, flush_bytes=100_000) as w:
        for k in range(0, len(raw), 77_777):
            w.write(cp.asarray(np.frombuffer(raw[k : k + 77_777], np.uint8)))
        w.write(b"end")
    assert gzip.decompress(p.read_bytes()) == raw + b"end"
    assert cp.concatenate(list(inflate_chunks(p, 50_000))).get().tobytes() == raw + b"end"
    with BgzfWriter(tmp_path / "e.gz"):
        pass
    assert (tmp_path / "e.gz").read_bytes() == EOF_BLOCK
