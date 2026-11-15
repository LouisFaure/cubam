"""
BGZF files read and inflated on the GPU.

BGZF (BAM, BCF, bgzip and tabix files) is a series of independent Deflate
blocks of at most 64 KiB of data, each with its compressed size in the header
and the uncompressed size in the trailer. :func:`inflate_chunks` reads the
file in chunks and returns the inflated data of each chunk on the GPU:

1. **Read.** A background thread reads the next chunks into pinned host
   memory with parallel ``pread`` calls (network file systems read about
   twice as fast with several requests in flight).
2. **Find blocks.** Every offset whose bytes match the fixed fields of a
   BGZF header, and whose block fits in the chunk, is a candidate. The chain
   from the first block is kept. The bytes of the incomplete last block stay
   on the GPU for the next chunk.
3. **Inflate.** One call of the batched Deflate function of the nvCOMP C
   library, with device arrays of pointers and sizes, inflates all blocks
   into one contiguous buffer.
"""

from __future__ import annotations

import ctypes
import os
import queue
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from cubam._kernels import append_offsets, chain, copy_spans, launch

_HEADER, _TRAILER = 18, 8
_MAX_ISIZE = 1 << 16


class _ChunkReader:
    """Chunks of a file read ahead into pinned buffers by a background thread.

    Each chunk is read by ``n_threads`` parallel ``pread`` calls of ``piece``
    bytes. Iterating yields ``(slot, nbytes, eof)``. The consumer copies
    :meth:`host` ``(slot)[:nbytes]`` and then calls :meth:`release` ``(slot)``.
    """

    def __init__(
        self,
        path: str | Path,
        chunk_bytes: int,
        *,
        n_buffers: int = 3,
        n_threads: int = 8,
        piece: int = 16 << 20,
    ):
        import cupy as cp

        self.path, self.chunk, self.n_threads, self.piece = Path(path), chunk_bytes, n_threads, piece
        self._mem = [cp.cuda.alloc_pinned_memory(chunk_bytes) for _ in range(n_buffers)]
        self._free: queue.Queue = queue.Queue()
        for i in range(n_buffers):
            self._free.put(i)
        self._full: queue.Queue = queue.Queue()
        self._error: BaseException | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def host(self, slot: int) -> np.ndarray:
        return np.frombuffer(self._mem[slot], np.uint8, self.chunk)

    def release(self, slot: int) -> None:
        self._free.put(slot)

    def close(self) -> None:
        self._stop = True
        self._free.put(-1)

    @staticmethod
    def _pread(fd: int, mv: memoryview, offset: int) -> None:
        while len(mv):
            got = os.preadv(fd, [mv], offset)
            if not got:
                raise ValueError("The file is shorter than its size")
            mv, offset = mv[got:], offset + got

    def _run(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        try:
            fd = os.open(self.path, os.O_RDONLY)
            try:
                size = os.fstat(fd).st_size
                off = 0
                with ThreadPoolExecutor(self.n_threads) as ex:
                    while True:
                        slot = self._free.get()
                        if self._stop:
                            return
                        n = min(self.chunk, size - off)
                        mv = memoryview(self.host(slot))
                        futures = [
                            ex.submit(self._pread, fd, mv[p : min(p + self.piece, n)], off + p)
                            for p in range(0, n, self.piece)
                        ]
                        for f in futures:
                            f.result()
                        off += n
                        eof = off >= size
                        self._full.put((slot, n, eof))
                        if eof:
                            return
            finally:
                os.close(fd)
        except BaseException as e:  # noqa: BLE001  (re-raised in the consumer)
            self._error = e
            self._full.put(None)

    def __iter__(self) -> Iterator[tuple[int, int, bool]]:
        while True:
            item = self._full.get()
            if item is None:
                raise self._error  # type: ignore[misc]
            yield item
            if item[2]:
                return


class _Opts(ctypes.Structure):
    _fields_ = [("backend", ctypes.c_int), ("sort", ctypes.c_int), ("reserved", ctypes.c_char * 56)]


class _CompressOpts(ctypes.Structure):
    _fields_ = [("algorithm", ctypes.c_int), ("reserved", ctypes.c_char * 60)]


class _Alignments(ctypes.Structure):
    _fields_ = [("input", ctypes.c_size_t), ("output", ctypes.c_size_t), ("temp", ctypes.c_size_t)]


_lib = None
_compress_ready = False
_decompress_ready = False


def _nvcomp_lib():
    """libnvcomp 5 of the nvidia-libnvcomp wheel."""
    global _lib
    if _lib is None:
        import nvidia.libnvcomp as m

        for d in m.__path__:
            f = os.path.join(d, "lib64", "libnvcomp.so.5")
            if os.path.exists(f):
                _lib = ctypes.CDLL(f)
                break
        else:
            raise ImportError("libnvcomp.so.5 not found (install nvidia-libnvcomp-cu12)")
    return _lib


def _nvcomp_decompress():
    """``nvcompBatchedDeflateDecompressAsync`` of libnvcomp 5."""
    global _decompress_ready
    lib = _nvcomp_lib()
    if not _decompress_ready:
        _decompress_ready = True
        fn = lib.nvcompBatchedDeflateDecompressAsync
        fn.restype = ctypes.c_int
        fn.argtypes = [ctypes.c_void_p] * 4 + [
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            _Opts,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
    return lib.nvcompBatchedDeflateDecompressAsync


def _nvcomp_compress_lib():
    """libnvcomp 5 with the argument types of its batched Deflate compression functions set."""
    global _compress_ready
    lib = _nvcomp_lib()
    if not _compress_ready:
        size_p = ctypes.POINTER(ctypes.c_size_t)
        lib.nvcompBatchedDeflateCompressGetRequiredAlignments.argtypes = [_CompressOpts, ctypes.POINTER(_Alignments)]
        lib.nvcompBatchedDeflateCompressGetTempSizeAsync.argtypes = [
            ctypes.c_size_t,
            ctypes.c_size_t,
            _CompressOpts,
            size_p,
            ctypes.c_size_t,
        ]
        lib.nvcompBatchedDeflateCompressGetMaxOutputChunkSize.argtypes = [ctypes.c_size_t, _CompressOpts, size_p]
        lib.nvcompBatchedDeflateCompressAsync.argtypes = [ctypes.c_void_p] * 2 + [
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_void_p,
            _CompressOpts,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        for fn in ("GetRequiredAlignments", "GetTempSizeAsync", "GetMaxOutputChunkSize", "Async"):
            getattr(lib, "nvcompBatchedDeflateCompress" + fn).restype = ctypes.c_int
        _compress_ready = True
    return lib


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"nvCOMP {what} failed (status {rc})")


def _deflate_spans(data, off, size, level: int = 1, max_chunks: int = 4096):
    """Raw Deflate streams of ``data[off[i] : off[i] + size[i]]`` (cupy), each at most 65,536 bytes.

    ``level`` is the nvCOMP Deflate algorithm, 0 (entropy coding only) to 5.
    On BAM data, level 1 runs at about 12 GB/s on an A100 and level 2 at 7 GB/s
    with files 17% larger than zlib level 6; level 5 gives files 9% smaller
    at 0.2 GB/s. Returns ``(out,
    out_off, out_size)``: the streams packed in one cupy buffer, with their
    offsets and sizes. nvCOMP sizes its buffers for the worst case (about
    twice the input), so chunks are compressed ``max_chunks`` at a time.
    """
    import cupy as cp

    off = cp.asarray(off, cp.int64)
    size = cp.asarray(size, cp.int64)
    m = int(size.size)
    if m <= max_chunks:
        out, out_off, out_size = _deflate_batch(data, off, size, level)
        return _pack(out, out_off, out_size)
    parts = [
        _pack(*_deflate_batch(data, off[k : k + max_chunks], size[k : k + max_chunks], level))
        for k in range(0, m, max_chunks)
    ]
    base = np.cumsum([0] + [int(p[0].size) for p in parts[:-1]])
    out = cp.concatenate([p[0] for p in parts])
    out_off = cp.concatenate([p[1] + int(b) for p, b in zip(parts, base, strict=True)])
    return out, out_off, cp.concatenate([p[2] for p in parts])


def _pack(out, out_off, out_size):
    """The streams of ``out`` moved next to each other."""
    import cupy as cp

    if out_size.size == 0:
        return out, out_off, out_size
    new_off = cp.cumsum(out_size) - out_size
    packed = cp.empty(int(new_off[-1] + out_size[-1]), cp.uint8)
    copy_spans(out, packed, out_off, new_off, out_size)
    return packed, new_off, out_size


def _deflate_batch(data, off, size, level: int):
    import cupy as cp

    lib = _nvcomp_compress_lib()
    m = int(size.size)
    if m == 0:
        return cp.empty(0, cp.uint8), cp.empty(0, cp.int64), cp.empty(0, cp.int64)
    max_in, total_in = (int(x) for x in cp.stack([size.max(), size.sum()]).get())
    if max_in > 1 << 16:
        raise ValueError("Deflate chunks hold at most 65,536 bytes")
    opts = _CompressOpts(algorithm=level)
    al = _Alignments()
    _check(lib.nvcompBatchedDeflateCompressGetRequiredAlignments(opts, ctypes.byref(al)), "alignments")
    # inputs at aligned offsets
    a_in = max(int(al.input), 1)
    if a_in > 1 and bool((off % a_in != 0).any()):
        padded = (size + a_in - 1) // a_in * a_in
        new_off = cp.cumsum(padded) - padded
        buf = cp.empty(int(new_off[-1] + padded[-1]), cp.uint8)
        copy_spans(data, buf, off, new_off, size)
        data, off = buf, new_off
    max_out = ctypes.c_size_t()
    _check(lib.nvcompBatchedDeflateCompressGetMaxOutputChunkSize(max(max_in, 1), opts, ctypes.byref(max_out)), "size")
    a_out = max(int(al.output), 1)
    stride = (int(max_out.value) + a_out - 1) // a_out * a_out
    temp_bytes = ctypes.c_size_t()
    _check(
        lib.nvcompBatchedDeflateCompressGetTempSizeAsync(
            m, max(max_in, 1), opts, ctypes.byref(temp_bytes), max(total_in, 1)
        ),
        "temp size",
    )
    temp = cp.empty(max(int(temp_bytes.value), 1) + int(al.temp), cp.uint8)
    temp_ptr = (temp.data.ptr + int(al.temp) - 1) // max(int(al.temp), 1) * max(int(al.temp), 1)
    out = cp.empty(m * stride, cp.uint8)
    in_ptr = off.astype(cp.uint64) + np.uint64(data.data.ptr)
    out_off = cp.arange(m, dtype=cp.int64) * stride
    out_ptr = out_off.astype(cp.uint64) + np.uint64(out.data.ptr)
    in_size = size.astype(cp.uint64)
    out_size = cp.empty(m, cp.uint64)
    status = cp.empty(m, cp.int32)
    _check(
        lib.nvcompBatchedDeflateCompressAsync(
            in_ptr.data.ptr,
            in_size.data.ptr,
            max(max_in, 1),
            m,
            temp_ptr,
            int(temp_bytes.value),
            out_ptr.data.ptr,
            out_size.data.ptr,
            opts,
            status.data.ptr,
            cp.cuda.get_current_stream().ptr,
        ),
        "Deflate compression",
    )
    if bool((status != 0).any()):
        raise RuntimeError("nvCOMP Deflate compression failed on some chunks")
    return out, out_off, out_size.astype(cp.int64)


def _checksum(kernel: str, data, off, size):
    import cupy as cp

    off = cp.asarray(off, cp.int64)
    size = cp.asarray(size, cp.int64)
    out = cp.empty(int(size.size), cp.uint32)
    launch(kernel, int(size.size), data, off, size, np.int64(size.size), out)
    return out


def _crc32_spans(data, off, size):
    """CRC32 (as ``zlib.crc32``) of ``data[off[i] : off[i] + size[i]]``, one uint32 per span."""
    return _checksum("crc32_spans", data, off, size)


def _adler32_spans(data, off, size):
    """Adler-32 (as ``zlib.adler32``) of ``data[off[i] : off[i] + size[i]]``, one uint32 per span."""
    return _checksum("adler32_spans", data, off, size)


#: The empty BGZF block that ends a file
EOF_BLOCK = bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")
#: Data per BGZF block, as samtools and htslib write
BLOCK_DATA = 0xFF00


def _bgzf_compress(data, level: int = 2, block_off=None, block_size=None, return_offsets: bool = False):
    """``data`` (cupy uint8) as BGZF blocks, without the EOF block (cupy uint8).

    Blocks hold ``BLOCK_DATA`` bytes each, or the spans ``block_off``,
    ``block_size`` (at most ``BLOCK_DATA`` bytes each). A block that Deflate
    does not shrink enough to fit is stored uncompressed. With
    ``return_offsets``, also returns where each block starts in the output.
    """
    import cupy as cp

    n = int(data.size)
    if block_off is None:
        off = cp.arange(0, n, BLOCK_DATA, dtype=cp.int64)
        isize = cp.minimum(off + BLOCK_DATA, n) - off
    else:
        off, isize = cp.asarray(block_off, cp.int64), cp.asarray(block_size, cp.int64)
    if off.size == 0:
        empty = cp.empty(0, cp.uint8)
        return (empty, cp.empty(0, cp.int64)) if return_offsets else empty
    comp, comp_off, csize = _deflate_spans(data, off, isize, level)
    crc = _crc32_spans(data, off, isize)
    stored = (csize + 26 > 1 << 16).astype(cp.uint8)
    bsize = 26 + cp.where(stored.astype(cp.bool_), isize + 5, csize)
    out_off = cp.cumsum(bsize) - bsize
    out = cp.empty(int(out_off[-1] + bsize[-1]), cp.uint8)
    m = int(off.size)
    launch(
        "bgzf_frame",
        m,
        comp,
        comp_off,
        csize,
        data,
        off,
        isize,
        stored,
        crc,
        out_off,
        np.int64(m),
        out,
        threads_per_item=32,
    )
    return (out, out_off) if return_offsets else out


class BgzfWriter:
    """A BGZF file written from data on the GPU.

    :meth:`write` appends bytes (cupy uint8 or bytes); full blocks are
    compressed on the GPU once ``flush_bytes`` are buffered. :meth:`close`
    compresses the rest and adds the EOF block.
    """

    def __init__(self, path: str | Path, level: int = 2, flush_bytes: int = 1024 * BLOCK_DATA):
        import cupy as cp

        self.path = Path(path)
        self.level = level
        self.flush_bytes = flush_bytes
        self._fh = open(self.path, "wb")  # noqa: SIM115  (closed in close())
        self._pending: list = []
        self._n = 0
        self._cp = cp

    def write(self, data) -> None:
        cp = self._cp
        if isinstance(data, (bytes, bytearray, memoryview)):
            data = cp.asarray(np.frombuffer(bytes(data), np.uint8))
        if data.size:
            self._pending.append(data)
            self._n += int(data.size)
        if self._n >= self.flush_bytes:
            self._flush(final=False)

    def _flush(self, final: bool) -> None:
        cp = self._cp
        if not self._pending:
            return
        buf = cp.concatenate(self._pending) if len(self._pending) > 1 else self._pending[0]
        cut = int(buf.size) if final else int(buf.size) // BLOCK_DATA * BLOCK_DATA
        if cut:
            self._fh.write(_bgzf_compress(buf[:cut], self.level).get().tobytes())
        rest = buf[cut:]
        self._pending = [rest.copy()] if rest.size else []
        self._n = int(rest.size)

    def close(self) -> None:
        if self._fh.closed:
            return
        try:
            self._flush(final=True)
            self._fh.write(EOF_BLOCK)
        finally:
            self._fh.close()

    def __enter__(self) -> BgzfWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _inflate_blocks(buf, eof: bool):
    """Inflate the complete BGZF blocks of ``buf`` (cupy uint8, starting at a block).

    Returns the inflated data and the number of bytes of ``buf`` used. When
    ``eof``, ``buf`` must end with a complete block.
    """
    import cupy as cp

    n = int(buf.size)
    cand = append_offsets("bgzf_candidates", buf, n // 28 + 1024)
    if cand.size == 0:
        if eof and n:
            raise ValueError("Truncated BGZF file: the last block is incomplete")
        return cp.empty(0, cp.uint8), 0
    bsize = cp.empty(cand.size, cp.int64)
    isize = cp.empty(cand.size, cp.int64)
    launch("bgzf_sizes", int(cand.size), buf, cand, np.int64(cand.size), bsize, isize)
    keep = chain(cand, cand + bsize)
    if keep is not None:
        cand, bsize, isize = cand[keep], bsize[keep], isize[keep]
    used = int(cand[-1] + bsize[-1])
    if eof and used != n:
        raise ValueError("Truncated BGZF file: the last block is incomplete")
    # An empty block (the EOF marker) must hold the empty Deflate stream
    empty = isize == 0
    bad = (isize > _MAX_ISIZE) | (empty & ((bsize != 28) | (buf[cand + 18] != 3) | (buf[cand + 19] != 0)))
    k = cp.flatnonzero(~empty)
    cand, bsize, isize = cand[k], bsize[k], isize[k]
    m = int(cand.size)
    if m == 0:
        if bool(bad.any()):
            raise ValueError("Corrupt BGZF block")
        return cp.empty(0, cp.uint8), used
    pay = bsize - _HEADER - _TRAILER
    padded = (pay + 7) // 8 * 8  # nvCOMP wants 4-byte aligned inputs
    in_off = cp.cumsum(padded) - padded
    out_off = cp.cumsum(isize) - isize
    n_bad, n_in, n_out = (int(x) for x in cp.stack([bad.sum(), in_off[-1] + pay[-1], out_off[-1] + isize[-1]]).get())
    if n_bad:
        raise ValueError("Corrupt BGZF block")
    aligned = cp.empty(n_in + 8, cp.uint8)
    copy_spans(buf, aligned, cand + _HEADER, in_off, pay)
    out = cp.empty(n_out, cp.uint8)
    in_ptr = in_off.astype(cp.uint64) + np.uint64(aligned.data.ptr)
    out_ptr = out_off.astype(cp.uint64) + np.uint64(out.data.ptr)
    in_size, out_size = pay.astype(cp.uint64), isize.astype(cp.uint64)
    actual = cp.empty(m, cp.uint64)
    status = cp.empty(m, cp.int32)
    rc = _nvcomp_decompress()(
        in_ptr.data.ptr,
        in_size.data.ptr,
        out_size.data.ptr,
        actual.data.ptr,
        m,
        None,
        0,
        out_ptr.data.ptr,
        _Opts(),
        status.data.ptr,
        cp.cuda.get_current_stream().ptr,
    )
    if rc != 0:
        raise RuntimeError(f"nvCOMP Deflate decompression failed (status {rc})")
    if bool(((status != 0) | (actual != out_size)).any()):
        raise ValueError("Corrupt BGZF block: it does not inflate to the size in its trailer")
    return out, used


def inflate_chunks(path: str | Path, chunk_bytes: int | None = None, **reader_kw) -> Iterator:
    """Inflated data of a BGZF file on the GPU, chunk by chunk (cupy uint8 arrays).

    Each chunk holds the data of the complete blocks read so far, so chunks
    split the data at block boundaries, not at record or line boundaries.
    ``chunk_bytes`` (compressed) defaults to :func:`cubam.auto_chunk_bytes`.
    """
    import cupy as cp

    from cubam._memory import auto_chunk_bytes

    chunk_bytes = chunk_bytes or auto_chunk_bytes()

    reader = _ChunkReader(path, chunk_bytes, **reader_kw)
    carry = cp.empty(0, cp.uint8)  # compressed bytes of the incomplete last block
    try:
        for slot, nbytes, eof in reader:
            buf = cp.empty(carry.size + nbytes, cp.uint8)
            buf[: carry.size] = carry
            buf[carry.size :].set(reader.host(slot)[:nbytes])
            data, used = _inflate_blocks(buf, eof)  # synchronises: the copy is done
            reader.release(slot)
            carry = buf[used:].copy()
            del buf
            yield data
    finally:
        reader.close()
