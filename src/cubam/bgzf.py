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


class ChunkReader:
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


_lib = None


def _nvcomp_decompress():
    """``nvcompBatchedDeflateDecompressAsync`` of libnvcomp 5."""
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
        fn = _lib.nvcompBatchedDeflateDecompressAsync
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
    return _lib.nvcompBatchedDeflateDecompressAsync


def inflate_blocks(buf, eof: bool):
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


def inflate_chunks(path: str | Path, chunk_bytes: int = 256 << 20, **reader_kw) -> Iterator:
    """Inflated data of a BGZF file on the GPU, chunk by chunk (cupy uint8 arrays).

    Each chunk holds the data of the complete blocks read so far, so chunks
    split the data at block boundaries, not at record or line boundaries.
    """
    import cupy as cp

    reader = ChunkReader(path, chunk_bytes, **reader_kw)
    carry = cp.empty(0, cp.uint8)  # compressed bytes of the incomplete last block
    try:
        for slot, nbytes, eof in reader:
            buf = cp.empty(carry.size + nbytes, cp.uint8)
            buf[: carry.size] = carry
            buf[carry.size :].set(reader.host(slot)[:nbytes])
            data, used = inflate_blocks(buf, eof)  # synchronises: the copy is done
            reader.release(slot)
            carry = buf[used:].copy()
            del buf
            yield data
    finally:
        reader.close()
