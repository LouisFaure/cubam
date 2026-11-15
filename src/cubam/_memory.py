"""GPU memory: the pool, and how much to read at a time."""

from __future__ import annotations

from contextlib import contextmanager

#: Peak GPU memory of the apps is about 25 times the compressed chunk size, plus what they keep
_CHUNK_SHARE = 64


def auto_chunk_bytes() -> int:
    """Compressed bytes to read per chunk: 256 MB, halved while 64 chunks do not fit in the free GPU memory.

    About 64 MB on an 8 GB GPU, 128 MB on a 12 or 16 GB GPU and 256 MB from
    24 GB on (never below 32 MB). Smaller chunks cost about 10% more time.
    """
    import cupy as cp

    free, _ = cp.cuda.runtime.memGetInfo()
    size = 256 << 20
    while size > 32 << 20 and size * _CHUNK_SHARE > free:
        size //= 2
    return size


@contextmanager
def memory_pool():
    """Use the stream-ordered pool of RMM (``cudaMallocAsync``) unless a memory resource is already set.

    cuDF routes CuPy allocations through RMM. Without a pool, every
    per-chunk buffer is a ``cudaMalloc``, which costs about as much as the
    work on the chunk.
    """
    import cudf  # noqa: F401  (sets the CuPy allocator to RMM)
    import rmm

    previous = rmm.mr.get_current_device_resource()
    if type(previous) is not rmm.mr.CudaMemoryResource:
        yield
        return
    rmm.mr.set_current_device_resource(rmm.mr.CudaAsyncMemoryResource())
    try:
        yield
    finally:
        rmm.mr.set_current_device_resource(previous)
