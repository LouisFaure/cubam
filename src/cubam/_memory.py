"""GPU memory pool."""

from __future__ import annotations

from contextlib import contextmanager


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
