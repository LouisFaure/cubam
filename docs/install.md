# Installation

cubam needs Linux, Python 3.11 or newer, and an NVIDIA GPU with CUDA 12
that RAPIDS supports. The GPU libraries come with the `rapids-cu12` extra:
cuDF, CuPy and the nvCOMP C library.

```bash
pip install \
    "cubam[rapids-cu12] @ git+https://github.com/LouisFaure/cubam"
```

or with uv:

```bash
uv pip install \
    "cubam[rapids-cu12] @ git+https://github.com/LouisFaure/cubam"
```

Add the `anndata` extra (`cubam[rapids-cu12,anndata]`) to get AnnData
objects and h5ad files from the count applications.

If RAPIDS is already installed with conda, install cubam without the extra
(`pip install --no-deps ...`) so that pip does not pull a second copy of
cuDF.

## GPU memory

cubam reads 256 MB of the BAM file at a time on GPUs with 24 GB or more,
and smaller chunks on smaller GPUs (see {func}`cubam.auto_chunk_bytes`), so
that every application fits on an 8 GB GPU. Pass `chunk_bytes` to any
application, or `--chunk-bytes` to any command, to choose the chunk size. Measured peaks are in
[Benchmarks](benchmarks.md).
