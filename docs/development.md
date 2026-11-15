# Development

```bash
uv sync --extra rapids-cu12 --extra anndata
uv run --extra rapids-cu12 --extra anndata pytest
```

The tests write small BAM files with a pure-Python writer
(`tests/bamwriter.py`) and need a GPU. The bigWig tests read files back with
pyBigWig (in the `test` group). The AnnData and h5ad tests need the
`anndata` extra and are skipped without it.

The docs build without a GPU:

```bash
uv run --group docs sphinx-build -W docs docs/_build/html
```

## Benchmarks

`benchmarks/speed_cubam.py` times each application on one multiome sample,
with its peak GPU and host memory. `benchmarks/speed_baselines.sh` runs the
CPU tools on the same inputs. `benchmarks/compare_cellranger.py` and
`benchmarks/compare_cellsnp.py` compare outputs with those of Cell Ranger
and cellsnp-lite.

## Open items

- **nvCOMP 6.** cubam loads `libnvcomp.so.5`. nvCOMP 6 needs a check of the
  C API (the options struct).
- **Region queries** through the BAI index.
- **Name-sorted and single-end input** for the fragment caller.
