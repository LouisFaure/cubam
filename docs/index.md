# cubam

Read BAM files on the GPU, as columns.

cubam reads a BAM file in chunks and returns the records of each chunk as
columns on the GPU (CuPy arrays and cuDF columns). Applications built on the
reader turn reads into counts, tracks and files without leaving the GPU, and
give the same output as the CPU tools they replace.

```python
import cubam

with cubam.memory_pool():
    for batch in cubam.read_bam("possorted_bam.bam"):
        keep = batch.has_flags(none_of=0x904) & (batch.mapq > 30)
        reads = batch.take(keep)
        barcodes = reads.tag("CB").strings()  # cuDF string column
        ends = reads.five_prime()  # 5' ends, soft clips included
```

| Application | Output | Same output as |
|---|---|---|
| [Fragments](apps/fragments.md) | ATAC fragments | Cell Ranger ARC 2.0.2 |
| [Allele counts](apps/alleles.md) | Per-cell allele counts at known sites | cellsnp-lite 1.2.3 (mode 1a) |
| [Coverage](apps/coverage.md) | Coverage tracks per group of cells, as bigWig | `samtools depth` |
| [Split](apps/split.md) | One BAM file per group of cells | `samtools view -D` |
| [Feature counts](apps/features.md) | Counts per cell and gene, repeat or peak | featureCounts, Cell Ranger |
| [Velocity](apps/velocity.md) | Spliced, unspliced and ambiguous molecules | velocyto 0.17 |

[Benchmarks](benchmarks.md) gives the speed and the memory of each, against
the CPU tools. Every application runs on an 8 GB GPU.

```{toctree}
:hidden:
:caption: Guide

install
reading
```

```{toctree}
:hidden:
:caption: Applications

apps/fragments
apps/alleles
apps/coverage
apps/split
apps/features
apps/velocity
```

```{toctree}
:hidden:
:caption: Reference

benchmarks
api
development
```
