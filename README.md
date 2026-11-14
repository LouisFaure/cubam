# cubam

Read BAM files on the GPU, as columns.

cubam reads a BAM file in chunks and returns the records of each chunk as
columns on the GPU (CuPy arrays and cuDF columns). Applications built on the
reader give the same output as the CPU tools they replace, in seconds.

Documentation: https://cubam.readthedocs.io

## Applications

On one 10x multiome sample (GEX BAM 15 GB, ATAC BAM 46 GB), one NVIDIA A100
against the CPU tool on 6 threads:

| Application | Same output as | CPU tool | CPU | cubam | Speedup |
|---|---|---|---:|---:|---:|
| Velocity counts | velocyto 0.17 | velocyto | 117 min | 30 s | 238× |
| Allele counts, GEX | cellsnp-lite 1.2.3 | cellsnp-lite | 78 min | 17 s | 273× |
| Allele counts, ATAC | cellsnp-lite 1.2.3 | cellsnp-lite | 26 min | 61 s | 26× |
| Fragments | Cell Ranger ARC 2.0.2 | sinto | 49 min | 60 s | 49× |
| Coverage per group, as bigWig | `samtools depth` | samtools, bedtools | 15 min | 11 s | 85× |
| One BAM per group, GEX | `samtools view -D` | sinto | 15 min | 21 s | 42× |
| One BAM per group, ATAC | `samtools view -D` | sinto | 27 min | 54 s | 30× |
| Counts per cell and gene | featureCounts, Cell Ranger | featureCounts, no cells | 56 s | 16 s | 3.6× |

Times include reading the inputs and writing the outputs. Every
application also runs on an 8 GB GPU, 5 to 50% slower. The CPU tools and
their options, memory, and how to reproduce the numbers are in the
[benchmarks](https://cubam.readthedocs.io/en/latest/benchmarks.html).

## Install

cubam needs Linux, Python 3.11 or newer, and an NVIDIA GPU with CUDA 12.

```bash
pip install "cubam[rapids-cu12] @ git+https://github.com/LouisFaure/cubam"
```

Add the `anndata` extra for AnnData and h5ad outputs.

## Example

```python
import cubam

with cubam.memory_pool():
    for batch in cubam.read_bam("possorted_bam.bam"):
        keep = batch.has_flags(none_of=0x904) & (batch.mapq > 30)
        reads = batch.take(keep)
        barcodes = reads.tag("CB").strings()  # cuDF string column
        ends = reads.five_prime()  # 5' ends, soft clips included
```

```python
from cubam.apps.velocity import count_velocity

res = count_velocity("gex_possorted_bam.bam", "genes.gtf", "barcodes.tsv.gz")
res.write_h5ad("sample.h5ad")  # layers spliced, unspliced, ambiguous
```

## Development

```bash
uv sync --extra rapids-cu12 --extra anndata
uv run --extra rapids-cu12 --extra anndata pytest
```

The tests need a GPU. See the
[development page](https://cubam.readthedocs.io/en/latest/development.html).
