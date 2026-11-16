# cubam: GPU-accelerated BAM parsing into columns

cubam reads a BAM file in chunks and returns the records of each chunk as
columns on the GPU (CuPy arrays and cuDF columns). Applications built on the
reader turn reads into counts, tracks and files without leaving the GPU, and
give the same output as the CPU tools they replace.

```bash
cubam fragments atac_possorted_bam.bam -o fragments.tsv.gz

cubam cellsnp -s gex_possorted_bam.bam -b barcodes.tsv.gz \
    -R snps.vcf.gz -O pileup/

cubam coverage gex_possorted_bam.bam \
    --barcodes clusters.csv -o tracks/

cubam split atac_possorted_bam.bam \
    --barcodes clusters.csv -o by_cluster/

cubam markdup sorted.bam --barcode-tag CB -o marked.bam

cubam count gex_possorted_bam.bam --features genes.gtf \
    --barcodes barcodes.tsv.gz -o counts/

cubam velocity gex_possorted_bam.bam --gtf genes.gtf \
    --barcodes barcodes.tsv.gz -o sample.h5ad
```

The [command line](cli/index.md) runs each application, and
`cubam <command> --help` lists its options. The same applications, and the
reader that gives the records of a BAM file as GPU columns, are also
available in Python: see [Reading a BAM file](reading.md).

| Application | Command | Output | Same output as |
|---|---|---|---|
| [Fragments](apps/fragments.md) | `cubam fragments` | ATAC fragments | Cell Ranger ARC 2.0.2 |
| [Allele counts](apps/alleles.md) | `cubam cellsnp` | Per-cell allele counts at known sites | cellsnp-lite 1.2.3 (mode 1a) |
| [Coverage](apps/coverage.md) | `cubam coverage` | Coverage tracks per group of cells, as bigWig | `samtools depth` |
| [Split](apps/split.md) | `cubam split` | One BAM file per group of cells | `samtools view -D` |
| [Duplicate marking](apps/markdup.md) | `cubam markdup` | Duplicate flags, within a cell barcode | samtools fixmate -m + markdup 1.22 |
| [Feature counts](apps/features.md) | `cubam count` | Counts per cell and gene, repeat or peak | featureCounts, Cell Ranger |
| [Velocity](apps/velocity.md) | `cubam velocity` | Spliced, unspliced and ambiguous molecules | velocyto 0.17 |

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
apps/markdup
apps/features
apps/velocity
```

```{toctree}
:hidden:
:caption: Reference

cli/index
api/index
benchmarks
development
```
