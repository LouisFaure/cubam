# API

The Python API follows the path of a read: the reader yields batches of
records as columns on the GPU, the selection modules keep the reads of the
cells, contigs and regions of interest, and each application turns them into
counts, tracks or files.

```{toctree}
:hidden:
:maxdepth: 2

reading
selection
fragments
alleles
coverage
split
sort
markdup
features
velocity
bgzf
bigwig
```

::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card} {fas}`book-open` Reading — `cubam`
:link: reading
:link-type: doc
:shadow: none

Read a BAM file in chunks, as batches of records with their fields as GPU
columns.
:::

:::{grid-item-card} {fas}`filter` Selecting reads
:link: selection
:link-type: doc
:shadow: none

Cell barcode lookups, read filters on flags and MAPQ, and overlaps with
GTF or BED intervals.
:::

:::{grid-item-card} {fas}`scissors` Fragments — `cubam.apps.fragments`
:link: fragments
:link-type: doc
:shadow: none

ATAC fragments from a paired-end BAM, with the rules of Cell Ranger.
:::

:::{grid-item-card} {fas}`dna` Allele counts — `cubam.apps.alleles`
:link: alleles
:link-type: doc
:shadow: none

Per-cell allele counts at known sites, as cellsnp-lite counts them.
:::

:::{grid-item-card} {fas}`chart-area` Coverage — `cubam.apps.coverage`
:link: coverage
:link-type: doc
:shadow: none

Coverage tracks per group of cells, from reads or fragments, as bigWig.
:::

:::{grid-item-card} {fas}`code-branch` Split BAM — `cubam.apps.split`
:link: split
:link-type: doc
:shadow: none

One BAM file per group of cells or per cell.
:::

:::{grid-item-card} {fas}`arrow-down-a-z` Sort BAM — `cubam.apps.sort`
:link: sort
:link-type: doc
:shadow: none

A BAM file sorted by coordinate or by read name, as samtools sort.
:::

:::{grid-item-card} {fas}`clone` Duplicate marking — `cubam.apps.markdup`
:link: markdup
:link-type: doc
:shadow: none

Duplicate flags as samtools markdup, optionally within a cell barcode.
:::

:::{grid-item-card} {fas}`table-cells` Feature counts — `cubam.apps.features`
:link: features
:link-type: doc
:shadow: none

Counts per cell and feature: genes, transposable elements, peaks.
:::

:::{grid-item-card} {fas}`gauge-high` Velocity counts — `cubam.apps.velocity`
:link: velocity
:link-type: doc
:shadow: none

Spliced, unspliced and ambiguous molecules per cell and gene, as velocyto
counts them.
:::

:::{grid-item-card} {fas}`file-zipper` BGZF — `cubam.bgzf`
:link: bgzf
:link-type: doc
:shadow: none

Read and inflate BGZF files on the GPU, and write them.
:::

:::{grid-item-card} {fas}`wave-square` bigWig — `cubam.bigwig`
:link: bigwig
:link-type: doc
:shadow: none

Write bigWig files from coverage computed on the GPU.
:::

::::
