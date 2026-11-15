# Command line

The `cubam` command runs each application on a BAM file and writes its usual
output files, so most analyses need no Python:

```bash
cubam fragments atac_possorted_bam.bam -o fragments.tsv.gz

cubam count gex_possorted_bam.bam --features genes.gtf \
    --barcodes barcodes.tsv.gz -o counts/

cubam velocity gex_possorted_bam.bam --gtf genes.gtf \
    --barcodes barcodes.tsv.gz -o sample.h5ad
```

The options of a command are the parameters of its Python function, with
the same defaults. `cubam <command> --help` lists them, and works on a
machine without a GPU.

```{toctree}
:hidden:
:maxdepth: 1

fragments
cellsnp
coverage
split
count
velocity
```

::::{grid} 1 1 2 3
:gutter: 2

:::{grid-item-card} {fas}`scissors` cubam fragments
:link: fragments
:link-type: doc
:shadow: none

ATAC fragments of a paired-end BAM file, as Cell Ranger writes them.
:::

:::{grid-item-card} {fas}`dna` cubam cellsnp
:link: cellsnp
:link-type: doc
:shadow: none

Per-barcode allele counts at known sites: a drop-in for cellsnp-lite.
:::

:::{grid-item-card} {fas}`chart-area` cubam coverage
:link: coverage
:link-type: doc
:shadow: none

One bigWig track per group of barcodes, from reads or ATAC fragments.
:::

:::{grid-item-card} {fas}`code-branch` cubam split
:link: split
:link-type: doc
:shadow: none

One BAM file per group of barcodes, or per barcode.
:::

:::{grid-item-card} {fas}`table-cells` cubam count
:link: count
:link-type: doc
:shadow: none

Counts per barcode and gene, peak or repeat, as a 10x matrix or h5ad.
:::

:::{grid-item-card} {fas}`gauge-high` cubam velocity
:link: velocity
:link-type: doc
:shadow: none

Spliced, unspliced and ambiguous counts, as velocyto, for scVelo.
:::

::::

## Barcode files

`--barcodes` gives the barcodes to keep: cells, but also spots or beads.
It takes a list with one barcode per line, such as `barcodes.tsv.gz` of
Cell Ranger, or a CSV or TSV table with the barcodes in its first column.

`coverage` and `split` work per group of barcodes (a cluster, a cell type,
a sample), given by the second column of the table:

```text
barcode,cluster
AAACAGCCAAGGAATC-1,B cells
AAACAGCCAATCCCTT-1,CD14 monocytes
```

The first line is a header when it starts with a name such as `barcode`,
`cell` or `spot`, or with an empty field, as in the CSV files written by
pandas (`adata.obs[["leiden"]].to_csv("clusters.csv")`) and by R. Pick
another column with `--group-column`, by its name or its number. A barcode
with an empty group belongs to no group.

## Options of every command

| Option | |
|---|---|
| `--barcode-tag TAG` | Tag of the barcode of each read (default `CB`) |
| `--keep-chroms NAMES`, `--drop-chroms NAMES` | Contigs to keep or drop, comma-separated (`--drop-chroms chrM`) |
| `--chunk-bytes SIZE` | Compressed bytes read at a time, e.g. `64M` (default: from the free GPU memory) |
| `-v`, `--verbose` | Before the command: print the statistics of the run, and tracebacks |

Each run ends with a line on what it wrote, how many records it read and
how long it took.
