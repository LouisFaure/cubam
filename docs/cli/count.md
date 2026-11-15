# cubam count

```bash
cubam count gex_possorted_bam.bam --features genes.gtf \
    --barcodes barcodes.tsv.gz --strand sense -o counts/

cubam count atac_possorted_bam.bam --features peaks.bed \
    --barcodes barcodes.tsv.gz --umi-tag none -o peaks.h5ad

cubam count gex_possorted_bam.bam --features GX \
    --barcodes barcodes.tsv.gz -o cellranger_genes/
```

`--features` takes a GTF file (exons grouped by `gene_id`, or gene bodies
with `--feature-type gene`), a BED file (named by its 4th column), or the
tag that names the feature of each read. An output ending in `.h5ad` is an
AnnData file (needs the `anndata` extra); any other is a directory with
`matrix.mtx.gz`, `barcodes.tsv.gz` and `features.tsv.gz`, as Cell Ranger
writes them.

The rules, and how the output compares with that of the CPU tool, are in
{doc}`../apps/features`.

## Options

```{eval-rst}
.. argparse::
   :module: cubam.cli
   :func: build_parser
   :prog: cubam
   :path: count
   :nodescription:
   :nodefault:
```
