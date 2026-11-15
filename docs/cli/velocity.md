# cubam velocity

```bash
cubam velocity gex_possorted_bam.bam --gtf genes.gtf \
    --barcodes barcodes.tsv.gz -o sample.h5ad

cubam velocity gex_possorted_bam.bam --gtf genes.gtf \
    --barcodes barcodes.tsv.gz --mask rmsk.gtf -o sample.h5ad
```

The `.h5ad` file has the layers `spliced`, `unspliced` and `ambiguous`,
which scVelo reads (needs the `anndata` extra). The BAM file does not
need velocyto's cell-sorted copy.

The rules, and how the output compares with that of the CPU tool, are in
{doc}`../apps/velocity`.

## Options

```{eval-rst}
.. argparse::
   :module: cubam.cli
   :func: build_parser
   :prog: cubam
   :path: velocity
   :nodescription:
   :nodefault:
```
