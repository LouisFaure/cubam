# cubam coverage

```bash
cubam coverage gex_possorted_bam.bam --barcodes clusters.csv \
    -o tracks/rna/

cubam coverage atac_possorted_bam.bam --barcodes clusters.csv --atac \
    --bin-size 10 -o tracks/atac/
```

Each group of `--barcodes` gets a file `<prefix><group>.bw`. Without a
group column, all barcodes form one group, `all`.

The rules, and how the output compares with that of the CPU tool, are in
{doc}`../apps/coverage`.

## Options

```{eval-rst}
.. argparse::
   :module: cubam.cli
   :func: build_parser
   :prog: cubam
   :path: coverage
   :nodescription:
   :nodefault:
```
