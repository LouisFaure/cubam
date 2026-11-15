# cubam split

```bash
cubam split atac_possorted_bam.bam --barcodes clusters.csv \
    -o by_cluster/
for f in by_cluster/*.bam; do samtools index "$f"; done
```

Each group of `--barcodes` gets a file `<prefix><group>.bam`; a list of
barcodes without groups, or `--per-barcode`, gives one file per barcode.

The rules, and how the output compares with that of the CPU tool, are in
{doc}`../apps/split`.

## Options

```{eval-rst}
.. argparse::
   :module: cubam.cli
   :func: build_parser
   :prog: cubam
   :path: split
   :nodescription:
   :nodefault:
```
