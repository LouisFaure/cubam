# Allele counts

{func}`cubam.apps.alleles.count_alleles` counts the alleles of each cell at
known sites, as cellsnp-lite does in mode 1a (a VCF of sites and a barcode
list).

::::{tab-set}
:sync-group: interface

:::{tab-item} Shell
:sync: shell

```bash
cubam cellsnp -s gex_possorted_bam.bam -b barcodes.tsv -R snps.vcf \
    -O pileup/sample_rna --UMItag UB --minCOUNT 2
```

See {doc}`../cli/cellsnp` for the options.
:::

:::{tab-item} Python
:sync: python

```python
from cubam.apps.alleles import count_alleles

res = count_alleles("gex_possorted_bam.bam", "snps.vcf", "barcodes.tsv", umi_tag="UB", min_count=2)
res.write_cellsnp("pileup/sample_rna")  # cellSNP.base.vcf, cellSNP.samples.tsv, AD/DP/OTH matrices
```
:::

::::

## Rules

The rules are those of cellsnp-lite 1.2.3 in mode 1a:

- reads with MAPQ of at least 20 and at least 30 aligned bases, proper
  pairs only;
- one count per UMI, from the first read of the UMI (`umi_tag="UB"`);
- every read counts and duplicates are skipped (`umi_tag=None`, for ATAC);
- sites with more than two alleles are skipped.

## A drop-in for cellsnp-lite

`cubam cellsnp` takes the arguments of cellsnp-lite. Mode 1a (one BAM with
`-s`, a barcode list with `-b`, sites with `-R`) runs on the GPU and writes
the same files. Any other mode, or an option that cubam does not implement,
runs the real cellsnp-lite.

Run under the name `cellsnp-lite`, `cubam` is `cubam cellsnp`: a link put
first on `PATH` speeds up tools that call cellsnp-lite, such as the
`pileup_and_phase.R` script of numbat. See {doc}`../cli/cellsnp`.

## Validation

The output files are the same as those of cellsnp-lite 1.2.3, byte for
byte, on the GEX and ATAC BAM files of one multiome sample.
`benchmarks/compare_cellsnp.py` compares the counts with an output of
cellsnp-lite.

A mode for every position of chrM, as mgatk counts for lineage tracing, is
planned.
