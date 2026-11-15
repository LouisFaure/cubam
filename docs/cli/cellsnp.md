# cubam cellsnp

```bash
cubam cellsnp -s gex_possorted_bam.bam -b barcodes.tsv.gz \
    -R snps.vcf.gz -O pileup/ --UMItag UB --minCOUNT 2
```

`cubam cellsnp` takes the options of cellsnp-lite, not those of the other
commands, and writes the files of cellsnp-lite (`cellSNP.base.vcf`,
`cellSNP.samples.tsv` and the `AD`, `DP` and `OTH` matrices), the same
byte for byte. Mode 1a (one BAM file with `-s`, a barcode list with `-b`,
sites with `-R`) runs on the GPU with these options:

| Option | |
|---|---|
| `-s`, `--samFile` | One BAM file |
| `-b`, `--barcodeFile` | The barcodes, one per line |
| `-R`, `--regionsVCF` | The sites |
| `-O`, `--outDir` | The output directory |
| `--minCOUNT`, `--minMAF` | Minimum count and minor allele frequency of a site |
| `--UMItag`, `--cellTAG` | The UMI tag (`Auto`, `UB`, `None`) and barcode tag |
| `--minMAPQ`, `--minLEN`, `--exclFLAG`, `--countORPHAN` | Read filters |
| `-p`, `--nproc` | Accepted and ignored |

Any other mode or option runs the real cellsnp-lite: the one named by the
`CUBAM_CELLSNP_FALLBACK` variable, or else the first `cellsnp-lite` on
`PATH` that is not cubam.

## In place of cellsnp-lite

Run under the name `cellsnp-lite`, `cubam` is `cubam cellsnp`. A link put
first on `PATH` makes tools that call cellsnp-lite, such as the
`pileup_and_phase.R` script of numbat, run on the GPU:

```bash
mkdir -p ~/cubam-bin
ln -s "$(command -v cubam)" ~/cubam-bin/cellsnp-lite
PATH=~/cubam-bin:$PATH Rscript pileup_and_phase.R ...
```

The rules, and the validation against cellsnp-lite, are in
{doc}`../apps/alleles`.
