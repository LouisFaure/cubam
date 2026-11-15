# Fragments

{func}`cubam.apps.fragments.call_fragments` calls ATAC fragments from a
coordinate-sorted, paired-end BAM file with the rules of Cell Ranger.

::::{tab-set}
:sync-group: interface

:::{tab-item} Shell
:sync: shell

```bash
cubam fragments atac_possorted_bam.bam -o fragments.tsv.gz \
    --keep-chroms "$(paste -sd, primary_contigs.txt)"
```

See {doc}`../cli/fragments` for the options.
:::

:::{tab-item} Python
:sync: python

```python
from cubam.apps.fragments import call_fragments

frags = call_fragments("atac_possorted_bam.bam", keep_chroms=primary_contigs)
frags.table  # ref, start, end, bc (packed barcode), count: a pandas DataFrame
frags.write_tsv("fragments.tsv.gz")  # as Cell Ranger writes it, BGZF compressed on the GPU
```
:::

::::

`keep_chroms` lists the primary contigs of the reference: the
`# primary_contig=` lines of a Cell Ranger fragments file, or its
`contig-defs.json`. The BAM header does not list them.

## Passing fragments to other tools

The table holds integers only: barcodes are packed into 64-bit codes. Three
methods give the fragments to other tools without a string per fragment:

- {meth}`~cubam.apps.fragments.Fragments.iter_gpu` yields slices of the
  table as cuDF DataFrames, for GPU tools.
- {meth}`~cubam.apps.fragments.Fragments.barcode_stats` gives the fragments
  and read pairs of each barcode, sorted by decreasing number of fragments.
- {meth}`~cubam.apps.fragments.Fragments.to_arrow` gives an Arrow table with
  dictionary-encoded chromosomes and barcodes, which pandas, polars and
  DuckDB read as is (7 s for 348 million fragments).

## Rules

The rules come from the source of Cell Ranger ATAC 1.2
(`lib/python/tools/peaks.py` and the `MARK_DUPLICATES` stage):

- The fragment runs from the 5' end of the forward read to the 5' end of
  the reverse read, soft clips included, shifted by +4 and -5 bp (Tn5).
- Both reads need a MAPQ above 30.
- Chimeric pairs are dropped: mates on different chromosomes or the same
  strand, facing outward (5' ends compared), or with alignment starts more
  than 5,000 bp apart.
- Contigs outside `keep_chroms` are dropped. A fragment whose start lies
  before the contig start is dropped, and the end is clipped to the contig
  length.
- Pairs with the same barcode, chromosome, start and end are one fragment,
  with `count` pairs. The filters apply to the representative of a group,
  the pair without the duplicate flag: when it fails, the group is dropped.

Mates are joined on the hash of the read name: the BAM files of Cell Ranger
have no `MQ` tag, so the left mate alone cannot tell whether the right one
passes the MAPQ filter. Fragments are final once they start far enough
behind the last position read, and move to host memory then, so GPU memory
does not grow with the number of fragments.

## Validation

The fragments equal the `atac_fragments.tsv.gz` of Cell Ranger ARC 2.0.2 on
every row and count, on 4 samples. `benchmarks/compare_cellranger.py` runs
the comparison on a Cell Ranger output folder.
