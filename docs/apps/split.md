# One BAM per group of cells

{func}`cubam.apps.split.split_bam` writes one BAM file per group of cells,
or per cell, in one pass over the input.

::::{tab-set}
:sync-group: interface

:::{tab-item} Shell
:sync: shell

```bash
cubam split atac_possorted_bam.bam --barcodes clusters.csv \
    -o by_cluster/
```

See {doc}`../cli/split` for the options.
:::

:::{tab-item} Python
:sync: python

```python
from cubam.cells import CellIndex
from cubam.apps.split import split_bam

res = split_bam("atac_possorted_bam.bam", CellIndex(barcodes, labels=clusters), "by_cluster/")
res.paths  # {cluster: Path("by_cluster/<cluster>.bam")}, then samtools index
```
:::

::::

Records keep the order of the input, and each file has the header of the
input with a `@PG` line. Without labels, or with `by="cell"`, there is one
file per barcode. The BGZF blocks of all groups are compressed together, in
one batched call of the Deflate compressor of nvCOMP.

`level=2` (the default) writes files about 17% larger than those of
samtools (zlib level 6). `level=5` writes files about 9% smaller, about 6
times slower.

## Validation

Each file holds the same records as `samtools view -D CB:barcodes.txt` for
that group.
