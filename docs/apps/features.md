# Counts per cell and feature

{func}`cubam.apps.features.count_features` counts reads or UMIs per cell
and feature: genes from a GTF file, repeats, peaks from a BED file, or the
gene Cell Ranger gave each read.

```python
from cubam.intervals import IntervalIndex
from cubam.apps.features import count_features

genes = IntervalIndex.from_gtf("genes.gtf", feature="exon", group_by="gene_id")  # or feature="gene"
res = count_features("gex_possorted_bam.bam", genes, "barcodes.tsv.gz", umi_tag="UB", strand="sense")
res.write_mtx("counts/")  # matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz
adata = res.to_anndata()  # needs the anndata extra
```

## Rules

{class}`~cubam.intervals.IntervalIndex` cuts the genome at every start and
end of the intervals of a GTF or BED file. Each segment between two cuts
holds a fixed set of features, and two binary searches match a read to its
features.

The rules per read are those of featureCounts:

- The aligned blocks (or the 5' or 3' end, with `positions`) must overlap
  a feature by `min_overlap` bases.
- `strand="sense"` requires the read and the feature to share a strand.
- `assign` picks exactly one feature (`"unique"`), every matching feature
  (`"any"`), or the feature with the most overlapping bases (`"largest"`).
- With `umi_tag`, a (cell, UMI, feature) counts once.

Features can also come from a tag: `count_features(bam, "GX", ...)` counts
the gene Cell Ranger gave each read. `where` adds a test on the records,
e.g. `where=lambda b: (b.tag("xf").value & 8) != 0` for the reads Cell
Ranger counted.

## Validation

On the GEX BAM file of one multiome sample:

| Features | Compared with |
|---|---|
| `GX` tag, `xf` bit 8 | Cell Ranger ARC 2.0.2 matrix: all 12,507,428 entries equal |
| GTF exons, sense, `UB` | featureCounts `-t exon -s 1 -M` on chr19: every gene equal |
| GTF gene bodies, sense, `UB` | featureCounts `-t gene -s 1 -M` on chr19: every gene equal |

Gene bodies give 3.8% more UMIs than the Cell Ranger matrix: Cell Ranger
prefers an exon over an intron when genes overlap, and drops UMIs seen on
several genes.
