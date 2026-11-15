# Coverage per group of cells

{func}`cubam.apps.coverage.coverage` and
{func}`cubam.apps.coverage.fragment_coverage` build one coverage track per
group of cells (clusters, cell types) and write them as bigWig files.

```python
from cubam.cells import CellIndex
from cubam.apps.coverage import coverage, fragment_coverage

cells = CellIndex(barcodes, labels=adata.obs["leiden"])  # one group per cell

rna = coverage("gex_possorted_bam.bam", cells)  # aligned blocks of reads
atac = fragment_coverage("atac_possorted_bam.bam", cells, keep_chroms=primary_contigs)  # Tn5 insertions
rna.write_bigwig("tracks/rna", normalize="cpm")  # one file per group
atac.write_bigwig("tracks/atac", bin_size=10)
```

## Rules

Each read or fragment adds +1 where it starts and -1 where it ends. The
events are summed per group into runs of constant coverage while the BAM is
read.

- `coverage` covers the aligned blocks of reads: introns split a read,
  deletions do not (as `bedtools genomecov -split`). By default it keeps
  primary alignments with MAPQ of at least 30 and no duplicate flag, which
  leaves one read per UMI in a Cell Ranger BAM.
- `fragment_coverage` takes fragments called with the rules of Cell Ranger
  (from a BAM file, or a {class}`~cubam.apps.fragments.Fragments` object).
  It covers either the Tn5 insertion at each end of a fragment (`start` and
  `end - 1`, as ArchR) or whole fragments (`mode="fragments"`).
- Tracks are normalised to counts per million reads, fragments or
  insertions of the group (`"cpm"`), per cell of the group (`"cells"`), or
  left as counts (`None`). `bin_size` averages the track over bins.

## bigWig files

{func}`cubam.bigwig.write_bigwig` writes the bigWig files on the GPU: data
sections and zoom levels are built and compressed there, the R-trees on the
host. `bigWigToBedGraph` (UCSC) gives the same lines as for the files
pyBigWig writes, which took 85 s for the ATAC tracks of one sample, against
13 s. The files are about 20% larger, at the default nvCOMP level of 2.

## Validation

For RNA, the depth equals that of `samtools depth` at every base of one
group, on chr19 of a GEX BAM file.
