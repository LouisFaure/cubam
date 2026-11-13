# cubam

Read BAM files on the GPU, as columns.

cubam reads a BAM file in chunks and returns the records of each chunk as
columns on the GPU (CuPy arrays and cuDF columns). Applications built on the
reader turn reads into counts without leaving the GPU.

```python
import cubam

with cubam.memory_pool():
    for batch in cubam.read_bam("possorted_bam.bam"):
        keep = batch.has_flags(none_of=0x904) & (batch.mapq > 30)
        reads = batch.take(keep)
        barcodes = reads.tag("CB").strings()  # cuDF string column
        ends = reads.five_prime()  # 5' ends, soft clips included
```

## How it reads

| Step | What happens |
|---|---|
| Read | A background thread reads chunks into pinned host memory, with 8 parallel `pread` calls |
| Find blocks | A kernel finds the BGZF block headers. The chain from the first block is kept |
| Inflate | One call of the batched Deflate function of the nvCOMP C library per chunk |
| Find records | A kernel tests every byte offset as a record start (length, ids, read name, CIGAR, `bin`) |
| Decode | Fixed fields in one kernel. Tags, read names and CIGAR-derived values on request |

On one NVIDIA L40S, a Cell Ranger ARC ATAC BAM of 8.7 GB (175 million
records) is read and turned into fragments in 2.9 s when the file is in the
page cache. From a network file system, the time is set by the read speed
(1 to 2 GB/s on WekaFS).

## Applications

| Module | Output | Status |
|---|---|---|
| `cubam.apps.fragments` | ATAC fragments with the rules of Cell Ranger | Matches Cell Ranger ARC 2.0.2 on every row and count of 4 samples |
| `cubam.apps.alleles` | Per-cell allele counts at known sites, as cellsnp-lite | Same output files as cellsnp-lite 1.2.3, byte for byte, on the GEX and ATAC BAM of one multiome sample |
| `cubam.apps.coverage` | Coverage tracks per group of cells (clusters), as bigwig | RNA: same depth as `samtools depth` at every base of one group on chr19 of a GEX BAM |
| `cubam.apps.split` | One BAM file per group of cells or per cell, compressed on the GPU | Same records as `samtools view -D CB:...` for one group of a GEX BAM |
| `cubam.apps.features` | Counts per cell and feature (genes, repeats, peaks) from a GTF, a BED or a tag | Same counts per gene as featureCounts 2.1.1 on chr19; with the `GX` tag, the same matrix as Cell Ranger ARC 2.0.2 |
| velocity counts | Spliced, unspliced and ambiguous UMIs per cell and gene | Planned |

### Fragments

```python
from cubam.apps.fragments import call_fragments

frags = call_fragments("atac_possorted_bam.bam", keep_chroms=primary_contigs)
frags.to_frame()  # chrom, start, end, barcode, count (cuDF)
```

`benchmarks/compare_cellranger.py` compares the result with the
`atac_fragments.tsv.gz` of the same Cell Ranger run.

### Allele counts

```python
from cubam.apps.alleles import count_alleles

res = count_alleles("gex_possorted_bam.bam", "snps.vcf", "barcodes.tsv", umi_tag="UB", min_count=2)
res.write_cellsnp("pileup/sample_rna")  # cellSNP.base.vcf, cellSNP.samples.tsv, AD/DP/OTH matrices
```

The rules are those of cellsnp-lite 1.2.3 in mode 1a, with a VCF of sites
and a barcode list:

- reads with MAPQ of at least 20 and at least 30 aligned bases, proper pairs
  only
- one count per UMI, from the first read of the UMI (`umi_tag="UB"`)
- every read counts, duplicates skipped (`umi_tag=None`, for ATAC)
- sites with more than two alleles skipped

On one multiome sample (5,647 cells, 7.4 million SNPs of 1000 Genomes), on
one NVIDIA L40S against cellsnp-lite with 20 threads:

| BAM | cubam | cellsnp-lite | Sites kept |
|---|---:|---:|---:|
| GEX, 15 GB | 11.5 s | 1,595 s | 316,964 |
| ATAC, 46 GB | 37.9 s | 546 s | 2,577,210 |

`benchmarks/compare_cellsnp.py` compares the counts with an output of
cellsnp-lite.

A mode for every position of chrM, as mgatk counts for lineage tracing, is
planned.

### Coverage per group

```python
from cubam.cells import CellIndex
from cubam.apps.coverage import coverage, fragment_coverage

cells = CellIndex(barcodes, labels=adata.obs["leiden"])  # one group per cell

rna = coverage("gex_possorted_bam.bam", cells)  # aligned blocks of reads
atac = fragment_coverage("atac_possorted_bam.bam", cells, keep_chroms=primary_contigs)  # Tn5 insertions
rna.write_bigwig("tracks/rna", normalize="cpm")  # one file per group
atac.write_bigwig("tracks/atac", bin_size=10)
```

Each read or fragment adds +1 where it starts and -1 where it ends. The
events are summed per group into runs of constant coverage while the BAM is
read. `coverage` covers the aligned blocks of reads: introns split a read,
deletions do not (as `bedtools genomecov -split`). By default it keeps
primary alignments with MAPQ of at least 30 and no duplicate flag, which
leaves one read per UMI in a Cell Ranger BAM. `fragment_coverage` takes
fragments called with the rules of Cell Ranger (or a `Fragments` object),
and covers either the Tn5 insertion at each end of a fragment (`start` and
`end - 1`, as ArchR) or whole fragments. Tracks are normalised to counts
per million reads, fragments or insertions of the group (`"cpm"`), per cell
of the group (`"cells"`), or left as counts.

On one multiome sample with 8 groups, on one NVIDIA A100:

| Input | Coverage | Bigwig files |
|---|---:|---:|
| GEX BAM, 15 GB, 300 million records | 6.8 s | 23 s |
| ATAC BAM, 46 GB: fragments, then insertions | 24 s + 1.0 s | 63 s (139 million runs) |

Bigwig files are written with pyBigWig (`pip install cubam[bigwig]`), one
group per process: pyBigWig writes about a million runs per second.

### One BAM per group of cells

```python
from cubam.apps.split import split_bam

res = split_bam("atac_possorted_bam.bam", CellIndex(barcodes, labels=clusters), "by_cluster/")
res.paths  # {cluster: Path("by_cluster/<cluster>.bam")}, then samtools index
```

Records keep the order of the input, and each file has the header of the
input with a `@PG` line. Without labels, or with `by="cell"`, there is one
file per barcode. The BGZF blocks of all groups are compressed together, in
one batched call of the Deflate compressor of nvCOMP per GB of records.

On one multiome sample with 8 groups, on one NVIDIA A100 (the GEX BAM
holds 94 million records of the listed cells, the ATAC BAM 235 million):

| BAM | `level=2` (default) | `level=5` |
|---|---:|---:|
| GEX, 15 GB | 20 s, 5.4 GB written | |
| ATAC, 46 GB | 53 s, 14.1 GB written | 302 s, 11.6 GB written |

Level 2 files are about 17% larger than those of samtools (zlib level 6),
level 5 files about 9% smaller. `samtools view -D CB:barcodes.txt` takes
52 s for one group of the GEX BAM on 6 threads.

`cubam.bgzf.BgzfWriter` writes any BGZF file (BAM, VCF.gz, fragments)
from data on the GPU.

### Counts per cell and feature

```python
from cubam.intervals import IntervalIndex
from cubam.apps.features import count_features

genes = IntervalIndex.from_gtf("genes.gtf", feature="exon", group_by="gene_id")  # or feature="gene"
res = count_features("gex_possorted_bam.bam", genes, "barcodes.tsv.gz", umi_tag="UB", strand="sense")
res.write_mtx("counts/")  # matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz
adata = res.to_anndata()  # needs anndata and scipy
```

`IntervalIndex` cuts the genome at every start and end of the intervals of
a GTF or BED file, so that each segment between two cuts holds a fixed set
of features; a read is matched to features with two binary searches. The
rules per read are those of featureCounts: the aligned blocks of the read
(or its 5' or 3' end) must overlap a feature by `min_overlap` bases, on the
same strand with `strand="sense"`, and `assign` picks the features of the
read (`"unique"`: exactly one, `"any"`: each, `"largest"`: the one with the
most bases). With `umi_tag`, a (cell, UMI, feature) counts once.

Features can also come from a tag: `count_features(bam, "GX", ...)`
counts the gene Cell Ranger gave each read. `where` adds a test on the
records, e.g. `where=lambda b: (b.tag("xf").value & 8) != 0` for the reads
Cell Ranger counted.

On the GEX BAM of one multiome sample (15 GB), on one NVIDIA A100:

| Features | Time | Compared with |
|---|---:|---|
| `GX` tag, `xf` bit 8 | 34 s | Cell Ranger matrix: all 12,507,428 entries equal |
| GTF exons, sense, `UB` | 11 s | featureCounts `-t exon -s 1 -M` on chr19: every gene equal |
| GTF gene bodies, sense, `UB` | 9.5 s | featureCounts `-t gene -s 1 -M` on chr19: every gene equal |

Gene bodies give 3.8% more UMIs than the Cell Ranger matrix: Cell Ranger
prefers an exon over an intron when genes overlap, and drops UMIs seen on
several genes.

### Velocity counts (planned)

Recent Cell Ranger versions record intronic and exonic reads in
`molecule_info.h5`, so this is for BAM files from other pipelines (STARsolo
without the Velocyto feature, alevin-fry, in-house). The rules follow velocyto: a molecule is
spliced, unspliced or ambiguous from how the aligned blocks of the reads
overlap exons and introns. Exon and intron intervals come from a GTF file,
sorted on the GPU. Each read is checked by binary search, then a group-by on
(barcode, UMI, gene) gives one class per molecule. Validation: velocyto on
a Cell Ranger GEX BAM.

## Open items

- **CRC32.** nvCOMP does not check the CRC32 in the BGZF trailer, and
  the reader does not either (`cubam.bgzf.crc32_spans` computes it, at
  about 100 GB/s). A corrupt block that still inflates to the right size
  goes undetected. Sizes and Deflate errors are checked.
- **nvCOMP 6.** cubam loads `libnvcomp.so.5`. nvCOMP 6 needs a check of the
  C API (the options struct).
- **Region queries.** Reading a region through the BAI index.
- **Barcodes.** The fragment caller packs barcodes into 64-bit codes
  (`[ACGT]{1,24}` with an optional `-N`). Other barcodes need a dictionary.
- **Fragments.** Name-sorted and single-end input.
- **CRAM** is out of scope: it needs the reference genome and codecs that
  nvCOMP does not provide.

## Development

```bash
uv sync --extra rapids-cu12 --extra bigwig
uv run --extra rapids-cu12 --extra bigwig pytest
```

The tests write small BAM files with a pure-Python writer (`tests/bamwriter.py`)
and need a GPU. The bigwig test is skipped without pyBigWig.
