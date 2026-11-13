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
  neither does cubam. A corrupt block that still inflates to the right size
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
