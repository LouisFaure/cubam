# Benchmarks

One 10x multiome sample, read from a network file system (WekaFS):

- GEX BAM: 15 GB, 300 million records, 31% of them from the 5,368 listed cells
- ATAC BAM: 46 GB, 348 million fragments called
- cells split into 8 groups, in turn

The BAM files are the `possorted_bam.bam` of Cell Ranger ARC, with every
droplet: each tool reads all records and keeps those of the listed cells.

cubam ran on one NVIDIA A100 (80 GB), and again with all but 8 GB of the
GPU held, as on an 8 GB GPU. Times include reading the inputs (the GTF
file too) and writing the outputs. The CPU tools ran on 6 threads of an
Intel Xeon at 2.6 GHz (Platinum 8358 or Gold 6348).

## Speed

| Application | CPU tool | CPU | cubam, A100 | Speedup | cubam, 8 GB |
|---|---|---:|---:|---:|---:|
| Fragments, ATAC | sinto fragments | 2,953 s | 60 s | 49× | 75 s |
| Allele counts, GEX | cellsnp-lite | 4,668 s | 17 s | 273× | 18 s |
| Allele counts, ATAC | cellsnp-lite | 1,571 s | 61 s | 26× | 67 s |
| Coverage, GEX | samtools, bedtools, bedGraphToBigWig | 897 s | 11 s | 85× | 12 s |
| Coverage, ATAC | | | 36 s | | 54 s |
| Split, GEX | sinto filterbarcodes | 871 s | 21 s | 42× | 25 s |
| Split, ATAC | sinto filterbarcodes | 1,633 s | 54 s | 30× | 65 s |
| Feature counts, GEX | featureCounts (no cells, no UMIs) | 56 s | 16 s | 3.6× | 17 s |
| Velocity, GEX | velocyto run | 7,030 s | 30 s | 238× | 46 s |

What each CPU tool ran (`benchmarks/speed_baselines.sh`):

- **Fragments**: `sinto fragments -p 6`, every barcode. Its output is
  neither sorted nor compressed; cubam writes a sorted TSV.
- **Allele counts**: `cellsnp-lite -p 6 --minCOUNT 2`, mode 1a with the
  barcode list and 7.4 million SNPs of 1000 Genomes, `--UMItag UB` for GEX
  and `None` for ATAC. cellsnp-lite fetches the reads over each SNP, so a
  spliced read is decoded again for every SNP of its intron: GEX is slower
  than ATAC, though the file is smaller.
- **Coverage**: per group, `samtools view -F 0xF04 -q 30 -D CB:...` into
  `bedtools genomecov -bg -split`, then `bedGraphToBigWig`, 6 groups at a
  time. cubam writes CPM-normalised bigWig files.
- **Split**: `sinto filterbarcodes -p 6`, one BAM per group.
- **Feature counts**: `featureCounts -T 6 -t exon -s 1 -M` counts reads per
  gene for the whole file. It does less than cubam, which counts UMIs per
  cell and gene.
- **Velocity**: `velocyto run -@ 6`, including its sort of the BAM by cell,
  for the 17,489 cells of the filtered matrix (cubam with the same cells).

## Memory

Peak GPU memory in use (CUDA context included) and peak host memory:

| Application | GPU, A100 | GPU, 8 GB | Host |
|---|---:|---:|---:|
| Fragments, ATAC | 8.9 GB | 4.7 GB | 16 GB |
| Allele counts, GEX | 8.7 GB | 3.1 GB | 3 GB |
| Allele counts, ATAC | 6.9 GB | 4.8 GB | 6 GB |
| Coverage, GEX | 7.0 GB | 2.2 GB | 2 GB |
| Coverage, ATAC | 10.4 GB | 6.1 GB | 7 GB |
| Split, GEX | 13.2 GB | 6.3 GB | 2 GB |
| Split, ATAC | 10.6 GB | 5.6 GB | 2 GB |
| Feature counts, GEX | 8.2 GB | 3.8 GB | 2 GB |
| Velocity, GEX | 16.1 GB | 4.0 GB | 3 to 12 GB |

On the A100, cubam reads 256 MB of compressed BAM at a time, and 64 MB on
an 8 GB GPU ({func}`cubam.auto_chunk_bytes`): smaller chunks cost 5 to 20%
more time, 50% for ATAC coverage and velocity. On a small GPU, velocity
keeps its rows per molecule in host memory (12 GB here), and the
fragments table always stays there (8.4 GB for 348 million fragments).

## Duplicate marking

Duplicate marking ran on another input: one 10x scATAC sample (two NovaSeq
runs of GSE273194), aligned by Parabricks 4.8 `fq2bam --no-markdups --gpusort`
with the cell barcode in a `CB` tag. The BAM file is 10.1 GB, with 207.7
million records, sorted by coordinate and with `MC` tags. Both tools mark
duplicates within a cell barcode, on the same machines as above.

| Tool | Time | Speedup | GPU | Host |
|---|---:|---:|---:|---:|
| samtools 1.22: `collate \| fixmate -m \| sort \| markdup --barcode-tag CB`, 6 threads | 522 s | | | 14.1 GB |
| `cubam markdup --barcode-tag CB`, A100 | 35 s | 14.8× | 8.7 GB | 3.0 GB |
| `cubam markdup --barcode-tag CB`, 8 GB | 43 s | 12.1× | 4.1 GB | 2.4 GB |

samtools needs `fixmate -m` on reads grouped by name, hence the `collate`
and the second sort. Both outputs hold 76,841,695 duplicates; on the same
record order every flag is the same ({doc}`apps/markdup`). The output of
cubam is 7.4 GB, that of samtools 6.7 GB. GPU memory does not grow with the
file; host memory holds 5 bytes per record (the pair scores and the flags).

```bash
uv run --extra rapids-cu12 python benchmarks/compare_markdup.py \
    sorted.bam WORKDIR --barcode-tag CB --threads 6 [--gpu-gb 8]
```

It runs both tools (samtools on the `PATH`), then compares the flags of
every read, and writes `WORKDIR/markdup_benchmark.json`. `--gpu-gb 8`
holds all but 8 GB of the GPU for cubam and writes
`markdup_benchmark_8gb.json`; `--skip-samtools` reuses the output of
samtools.

## Reproduce

```bash
uv run --extra rapids-cu12 --extra anndata \
    python benchmarks/speed_cubam.py \
    --outs OUTS --gtf genes.gtf --sites snps.vcf \
    --rna-barcodes bc_rna.tsv --atac-barcodes bc_atac.tsv \
    --workdir WORKDIR --only velocity_gex [--gpu-gb 8]

pixi exec -s sinto -s "setuptools<81" -s samtools -s bedtools \
    -s ucsc-bedgraphtobigwig -s subread -s cellsnp-lite -s bc -- \
    bash benchmarks/speed_baselines.sh \
    OUTS genes.gtf snps.vcf bc_rna.tsv bc_atac.tsv WORKDIR [STEP...]
```

velocyto ran apart: `velocyto run -b barcodes.tsv -@ 6 --samtools-memory 3000
gex_possorted_bam.bam genes.gtf`.
