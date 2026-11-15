# Reading a BAM file

{func}`cubam.read_bam` yields one {class}`~cubam.bam.RecordBatch` per chunk
of the file. Fixed fields are CuPy arrays (`ref`, `pos`, `end`, `mapq`,
`flag`, `nref`, `npos`, `tlen`, `l_seq`, `clip_left`, `clip_right`).
Tags, read names, aligned blocks and 5' ends are decoded on request.

```python
import cubam

with cubam.memory_pool():
    for batch in cubam.read_bam("possorted_bam.bam"):
        reads = batch.take(batch.has_flags(none_of=0x904) & (batch.mapq > 30))
        cb = reads.tag("CB")
        barcodes = cb.strings()  # cuDF string column
        codes = cb.packed()  # 64-bit codes of ACGT barcodes
        rec, start, end = reads.blocks()  # aligned blocks, split at introns
```

{func}`cubam.memory_pool` sets the stream-ordered pool of RMM. Without a
pool, every buffer of a chunk is a `cudaMalloc`, which costs about as much
as the work on the chunk. The applications set it themselves.

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

## Writing BGZF

{class}`cubam.bgzf.BgzfWriter` writes any BGZF file (BAM, VCF.gz,
fragments) from data on the GPU, compressed with the Deflate compressor of
nvCOMP. Level 2 (the default) files are about 17% larger than those of
samtools (zlib level 6), level 5 files about 9% smaller.

## Limits

- **CRC32.** nvCOMP does not check the CRC32 in the BGZF trailer, and the
  reader does not either (computing it on the GPU, as the writer does, runs
  at about 100 GB/s). A corrupt block that still inflates to the right size
  goes undetected. Sizes and Deflate errors are checked.
- **Region queries.** The whole file is read: there is no query through
  the BAI index yet.
- **Barcodes.** The fragment caller packs barcodes into 64-bit codes
  (`[ACGT]{1,24}` with an optional `-N`). Other barcodes need a dictionary.
- **Fragments.** Name-sorted and single-end input are not supported.
- **CRAM** is out of scope: it needs the reference genome and codecs that
  nvCOMP does not provide.
