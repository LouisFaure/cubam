# Duplicate marking

{func}`cubam.apps.markdup.mark_duplicates` marks the duplicate reads of a
coordinate-sorted BAM file with the flags of `samtools fixmate -m` followed by
`samtools markdup`. With a barcode tag, duplicates must share it, as
`samtools markdup --barcode-tag`: in 10x scATAC data, two fragments with the
same ends in two cells are not duplicates.

::::{tab-set}
:sync-group: interface

:::{tab-item} Shell
:sync: shell

```bash
cubam markdup sorted.bam --barcode-tag CB -o marked.bam -f markdup.json
samtools index marked.bam
```

See {doc}`../cli/markdup` for the options.
:::

:::{tab-item} Python
:sync: python

```python
from cubam.apps.markdup import mark_duplicates

res = mark_duplicates("sorted.bam", "marked.bam", barcode_tag="CB")
res.stats  # READ, EXAMINED, PAIRED, SINGLE, DUPLICATE PAIR, DUPLICATE SINGLE, ...
```
:::

::::

The input needs `MC` tags (BWA-MEM since 0.7.16 and Parabricks `fq2bam` write
them) and no `ms` tags: there is no `fixmate` step. It suits an aligner that
sorts on its own, such as `pbrun fq2bam --no-markdups --gpusort`, whose
duplicate marking does not know about barcodes.

## Rules

The rules are those of samtools markdup 1.22 in template mode. A read is
examined when it is primary, mapped and not QC-failed.

- **Pairs.** A read with a mapped mate gets the key of samtools: the barcode,
  its unclipped 5' end and that of its mate (from `MC`), the orientation of
  the pair, and whether the read is the leftmost. Among the reads with the
  same key, the read with the highest score stays, the others are
  duplicates. The score is the sum of the base qualities of at least 15, of
  the read and of its mate. Ties go to the smallest read name.
- **Single reads.** A read without a mapped mate is a duplicate when a read
  with a mate has the same unclipped 5' end, strand and barcode. Among single
  reads only, the highest score stays, ties going to the first in the file.

The file is read three times. The first pass sums the scores of the primary
reads per read name, which gives the score of a read plus its mate (the `ms`
tag of samtools). A score waits on the GPU only until its mate is read. The
second pass groups the reads by key. A group is decided once no read still
to come can join it: the reads still to come start at least the largest
leading clip of the file (found in the first pass) before the last position
read. The third pass sets the duplicate flag in the records and compresses
them again (nvCOMP Deflate), behind the header of the input with a `@PG`
line. GPU memory does not grow with the file, so it runs on an 8 GB GPU;
host memory holds 5 bytes per record (the pair scores and the flags). The duplicate flags
already in the input are kept, as samtools keeps them.

`level=2` (the default) writes a file about 11% larger than that of samtools
(zlib level 6).

## Validation

On the same records in the same order, every flag is the same as that of
`samtools collate | fixmate -m | sort | markdup` (samtools 1.22), with and
without `--barcode-tag`: on the 207.7 million records of a 10x scATAC sample
({doc}`../benchmarks`), and in the tests on simulated reads with clips,
indels, both strands, mates on other references, unmapped mates and
secondary alignments (`tests/test_markdup.py`).
{func}`~cubam.apps.markdup.duplicate_mask` makes this check on any file:
it ignores the duplicate flags already there, so it can be run on the output
of samtools.

The samtools pipeline sorts the reads again, and a sort may order the reads
at one position differently. Pairs do not depend on the order, but single
reads with an unmapped mate and the same score keep the first in the file.
From the same input, the outputs then hold the same number of duplicates
(76,841,695 of 207.7 million records) but swap 22,806 reads, all with an
unmapped mate.

Not implemented: optical duplicates (`-d`), marking supplementary reads
(`-S`), removing duplicates (`-r`), the sequence mode (`-m s`), read groups
as part of the key (`--use-read-groups`) and the precedence of reads that are
not QC-failed over those that are (QC-failed reads are not examined, the
samtools default). Read names are compared on their first 32 bytes; ties
between longer names with the same 32 bytes are counted as
`NAME PREFIX TIES`. Barcodes and read names are matched by 64-bit hashes.
