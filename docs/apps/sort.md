# Sort a BAM file

{func}`cubam.apps.sort.sort_bam` sorts a BAM file by coordinate, or by read
name, in the order of `samtools sort`.

::::{tab-set}
:sync-group: interface

:::{tab-item} Shell
:sync: shell

```bash
cubam sort aligned.bam -o sorted.bam
samtools index sorted.bam

cubam sort -n possorted_bam.bam -o by_name.bam
```

See {doc}`../cli/sort` for the options.
:::

:::{tab-item} Python
:sync: python

```python
from cubam.apps.sort import sort_bam

sort_bam("aligned.bam", "sorted.bam")
sort_bam("possorted_bam.bam", "by_name.bam", by="name")
```
:::

::::

The order is that of samtools 1.24:

- **By coordinate** (the default): reference, with unmapped reads last,
  then position, then the forward strand before the reverse strand.
- **By name** (`-n`, `by="name"`): read names in natural order, where
  numbers compare by value (`a7b` before `a12b`), then the first read of a
  pair before the second, and primary before supplementary before secondary
  records. `-N` (`natural=False`) compares names byte by byte.

Records that tie keep the order of the input, as in samtools. The output
has the header of the input with the `@HD` line samtools would write
(`SO:coordinate`, or `SO:queryname` with `SS:queryname:natural` or
`SS:queryname:lexicographical`) and a `@PG` line.

The sort runs on the GPU in two passes. The first pass sorts a run of
records at a time, 8 GB of records on a GPU of 24 GB or more (less on a
smaller GPU, or `--run-bytes`), and writes each sorted run to a temporary
BAM file next to the output (`-T` puts them elsewhere). A file that fits in
one run is written straight to the output. The second pass merges the
runs: it reads a piece of each, and writes the records up to the last
record read from any run. The record bytes are copied as they are.

The GPU memory in use peaks at about the run size plus 10 GB: 20 GB with
the default on an A100, and 6.4 GB on an 8 GB GPU. Larger runs save little
time, as the sort is bound by the file system and the Deflate compression.

`level=2` (the default) writes files about 13% larger than samtools (zlib
level 6). The runs take about twice the size of the input on disk while
the sort runs (31 GB for a 15 GB BAM sorted by name).

## Speed

On the GEX BAM of the {doc}`../benchmarks` (15 GB, 300 million records),
`cubam sort -n` takes 100 s on an A100 and 241 s on an 8 GB GPU, against
978 s for `samtools sort -n -@ 6`: 9.8 times faster.

## Validation

On a 10x GEX BAM of 300 million records, the output of `cubam sort -n`
holds the same records in the same order as that of `samtools sort -n`,
and sorting that output by coordinate gives the same records in the same
order as `samtools sort`. The header lines are the same, but for `@PG`.
