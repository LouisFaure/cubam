# cubam sort

```bash
cubam sort aligned.bam -o sorted.bam
samtools index sorted.bam

cubam sort -n possorted_bam.bam -o by_name.bam
```

`-n` sorts by read name in natural order, as `samtools sort -n`, and `-N`
byte by byte, as `samtools sort -N`. The output holds the same records in
the same order as that of samtools.

The order, and how the output compares with that of samtools, are in
{doc}`../apps/sort`.

## Options

```{eval-rst}
.. argparse::
   :module: cubam.cli
   :func: build_parser
   :prog: cubam
   :path: sort
   :nodescription:
   :nodefault:
```
