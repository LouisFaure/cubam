# cubam markdup

```bash
cubam markdup sorted.bam --barcode-tag CB -o marked.bam -f markdup.json
samtools index marked.bam
```

Marks duplicates as `samtools fixmate -m` followed by `samtools markdup`,
in one command on a coordinate-sorted BAM file with `MC` tags. With
`--barcode-tag`, duplicates must share the tag (`CB` for 10x scATAC). `-f`
writes the counts of the `samtools markdup -s` report, as JSON when the file
name ends in `.json`.

The rules, and how the output compares with that of samtools, are in
{doc}`../apps/markdup`.

## Options

```{eval-rst}
.. argparse::
   :module: cubam.cli
   :func: build_parser
   :prog: cubam
   :path: markdup
   :nodescription:
   :nodefault:
```
