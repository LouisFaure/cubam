# cubam fragments

```bash
cubam fragments atac_possorted_bam.bam -o fragments.tsv.gz
tabix -p bed fragments.tsv.gz
```

Cell Ranger keeps the fragments of its primary contigs only, listed in the
`# primary_contig=` lines of its fragments file or in `contig-defs.json`:

```bash
cubam fragments atac_possorted_bam.bam -o fragments.tsv.gz \
    --keep-chroms "$(paste -sd, primary_contigs.txt)"
```

The rules, and how the output compares with that of the CPU tool, are in
{doc}`../apps/fragments`.

## Options

```{eval-rst}
.. argparse::
   :module: cubam.cli
   :func: build_parser
   :prog: cubam
   :path: fragments
   :nodescription:
   :nodefault:
```
