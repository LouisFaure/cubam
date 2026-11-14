# Velocity counts

{func}`cubam.apps.velocity.count_velocity` counts spliced, unspliced and
ambiguous molecules per cell and gene, as `velocyto run` does.

```python
from cubam.apps.velocity import count_velocity

res = count_velocity("gex_possorted_bam.bam", "genes.gtf", "barcodes.tsv.gz")  # mask="rmsk.gtf" as velocyto -m
adata = res.to_anndata()  # layers spliced, unspliced, ambiguous (needs the anndata extra)
res.write_h5ad("sample.h5ad")
```

## Rules

The rules are those of `velocyto run` with its default logic
(Permissive10X), read from the velocyto 0.17 source:

- **Transcripts.** Each transcript is its exons and the introns between
  them. Anything 5' of an intron longer than 1 Mb is dropped.
- **Reads.** Each read is cut into segments as velocyto parses a CIGAR.
- **Matching.** A segment matches the exons and introns it overlaps by
  more than 5 bases, on the same strand as the read. A read keeps the
  transcripts with the most matches.
- **Validated introns.** An intron is validated when an unspliced read
  crosses either end of the intron into the neighbouring exon.
- **Molecules.** A molecule (cell, UMI) keeps the transcripts shared by all
  its reads. It is then classified spliced, unspliced or ambiguous from
  which exons, introns and validated introns they cover.

The two passes over the BAM file (validated introns first, then the
molecules) run on the coordinate-sorted BAM, so velocyto's cell-sorted copy
is not needed.

## velocyto's quirks

By default (`quirks=True`) three behaviours of velocyto are kept, so that
the counts are the same:

- A 5' soft clip shifts the segments of a read by the clip length.
- `chrM` of the BAM becomes `MT` but stays `M` in the GTF, so mitochondrial
  genes get no counts with Cell Ranger references.
- The last feature of each contig and strand is never matched.

`quirks=False` drops them. On chr19, chr22 and chrM of one sample, that
adds 65,809 spliced molecules on mitochondrial genes. Elsewhere, spliced
molecules change by +2.5% and unspliced by −0.6%, mostly from aligning
soft-clipped reads where they are.

## Validation

On the GEX BAM file of one multiome sample (17,489 cells), the counts equal
those of velocyto 0.17 on all 11,934,461 non-zero entries of the three
layers.
