"""Read filters on SAM flags, MAPQ, mates and aligned length."""

from __future__ import annotations

from dataclasses import dataclass

from cubam.bam import PAIRED, PROPER_PAIR, UNMAPPED, RecordBatch


@dataclass(frozen=True)
class ReadFilter:
    """Which records of a batch an application keeps.

    A record is kept when it is mapped (``ref`` of at least 0 and no
    unmapped flag), has every flag of ``require`` and none of ``exclude``,
    and a MAPQ of at least ``min_mapq``.

    Parameters
    ----------
    no_orphan
        Skip paired reads that are not properly paired (cellsnp-lite).
    same_contig_as_mate
        Keep only reads whose mate is on the same contig.
    min_aligned
        Minimum number of aligned bases (CIGAR M, = and X).
    """

    min_mapq: int = 0
    require: int = 0
    exclude: int = 0
    no_orphan: bool = False
    same_contig_as_mate: bool = False
    min_aligned: int = 0

    def mask(self, batch: RecordBatch, keep_ref=None):
        """A boolean cupy mask of the kept records. ``keep_ref`` is a boolean cupy mask over the contigs."""
        import cupy as cp

        f = batch.flag.astype(cp.int32)
        ok = (batch.ref >= 0) & (batch.mapq >= self.min_mapq)
        ok &= ((f & self.require) == self.require) & ((f & (self.exclude | UNMAPPED)) == 0)
        if self.no_orphan:
            ok &= ~(((f & PAIRED) != 0) & ((f & PROPER_PAIR) == 0))
        if self.same_contig_as_mate:
            ok &= batch.ref == batch.nref
        if keep_ref is not None:
            ok &= keep_ref[cp.maximum(batch.ref, 0)]
        if self.min_aligned > 0:
            ok &= batch.aligned_length() >= self.min_aligned
        return ok
