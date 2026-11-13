"""Read BAM files on the GPU, as columns."""

from cubam._memory import memory_pool
from cubam.bam import BamHeader, RecordBatch, Tag, read_bam
from cubam.bgzf import inflate_chunks

__all__ = ["BamHeader", "RecordBatch", "Tag", "inflate_chunks", "memory_pool", "read_bam"]
