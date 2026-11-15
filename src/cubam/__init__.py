"""Read BAM files on the GPU, as columns."""

from cubam._memory import auto_chunk_bytes, memory_pool
from cubam.bam import BamHeader, RecordBatch, Tag, read_bam
from cubam.bgzf import inflate_chunks

__all__ = ["BamHeader", "RecordBatch", "Tag", "auto_chunk_bytes", "inflate_chunks", "memory_pool", "read_bam"]
