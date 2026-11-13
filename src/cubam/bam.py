"""
BAM records on the GPU, as columns.

:func:`read_bam` yields one :class:`RecordBatch` per chunk of the file. A
batch keeps the inflated bytes of its records on the GPU, with the offset of
each record, and decodes the fixed fields into columns. Other fields are
decoded on request: :meth:`RecordBatch.tag` for an auxiliary tag,
:meth:`RecordBatch.name_hash` for the read name, and kernels of an
application can read the bytes themselves (CIGAR, sequence, qualities).

Records are found without walking them one by one. BAM records are not
line-based: each record starts with the record length, so the offsets form a
chain, and records cross BGZF block boundaries. A kernel tests every byte
offset as a record start. It checks the length, the reference ids, a
printable read name ending in NUL and the CIGAR operations, and the ``bin``
field must equal ``reg2bin`` of the alignment. When each candidate ends where
the next starts (the usual case), the candidates are the records. Otherwise
pointer jumping from the first record keeps the true chain.
"""

from __future__ import annotations

import gzip
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cubam._kernels import append_offsets, chain, copy_spans, launch
from cubam.bgzf import inflate_chunks

#: SAM flags
PAIRED, PROPER_PAIR, UNMAPPED, MATE_UNMAPPED = 0x1, 0x2, 0x4, 0x8
REVERSE, MATE_REVERSE, READ1, READ2 = 0x10, 0x20, 0x40, 0x80
SECONDARY, QC_FAIL, DUPLICATE, SUPPLEMENTARY = 0x100, 0x200, 0x400, 0x800


@dataclass(frozen=True)
class BamHeader:
    """The header of a BAM file: SAM text, references, and where the records start."""

    text: str
    names: list[str]
    lengths: list[int]
    size: int  # bytes of the uncompressed header

    @classmethod
    def read(cls, path: str | Path) -> BamHeader:
        with gzip.open(path, "rb") as fh:

            def take(n: int) -> bytes:
                b = fh.read(n)
                if len(b) != n:
                    raise ValueError(f"{path}: truncated BAM header")
                return b

            if take(4) != b"BAM\1":
                raise ValueError(f"{path}: not a BAM file")
            (l_text,) = struct.unpack("<i", take(4))
            text = take(l_text).split(b"\0", 1)[0].decode()
            (n_ref,) = struct.unpack("<i", take(4))
            names, lengths = [], []
            size = 12 + l_text
            for _ in range(n_ref):
                (ln,) = struct.unpack("<i", take(4))
                names.append(take(ln)[:-1].decode())
                (lr,) = struct.unpack("<i", take(4))
                lengths.append(lr)
                size += 8 + ln
        return cls(text, names, lengths, size)

    @property
    def sort_order(self) -> str:
        """``SO`` of the ``@HD`` line (``unknown`` without one)."""
        first = self.text.split("\n", 1)[0]
        if first.startswith("@HD"):
            for field in first.split("\t")[1:]:
                if field.startswith("SO:"):
                    return field[3:]
        return "unknown"

    def lines(self, kind: str) -> list[str]:
        """Header lines of one kind, e.g. ``"@PG"``."""
        return [line for line in self.text.split("\n") if line.startswith(kind + "\t")]


def strings_at(data, offset, length):
    """A cuDF string column of ``data[offset[i] : offset[i] + length[i]]``."""
    import cudf
    import cupy as cp
    import pylibcudf as plc

    n = int(offset.size)
    offsets = cp.zeros(n + 1, cp.int64)
    cp.cumsum(length, out=offsets[1:])
    total = int(offsets[-1])
    chars = cp.empty(max(total, 1), cp.uint8)
    copy_spans(data, chars, offset, offsets[:-1], length)
    if total < 2**31 - 1:
        offsets = offsets.astype(cp.int32)
    col = plc.Column(
        plc.DataType(plc.TypeId.STRING),
        n,
        plc.gpumemoryview(chars),
        None,
        0,
        0,
        [plc.Column.from_cuda_array_interface(offsets)],
    )
    return cudf.Series.from_pylibcudf(col)


def unpack_acgt(codes):
    """Strings of codes made by :meth:`Tag.packed` (a cuDF string column)."""
    import cudf
    import cupy as cp
    import pylibcudf as plc

    codes = cp.asarray(codes, cp.uint64)
    n = int(codes.size)
    nb = ((codes >> np.uint64(48)) & np.uint64(31)).astype(cp.int64)
    suffix = (codes >> np.uint64(53)).astype(cp.int64) - 1
    length = nb + cp.where(suffix >= 0, 2 + (suffix >= 10).astype(cp.int64) + (suffix >= 100).astype(cp.int64), 0)
    offsets = cp.zeros(n + 1, cp.int64)
    cp.cumsum(length, out=offsets[1:])
    chars = cp.empty(max(int(offsets[-1]), 1), cp.uint8)
    launch("unpack_acgt", n, codes, offsets[:-1], np.int64(n), chars)
    col = plc.Column(
        plc.DataType(plc.TypeId.STRING),
        n,
        plc.gpumemoryview(chars),
        None,
        0,
        0,
        [plc.Column.from_cuda_array_interface(offsets.astype(cp.int32))],
    )
    return cudf.Series.from_pylibcudf(col)


class Tag:
    """An auxiliary tag of the records of a batch.

    ``type`` is the SAM type character of each record (0 when the tag is
    missing, 255 when the aux data is malformed). ``value`` holds integer
    values. String values (``Z``, ``H``) are at ``offset`` with ``length``.
    """

    def __init__(self, batch: RecordBatch, name: str):
        import cupy as cp

        if len(name) != 2:
            raise ValueError(f"A tag name has two characters, not {name!r}")
        n = len(batch)
        self.batch = batch
        self.type = cp.empty(n, cp.uint8)
        self.offset = cp.empty(n, cp.int64)
        self.length = cp.empty(n, cp.int32)
        self.value = cp.empty(n, cp.int64)
        launch(
            "find_tag",
            n,
            batch.data,
            batch.offset,
            np.int64(n),
            np.uint8(ord(name[0])),
            np.uint8(ord(name[1])),
            self.type,
            self.offset,
            self.length,
            self.value,
        )

    @property
    def present(self):
        return (self.type != 0) & (self.type != 255)

    def strings(self):
        """String values as a cuDF column (empty strings where the tag is missing or not a string)."""
        import cupy as cp

        is_str = (self.type == ord("Z")) | (self.type == ord("H"))
        return strings_at(self.batch.data, self.offset, cp.where(is_str, self.length, 0).astype(cp.int64))

    def packed(self):
        """String values packed into uint64 codes, and whether each value fits.

        A value fits when it is ``[ACGT]{1,24}`` with an optional ``-N``
        suffix of up to 3 digits, which covers 10x barcodes. The code holds 2
        bits per base, the length and the suffix (:func:`unpack_acgt` turns
        codes back into strings). Values that do not fit get code 0.
        """
        import cupy as cp

        n = len(self.batch)
        is_str = self.type == ord("Z")
        out = cp.empty(n, cp.uint64)
        ok = cp.empty(n, cp.uint8)
        length = cp.where(is_str, self.length, 0).astype(cp.int32)
        launch("pack_acgt", n, self.batch.data, self.offset, length, np.int64(n), out, ok)
        return out, ok.astype(cp.bool_)


class RecordBatch:
    """BAM records of one chunk, as columns.

    Fixed fields are attributes (cupy arrays): ``ref``, ``pos`` (0-based),
    ``end`` (alignment end, exclusive), ``mapq``, ``flag``, ``nref``,
    ``npos``, ``tlen``, ``l_seq``, and the soft clips ``clip_left`` and
    ``clip_right`` at the left and right ends of the alignment.
    """

    FIELDS = ("ref", "pos", "end", "mapq", "flag", "nref", "npos", "tlen", "l_seq", "clip_left", "clip_right")

    def __init__(self, data, offset, header: BamHeader, columns: dict):
        self.data = data
        self.offset = offset
        self.header = header
        self.columns = columns

    @classmethod
    def decode(cls, data, offset, header: BamHeader) -> RecordBatch:
        import cupy as cp

        n = int(offset.size)
        types = {
            "ref": cp.int32,
            "pos": cp.int32,
            "end": cp.int32,
            "mapq": cp.uint8,
            "flag": cp.uint16,
            "nref": cp.int32,
            "npos": cp.int32,
            "tlen": cp.int32,
            "l_seq": cp.int32,
            "clip_left": cp.uint16,
            "clip_right": cp.uint16,
        }
        cols = {k: cp.empty(n, t) for k, t in types.items()}
        launch("decode_fixed", n, data, offset, np.int64(n), *(cols[k] for k in cls.FIELDS))
        return cls(data, offset, header, cols)

    def __len__(self) -> int:
        return int(self.offset.size)

    def __getattr__(self, name: str):
        cols = self.__dict__.get("columns")
        if cols is not None and name in cols:
            return cols[name]
        raise AttributeError(name)

    def take(self, idx) -> RecordBatch:
        """The records at ``idx`` (indices or a boolean mask), sharing the bytes of this batch."""
        import cupy as cp

        idx = cp.asarray(idx)
        if idx.dtype == cp.bool_:
            idx = cp.flatnonzero(idx)
        return RecordBatch(self.data, self.offset[idx], self.header, {k: v[idx] for k, v in self.columns.items()})

    def has_flags(self, all_of: int = 0, none_of: int = 0):
        """Records with every flag of ``all_of`` and none of ``none_of``."""
        f = self.flag.astype(np.int32)
        return ((f & all_of) == all_of) & ((f & none_of) == 0)

    def tag(self, name: str) -> Tag:
        return Tag(self, name)

    def name_hash(self):
        """64-bit FNV-1a hash of the read names (mates share it)."""
        import cupy as cp

        n = len(self)
        out = cp.empty(n, cp.uint64)
        launch("name_hash", n, self.data, self.offset, np.int64(n), out)
        return out

    def names(self):
        """Read names as a cuDF string column."""
        import cupy as cp

        lrn = self.data[self.offset + 12].astype(cp.int64)
        return strings_at(self.data, self.offset + 36, lrn - 1)

    def five_prime(self):
        """The 5' end of each read, soft clips included.

        For a forward read, the alignment start minus the left soft clip. For
        a reverse read, the alignment end (exclusive) plus the right soft clip.
        """
        import cupy as cp

        rev = (self.flag & REVERSE) != 0
        return cp.where(rev, self.end + self.clip_right.astype(cp.int32), self.pos - self.clip_left.astype(cp.int32))


def _record_chain(data, n_ref: int):
    """Offsets of the complete records of ``data`` (starting at a record), and where the next one starts."""
    import cupy as cp

    n = int(data.size)
    if n == 0:
        return cp.empty(0, cp.int64), 0
    cand = append_offsets("record_candidates", data, n // 32 + 1024, np.int32(n_ref))
    if cand.size == 0:
        return cand, 0
    nxt = cp.empty(cand.size, cp.int64)
    launch("record_ends", int(cand.size), data, cand, np.int64(cand.size), nxt)
    keep = chain(cand, nxt)
    if keep is not None:
        cand, nxt = cand[keep], nxt[keep]
    return cand, int(nxt[-1])


def read_bam(path: str | Path, chunk_bytes: int = 256 << 20, **reader_kw) -> Iterator[RecordBatch]:
    """Records of a BAM file on the GPU, one :class:`RecordBatch` per chunk.

    ``chunk_bytes`` is the size of the compressed chunks (each inflates to
    about 4 to 7 times as many bytes of records). Run it inside
    :func:`cubam.memory_pool` unless an RMM pool is already set.
    """
    import cupy as cp

    header = BamHeader.read(path)
    n_ref = len(header.names)
    skip = header.size
    carry = cp.empty(0, cp.uint8)  # bytes of the incomplete last record
    for text in inflate_chunks(path, chunk_bytes, **reader_kw):
        if skip:
            k = min(skip, int(text.size))
            text = text[k:]
            skip -= k
        data = cp.concatenate([carry, text]) if carry.size else text
        del text
        offset, tail = _record_chain(data, n_ref)
        carry = data[tail:].copy()
        if offset.size:
            yield RecordBatch.decode(data, offset, header)
        del data, offset
    if carry.size:
        raise ValueError(f"{path}: truncated BAM file (incomplete record at the end)")
