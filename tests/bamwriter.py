"""A minimal BAM and BGZF writer for tests (pure Python, no samtools)."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

EOF_BLOCK = bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")
_CIGAR_OPS = "MIDNSHP=X"
_SEQ_CODES = {c: i for i, c in enumerate("=ACMGRSVTWYHKDBN")}


def bgzf(data: bytes, block: int = 4096, eof: bool = True) -> bytes:
    """``data`` compressed as BGZF, ``block`` uncompressed bytes per block."""
    out = bytearray()
    for i in range(0, len(data), block):
        chunk = data[i : i + block]
        c = zlib.compressobj(6, zlib.DEFLATED, -15)
        payload = c.compress(chunk) + c.flush()
        bsize = 18 + len(payload) + 8
        out += b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00"
        out += struct.pack("<H", bsize - 1)
        out += payload
        out += struct.pack("<II", zlib.crc32(chunk) & 0xFFFFFFFF, len(chunk))
    if eof:
        out += EOF_BLOCK
    return bytes(out)


def reg2bin(beg: int, end: int) -> int:
    end -= 1
    for shift, offset in ((14, 4681), (17, 585), (20, 73), (23, 9), (26, 1)):
        if beg >> shift == end >> shift:
            return offset + (beg >> shift)
    return 0


def parse_cigar(cigar: str) -> list[tuple[str, int]]:
    out, num = [], ""
    for ch in cigar:
        if ch.isdigit():
            num += ch
        else:
            out.append((ch, int(num)))
            num = ""
    return out


@dataclass
class Read:
    name: str
    flag: int = 0
    ref: int = -1
    pos: int = -1  # 0-based
    mapq: int = 60
    cigar: str = ""
    nref: int = -1
    npos: int = -1
    tlen: int = 0
    seq: str = ""
    qual: bytes | None = None
    tags: list = field(default_factory=list)  # (name, type, value)

    @property
    def ref_len(self) -> int:
        return sum(n for op, n in parse_cigar(self.cigar) if op in "MDN=X")

    @property
    def end(self) -> int:
        return self.pos + self.ref_len

    def encode(self) -> bytes:
        cig = parse_cigar(self.cigar)
        if not self.seq:
            self.seq = "A" * sum(n for op, n in cig if op in "MIS=X")
        name = self.name.encode() + b"\0"
        reflen = self.ref_len
        bin_ = reg2bin(self.pos, self.pos + (reflen if reflen else 1)) if self.pos >= 0 else 4680
        l_seq = len(self.seq)
        body = struct.pack(
            "<iiBBHHHIiii",
            self.ref,
            self.pos,
            len(name),
            self.mapq,
            bin_,
            len(cig),
            self.flag,
            l_seq,
            self.nref,
            self.npos,
            self.tlen,
        )
        body += name
        body += b"".join(struct.pack("<I", n << 4 | _CIGAR_OPS.index(op)) for op, n in cig)
        codes = [_SEQ_CODES[c] for c in self.seq] + [0]
        body += bytes((codes[i] << 4) | codes[i + 1] for i in range(0, l_seq, 2))
        body += self.qual if self.qual is not None else b"\xff" * l_seq
        for tag, ty, value in self.tags:
            body += tag.encode()
            if ty == "Z":
                body += b"Z" + value.encode() + b"\0"
            elif ty == "B":
                sub, values = value
                fmt = {"c": "b", "C": "B", "s": "h", "S": "H", "i": "i", "I": "I", "f": "f"}[sub]
                body += b"B" + sub.encode() + struct.pack("<I", len(values))
                body += struct.pack(f"<{len(values)}{fmt}", *values)
            else:
                fmt = {"A": "c", "c": "b", "C": "B", "s": "h", "S": "H", "i": "i", "I": "I", "f": "f"}[ty]
                body += ty.encode() + struct.pack("<" + fmt, value.encode() if ty == "A" else value)
        return struct.pack("<i", len(body)) + body


def bam_bytes(refs: list[tuple[str, int]], reads: list[Read], text: str | None = None) -> bytes:
    if text is None:
        text = "@HD\tVN:1.6\tSO:coordinate\n" + "".join(f"@SQ\tSN:{n}\tLN:{ln}\n" for n, ln in refs)
    t = text.encode()
    out = b"BAM\1" + struct.pack("<i", len(t)) + t + struct.pack("<i", len(refs))
    for n, ln in refs:
        nb = n.encode() + b"\0"
        out += struct.pack("<i", len(nb)) + nb + struct.pack("<i", ln)
    return out + b"".join(r.encode() for r in reads)


def write_bam(path: Path, refs, reads, block: int = 4096, text: str | None = None) -> Path:
    path.write_bytes(bgzf(bam_bytes(refs, reads, text), block))
    return path


def sort_key(read: Read):
    return (read.ref if read.ref >= 0 else 1 << 30, read.pos)
