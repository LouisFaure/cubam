import gzip

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from bamwriter import Read, sort_key, write_bam

from cubam import read_bam
from cubam._contigs import keep_mask, ref_ids
from cubam.bam import DUPLICATE, PAIRED, PROPER_PAIR, SECONDARY
from cubam.cells import CellIndex, _pack_acgt, _packed_barcodes, _read_barcodes
from cubam.filters import ReadFilter

REFS = [("chr1", 100_000), ("chr2", 100_000), ("chrM", 16_569)]


def read(name, pos, *, ref=0, flag=0, mapq=60, cigar="50M", cb=None, nref=None):
    tags = [("CB", "Z", cb)] if cb else []
    nref = ref if nref is None else nref
    return Read(name, flag, ref, pos, mapq, cigar, nref, pos, 0, seq="A" * 50, tags=tags)


def one_batch(tmp_path, reads):
    p = write_bam(tmp_path / "a.bam", REFS, sorted(reads, key=sort_key))
    batches = list(read_bam(p))
    assert len(batches) == 1
    return batches[0]


def test_contigs():
    names = ["chr1", "2", "chrX"]
    got = ref_ids(pd.Series(["1", "chr1", "chr2", "2", "X", "Y"]), names)
    assert got.tolist() == [0, 0, 1, 1, 2, -1]
    names = ["chr1", "chr2", "chrM"]
    assert keep_mask(names).tolist() == [True, True, True]
    assert keep_mask(names, ["chr2", "chrM"], drop=["chrM", "chrUn"]).tolist() == [False, True, False]
    with pytest.raises(ValueError, match="not in the BAM header"):
        keep_mask(names, ["chr9"])


def test_read_filter(tmp_path):
    reads = [
        read("ok", 100),
        read("lowq", 200, mapq=10),
        read("dup", 300, flag=DUPLICATE),
        read("sec", 400, flag=SECONDARY),
        read("orphan", 500, flag=PAIRED),
        read("proper", 600, flag=PAIRED | PROPER_PAIR),
        read("short", 700, cigar="10M40S"),
        read("mate_away", 800, nref=1),
        read("chrM", 900, ref=2),
    ]
    b = one_batch(tmp_path, reads)
    names = b.names().to_arrow().to_pylist()

    def kept(f, keep_ref=None):
        m = f.mask(b, keep_ref).get()
        return {n for n, k in zip(names, m, strict=True) if k}

    assert kept(ReadFilter()) == set(names)
    assert kept(ReadFilter(min_mapq=20)) == set(names) - {"lowq"}
    assert kept(ReadFilter(exclude=DUPLICATE | SECONDARY)) == set(names) - {"dup", "sec"}
    assert kept(ReadFilter(no_orphan=True)) == set(names) - {"orphan"}
    assert kept(ReadFilter(require=PAIRED)) == {"orphan", "proper"}
    assert kept(ReadFilter(min_aligned=30)) == set(names) - {"short"}
    assert kept(ReadFilter(same_contig_as_mate=True)) == set(names) - {"mate_away"}
    keep_ref = cp.asarray(keep_mask([r[0] for r in REFS], drop=["chrM"]))
    assert kept(ReadFilter(), keep_ref) == set(names) - {"chrM"}


def test_cell_index_lookup_and_groups(tmp_path):
    bcs = ["AAAC-1", "sample2_GGGT", "CCCC-1"]
    reads = [
        read("r0", 100, cb="CCCC-1"),
        read("r1", 200, cb="TTTT-1"),
        read("r2", 300),
        read("r3", 400, cb="sample2_GGGT"),
    ]
    b = one_batch(tmp_path, reads)
    idx = CellIndex(bcs, labels=["T cell", None, "B cell"])
    cell = idx.lookup(b.tag("CB"))
    assert cell.get().tolist() == [2, -1, -1, 1]
    assert idx.groups == ["T cell", "B cell"]
    assert idx.group(cell).get().tolist() == [1, -1, -1, -1]
    assert idx.group(cp.asarray([0])).get().tolist() == [0]
    (tmp_path / "b.tsv").write_text("AAAC-1\nCCCG-1\n\n")
    with gzip.open(tmp_path / "b.tsv.gz", "wt") as fh:
        fh.write("AAAC-1\nCCCG-1\n")
    assert _read_barcodes(tmp_path / "b.tsv") == _read_barcodes(tmp_path / "b.tsv.gz") == ["AAAC-1", "CCCG-1"]
    assert len(CellIndex.from_file(tmp_path / "b.tsv.gz")) == 2


def test_packed_barcodes(tmp_path):
    b = one_batch(tmp_path, [read("r0", 100, cb="ACGT-1"), read("r1", 200)])
    code, ok = _packed_barcodes(b.tag("CB"))
    assert ok.get().tolist() == [True, False]
    assert np.asarray(code.get())[1] == 0
    b = one_batch(tmp_path, [read("r0", 100, cb="sample_ACGT")])
    with pytest.raises(ValueError, match="CB values"):
        _packed_barcodes(b.tag("CB"))
    # the same codes made on the host, looked up in a CellIndex
    cells = CellIndex(["AAAC-1", "ACGT-1", "sample_ACGT"])
    codes = cp.asarray(_pack_acgt(["ACGT-1", "ACGTACGTACGTACGT-1", "AAAC-1", ""]))
    assert cells.lookup_packed(codes).get().tolist() == [1, -1, 0, -1]
