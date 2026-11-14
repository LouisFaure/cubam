"""
Spliced, unspliced and ambiguous molecules per cell and gene, as velocyto counts them, on the GPU.

This follows ``velocyto run`` with its default logic (``Permissive10X``):

* **Transcript models.** Each transcript of the GTF is its exons and the
  introns between them. A transcript with an intron longer than 1 Mb loses
  everything 5' of it (velocyto's ``chop_if_long_intron``).
* **Reads.** Mapped, ``NH == 1``, with a cell barcode of ``cells`` and a
  UMI. A read is cut into segments as velocyto parses its CIGAR (M only;
  indels of up to 3 bases between two M are joined).
* **Read to transcripts.** A segment of 6 bases or more matches an exon or
  an intron of a transcript on the read's strand when it overlaps it by more
  than 5 bases (``MIN_FLANK``). A read keeps the transcripts with the most
  matches, minus those where a spliced read (CIGAR with N) has a segment that
  does not start or end within 6 bases of the feature it matches.
* **Validated introns.** A first pass over the BAM marks an intron as
  validated when an unspliced read crosses one of its ends into the
  neighbouring exon by more than 5 bases on each side.
* **Molecules.** A molecule is a (cell, UMI): it keeps the transcripts that
  all its reads (those with transcripts) share. When they belong to one
  gene, the molecule is spliced, unspliced or ambiguous by the rules of
  ``Permissive10X.count`` (see :func:`_classify`); otherwise it is dropped.

With ``quirks=True`` (the default) three behaviours of velocyto 0.17 are
kept, so that counts match it:

* a 5' soft clip shifts the segments of a read by its length, and ``=`` and
  ``X`` CIGAR operations are ignored;
* contig names are matched as velocyto does: ``chr`` is dropped from both
  names, and ``chrM`` of the BAM becomes ``MT``, so it does not match
  ``chrM`` of the GTF (no mitochondrial counts with Cell Ranger references);
* the last feature (by position) of each contig and strand is never matched.

``quirks=False`` aligns segments where the reads are, and matches contig
names with or without a ``chr`` prefix.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cubam._kernels import launch
from cubam.bam import REVERSE, UNMAPPED, BamHeader, RecordBatch, read_bam
from cubam.cells import CellIndex
from cubam.intervals import IntervalIndex, _read_text

MIN_FLANK = 5
SPLICE_INACCURACY = 6
LONGEST_INTRON = 1_000_000
MAX_SPAN = 3_000_000
LAYERS = ("spliced", "unspliced", "ambiguous")

# per (molecule, transcript): reads, and whether a segment matched an exon, an intron, a validated
# intron, or crossed a junction of a validated intron
#: GPU memory for the molecules of a share of the cells, as a multiple of their rows in host memory
_BUCKET_FACTOR = 16
_MT_AGG = {"r": "sum", "e": "max", "i": "max", "v": "max", "s": "max"}


def segments(batch: RecordBatch, quirks: bool = True):
    """Segments of each read as velocyto parses CIGARs: ``(rec, start, end, spliced)``.

    ``start`` and ``end`` are 1-based and inclusive; ``spliced`` (per read)
    is whether the CIGAR has an ``N``.
    """
    import cupy as cp

    n = len(batch)
    cnt = cp.zeros(n, cp.int32)
    spliced = cp.zeros(n, cp.uint8)
    d64, d32 = cp.empty(1, cp.int64), cp.empty(1, cp.int32)
    args = (batch.data, batch.offset, np.int64(n), np.int32(quirks))
    launch("vcy_segments", n, *args, np.int32(0), d64, cnt, d64, d32, d32, spliced)
    first = cp.zeros(n, cp.int64)
    if n > 1:
        first[1:] = cp.cumsum(cnt[:-1], dtype=cp.int64)
    total = int(first[-1] + cnt[-1]) if n else 0
    rec = cp.empty(total, cp.int64)
    start = cp.empty(total, cp.int32)
    end = cp.empty(total, cp.int32)
    launch("vcy_segments", n, *args, np.int32(1), first, cnt, rec, start, end, spliced)
    return rec, start, end, spliced.astype(cp.bool_)


def _gtf_chrom(c: str) -> str:
    return c[3:] if "chr" in c[:4] else c


def _bam_chrom(c: str) -> str:
    if c.startswith("chr"):
        if "_" in c:
            return c.split("_")[1]
        c = c[3:]
        return "MT" if c == "M" else c
    return c


class TranscriptModels:
    """Exons and introns of the transcripts of a GTF file, as velocyto builds them.

    ``features`` has one row per exon or intron of each transcript: ``tm``
    (transcript), ``kind`` (0 exon, 1 intron), ``start`` and ``end``
    (1-based, inclusive; an intron between abutting exons has
    ``end = start - 1``), and for introns ``down_start`` and ``up_end``: the
    start of the next and the end of the previous exon, found as velocyto
    does from ``exon_number``. ``genes`` lists the genes in velocyto's order.
    """

    def __init__(self, path: str | Path, quirks: bool = True):
        import cudf

        cols = ["chrom", "source", "type", "start", "end", "score", "strand", "frame", "attr"]
        df = cudf.read_csv(
            _read_text(Path(path)),
            sep="\t",
            comment="#",
            header=None,
            names=cols,
            usecols=["chrom", "type", "start", "end", "strand", "attr"],
            dtype={"chrom": "str", "type": "str", "start": "int64", "end": "int64", "strand": "str", "attr": "str"},
        )
        df = df[df["type"] == "exon"].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"{path}: no exon rows")
        a = df["attr"]
        ex = pd.DataFrame(
            {
                "chrom": df["chrom"].to_pandas().to_numpy(),
                "start": df["start"].to_pandas().to_numpy(),
                "end": df["end"].to_pandas().to_numpy(),
                "strand": df["strand"].to_pandas().to_numpy(),
                "trid": a.str.extract('transcript_id "([^"]+)"')[0].to_pandas().to_numpy(),
                "gene_id": a.str.extract('gene_id "([^"]+)"')[0].to_pandas().to_numpy(),
                "gene_name": a.str.extract('gene_name "([^"]+)"')[0].to_pandas().to_numpy(),
                "exon_number": a.str.extract(r'exon_number "?(\w+)')[0].to_pandas().to_numpy(),
            }
        )
        del df, a
        if ex["trid"].isna().any() or ex["gene_id"].isna().any():
            raise ValueError(f"{path}: exons without transcript_id or gene_id")
        ex["gene_name"] = ex["gene_name"].fillna(ex["gene_id"])
        ex["line"] = np.arange(len(ex))
        chroms = pd.Series(ex["chrom"].unique())
        norm = dict(zip(chroms, chroms.map(_gtf_chrom) if quirks else chroms, strict=True))
        ex["cs"] = ex["chrom"].map(norm) + ex["strand"]
        # velocyto's order: sort -k1,1 -k7,7 -k4,4n (the - strand first)
        ex["plus"] = ex["strand"] == "+"
        ex = ex.sort_values(["chrom", "plus", "start", "end", "line"], kind="stable").reset_index(drop=True)
        tm, tm_keys = pd.factorize(ex["cs"] + "\t" + ex["trid"])  # transcripts in order of first exon
        ex["tm"] = tm
        if ex["exon_number"].isna().any():  # numbered by position, as velocyto infers them
            asc = ex.groupby("tm").cumcount() + 1
            n = ex.groupby("tm")["tm"].transform("size")
            ex["exon_number"] = np.where(ex["plus"], asc, n - asc + 1)
        ex["exin"] = ex["exon_number"].astype(np.int64)

        # genes in velocyto's order: contig and strand blocks, then transcripts in order
        first_exon = ex.drop_duplicates("tm")
        gspan = ex.groupby("gene_id").agg(start=("start", "min"), end=("end", "max"))
        genes = first_exon.drop_duplicates("gene_id")[["gene_id", "gene_name", "cs"]].reset_index(drop=True)
        genes["chrom"] = genes["cs"].str[:-1]
        genes["strand"] = genes["cs"].str[-1]
        genes = genes.join(gspan, on="gene_id").drop(columns="cs")
        gene_of = pd.Series(np.arange(len(genes)), index=genes["gene_id"])
        tm_gene = gene_of.loc[first_exon["gene_id"].to_numpy()].to_numpy()
        tm_cs = first_exon["cs"].to_numpy()

        # exons and introns of each transcript, in order of position
        ex = ex.sort_values(["tm", "start", "end", "line"]).reset_index(drop=True)
        t = ex["tm"].to_numpy()
        s = ex["start"].to_numpy()
        e = ex["end"].to_numpy()
        x = ex["exin"].to_numpy()
        plus = ex["plus"].to_numpy()
        j = ex.groupby("tm").cumcount().to_numpy()
        nxt = np.r_[t[1:] == t[:-1], False]
        k = np.flatnonzero(nxt)
        feat = pd.DataFrame(
            {
                "tm": np.r_[t, t[k]],
                "pos": np.r_[2 * j, 2 * j[k] + 1],
                "kind": np.r_[np.zeros(len(t), np.int8), np.ones(k.size, np.int8)],
                "start": np.r_[s, e[k] + 1],
                "end": np.r_[e, s[k + 1] - 1],
                "exin": np.r_[x, np.where(plus[k], x[k], x[k] - 1)],
            }
        )
        feat = feat.sort_values(["tm", "pos"]).reset_index(drop=True)
        feat = self._chop(feat, tm_cs)
        self._neighbours(feat, tm_cs)
        # transcripts sorted by start (velocyto re-sorts them after chopping), for ties among features
        tstart = feat.groupby("tm")["start"].first()
        rank = np.empty(len(tm_keys), np.int64)
        order = np.lexsort((tstart.index.to_numpy(), tstart.to_numpy()))
        rank[tstart.index.to_numpy()[order]] = np.arange(order.size)
        feat["cs"] = pd.Categorical(tm_cs[feat["tm"].to_numpy()])
        feat["gene"] = tm_gene[feat["tm"].to_numpy()]
        if quirks:  # velocyto never looks at the last feature of each contig and strand
            codes = feat["cs"].cat.codes.to_numpy()
            o = np.lexsort((feat["pos"], rank[feat["tm"].to_numpy()], feat["end"], feat["start"], codes))
            cs = codes[o]
            last = o[np.r_[cs[1:] != cs[:-1], True]]
            feat["matchable"] = True
            feat.loc[last, "matchable"] = False
        else:
            feat["matchable"] = True
        self.features = feat
        self.genes = genes
        self.transcripts = pd.DataFrame({"trid": tm_keys.str.split("\t").str[1], "gene": tm_gene, "cs": tm_cs})
        self.quirks = quirks
        self._index = None
        self._device = None

    @staticmethod
    def _chop(feat: pd.DataFrame, tm_cs: np.ndarray) -> pd.DataFrame:
        """Drop what is 5' of an intron longer than 1 Mb, and number exons and introns again."""
        long_ = (feat["kind"] == 1) & (feat["end"] - feat["start"] + 1 > LONGEST_INTRON)
        if not long_.any():
            return feat
        tm = feat["tm"].to_numpy()
        tm_plus = np.char.endswith(tm_cs.astype(str), "+")
        lp = feat.loc[long_].assign(plus=tm_plus[tm[long_.to_numpy()]])
        cut_plus = lp[lp["plus"]].groupby("tm")["pos"].max()  # + strand: keep what follows the last one
        cut_minus = lp[~lp["plus"]].groupby("tm")["pos"].min()  # - strand: keep what precedes the first one
        cp_ = cut_plus.reindex(tm).to_numpy()
        cm_ = cut_minus.reindex(tm).to_numpy()
        pos = feat["pos"].to_numpy()
        keep = ~((~np.isnan(cp_) & (pos <= cp_)) | (~np.isnan(cm_) & (pos >= cm_)))
        chopped = np.unique(np.r_[cut_plus.index.to_numpy(), cut_minus.index.to_numpy()])
        feat = feat[keep].reset_index(drop=True)
        redo = feat["tm"].isin(chopped).to_numpy()
        if redo.any():
            sub = feat[redo]
            asc = sub.groupby(["tm", "kind"]).cumcount() + 1
            desc = sub.iloc[::-1].groupby(["tm", "kind"]).cumcount().iloc[::-1] + 1
            p = tm_plus[feat["tm"].to_numpy()[redo]]
            feat.loc[redo, "exin"] = np.where(p, asc.to_numpy(), desc.to_numpy())
            feat.loc[redo, "pos"] = sub.groupby("tm").cumcount().to_numpy()
        return feat

    @staticmethod
    def _neighbours(feat: pd.DataFrame, tm_cs: np.ndarray) -> None:
        """Start of the downstream and end of the upstream exon of each intron (velocyto's indexing)."""
        tm = feat["tm"].to_numpy()
        n = feat.groupby("tm")["tm"].transform("size").to_numpy()
        first = np.r_[0, np.flatnonzero(tm[1:] != tm[:-1]) + 1]
        offset = np.repeat(first, np.diff(np.r_[first, len(tm)]))
        k = feat["exin"].to_numpy()
        plus = np.char.endswith(tm_cs[tm].astype(str), "+")
        down = np.where(plus, 2 * k, n - 2 * k + 1)
        up = np.where(plus, 2 * k - 2, n - 2 * k - 1)
        start = feat["start"].to_numpy()
        end = feat["end"].to_numpy()
        intron = feat["kind"].to_numpy() == 1
        none = np.iinfo(np.int64).min // 4

        def at(ix, col):
            ix = np.where(ix < 0, ix + n, ix)  # Python indexing from the end
            ok = intron & (ix >= 0) & (ix < n)
            return np.where(ok, col[np.clip(offset + ix, 0, len(col) - 1)], none)

        feat["down_start"] = at(down, start)
        feat["up_end"] = at(up, end)

    def index(self) -> IntervalIndex:
        """The matchable features (one per row of ``features``) by contig and strand."""
        if self._index is None:
            f = self.features
            lo = np.minimum(f["start"], f["end"]).to_numpy() - 1
            hi = np.maximum(f["start"], f["end"]).to_numpy()
            ok = f["matchable"].to_numpy()
            idx = np.flatnonzero(ok)
            self._rows = idx
            self._index = IntervalIndex(
                pd.Categorical.from_codes(f["cs"].cat.codes.to_numpy()[idx], categories=f["cs"].cat.categories),
                lo[idx],
                hi[idx],
                np.arange(idx.size),
            )
        return self._index

    def device(self) -> dict:
        """Feature columns as cupy arrays (cached), and ``rows``: the feature of each interval of :meth:`index`."""
        if self._device is None:
            import cupy as cp

            f = self.features
            self.index()
            self._device = {
                "rows": cp.asarray(self._rows),
                "start": cp.asarray(f["start"].to_numpy(np.int64)),
                "end": cp.asarray(f["end"].to_numpy(np.int64)),
                "down_start": cp.asarray(f["down_start"].to_numpy(np.int64)),
                "up_end": cp.asarray(f["up_end"].to_numpy(np.int64)),
                "intron": cp.asarray(f["kind"].to_numpy() == 1),
                "tm": cp.asarray(f["tm"].to_numpy(np.int64)),
            }
        return self._device

    def query_names(self, names: list[str]) -> list[str]:
        """Names of the BAM contigs, two per contig (+ then -), as the index names them."""
        if self.quirks:
            return [_bam_chrom(n) + s for n in names for s in "+-"]
        return _match_names(names, self.features["cs"].cat.categories)


def _match_names(names: list[str], categories) -> list[str]:
    """BAM contig names, two per contig (+ then -), renamed to the contigs of an index (with or without chr)."""
    cats = {c[:-1] for c in categories}
    base = []
    for n in names:
        alt = n[3:] if n.startswith("chr") else "chr" + n
        base.append(n if n in cats or alt not in cats else alt)
    return [b + s for b in base for s in "+-"]


class RepeatMask:
    """Repeat intervals of a GTF file (``velocyto run -m``): reads inside them are not counted.

    Built as velocyto's ``read_repeats``: lines sorted by contig, strand and
    start; a line that starts within 5 bases of the end of the line before
    it extends the current interval to its own end (which can shorten it).
    With ``quirks``, an interval goes to the contig and strand of the line
    that closes it, and the last interval and the intervals of the last
    contig and strand are dropped.
    """

    def __init__(self, path: str | Path, quirks: bool = True, tolerance: int = 5):
        import cudf

        cols = ["chrom", "source", "type", "start", "end", "score", "strand", "frame", "attr"]
        df = cudf.read_csv(
            _read_text(Path(path)),
            sep="\t",
            comment="#",
            header=None,
            names=cols,
            usecols=["chrom", "source", "type", "start", "end", "strand"],
            dtype={"chrom": "str", "source": "str", "type": "str", "start": "int64", "end": "int64", "strand": "str"},
        )
        df["plus"] = df["strand"] == "+"
        df["end_str"] = df["end"].astype("str")  # velocyto breaks ties on the text of the line
        df = df.sort_values(["chrom", "plus", "start", "source", "type", "end_str"]).reset_index(drop=True)
        chrom = df["chrom"].to_pandas().to_numpy()
        start = df["start"].to_pandas().to_numpy()
        end = df["end"].to_pandas().to_numpy()
        strand = df["strand"].to_pandas().to_numpy()
        del df
        uniq = pd.unique(chrom)
        norm = {c: (c[3:] if c[:3].lower() == "chr" else c) if quirks else c for c in uniq}
        cs = pd.Series(chrom).map(norm).to_numpy().astype(object) + strand.astype(object)
        n = len(start)
        new = np.r_[True, start[1:] > end[:-1] + tolerance]  # a line opens an interval
        if not quirks:
            new[1:] |= cs[1:] != cs[:-1]
        first = np.flatnonzero(new)
        last = np.r_[first[1:] - 1, n - 1]
        ivl_start, ivl_end = start[first], end[last]
        if quirks:
            # interval g is stored by the line that opens interval g + 1, under that line's contig and strand
            owner = cs[first[1:]]
            ivl_start, ivl_end = ivl_start[:-1], ivl_end[:-1]
            owner_ok = owner != cs[-1]
            ivl_cs, ivl_start, ivl_end = owner[owner_ok], ivl_start[owner_ok], ivl_end[owner_ok]
        else:
            ivl_cs = cs[first]
        order = np.lexsort((ivl_end, ivl_start, ivl_cs))
        self.intervals = pd.DataFrame(
            {"cs": pd.Categorical(ivl_cs[order]), "start": ivl_start[order], "end": ivl_end[order]}
        )
        if quirks and len(self.intervals):  # never looked at: the last interval of each contig and strand
            codes = self.intervals["cs"].cat.codes.to_numpy()
            self.intervals = self.intervals[np.r_[codes[1:] == codes[:-1], False]].reset_index(drop=True)
        self.quirks = quirks
        iv = self.intervals
        lo = np.minimum(iv["start"], iv["end"]).to_numpy() - 1
        hi = np.maximum(iv["start"], iv["end"]).to_numpy()
        self.index = IntervalIndex(iv["cs"].array, lo, hi, np.arange(len(iv)))
        self._gpu = None

    def masked(self, names: list[str], sub: RecordBatch, rec, s, e):
        """Reads of ``sub`` whose every segment lies inside an interval (velocyto's ``has_ivls_enclosing``)."""
        import cupy as cp

        if self._gpu is None:
            self._gpu = (cp.asarray(self.intervals["start"].to_numpy()), cp.asarray(self.intervals["end"].to_numpy()))
        ms_all, me_all = self._gpu
        n = len(sub)
        if rec.size == 0 or len(self.intervals) == 0:
            return cp.zeros(n, cp.bool_)
        rev = ((sub.flag & REVERSE) != 0).astype(cp.int64)
        ref2 = sub.ref.astype(cp.int64)[rec] * 2 + rev[rec]
        a, b = s + MIN_FLANK, e - MIN_FLANK
        q, iv, _ = self.index.overlaps(names, ref2, cp.minimum(a, b) - 1, cp.maximum(a, b))
        ms, me = ms_all[iv], me_all[iv]
        sq, eq = s[q], e[q]
        contains = (sq + MIN_FLANK >= ms) & (eq - MIN_FLANK <= me) & (eq - sq > MIN_FLANK)
        partial = _flanked(ms, sq, eq) | _flanked(me, sq, eq)
        nseg = int(s.size)
        inside = cp.zeros(nseg, cp.int32)
        broken = cp.zeros(nseg, cp.int32)
        cp.add.at(inside, q, contains.astype(cp.int32))
        cp.add.at(broken, q, partial.astype(cp.int32))
        ok_seg = (inside > 0) & (broken == 0)
        # a read is masked when all its segments are
        bad = cp.zeros(n, cp.int32)
        cp.add.at(bad, rec, (~ok_seg).astype(cp.int32))
        has = cp.zeros(n, cp.bool_)
        has[rec] = True
        return has & (bad == 0)


@dataclass
class Velocity:
    """Molecules per cell and gene, by layer.

    ``counts`` is a cuDF DataFrame of ``cell`` (index into ``barcodes``),
    ``gene`` (row of ``genes``), ``spliced``, ``unspliced`` and
    ``ambiguous``, without all-zero rows.
    """

    counts: object
    barcodes: list[str]
    genes: pd.DataFrame
    stats: dict = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.barcodes), len(self.genes)

    def layer(self, name: str):
        """A cells × genes ``scipy.sparse.csr_matrix`` of one layer."""
        import scipy.sparse as sp

        c = self.counts[self.counts[name] > 0][["cell", "gene", name]].to_pandas()
        return sp.csr_matrix((c[name].to_numpy(np.uint32), (c["cell"], c["gene"])), shape=self.shape)

    def to_anndata(self):
        """An AnnData with the layers ``spliced``, ``unspliced`` and ``ambiguous``; ``X`` is ``spliced``."""
        import anndata

        layers = {k: self.layer(k) for k in LAYERS}
        var = self.genes.set_index("gene_id")
        var.index = var.index.astype(str)
        obs = pd.DataFrame(index=pd.Index(self.barcodes, name="barcode"))
        return anndata.AnnData(layers["spliced"].copy(), obs=obs, var=var, layers=layers)

    def write_h5ad(self, path: str | Path) -> None:
        """Write cells × genes with all three velocity layers (needs the ``anndata`` extra)."""
        self.to_anndata().write_h5ad(path)


def _read_segments(sub: RecordBatch, quirks: bool):
    """Segments of the reads of a batch, without reads that span more than 3 Mb (velocyto drops them)."""
    import cupy as cp

    rec, s, e, spliced = segments(sub, quirks)
    lo = cp.searchsorted(rec, rec, side="left")
    hi = cp.searchsorted(rec, rec, side="right") - 1
    k = cp.flatnonzero(e[hi] - s[lo] + 1 <= MAX_SPAN)
    return rec[k], s[k], e[k], spliced


def _matches(models: TranscriptModels, names: list[str], sub: RecordBatch, rec, s, e):
    """Segment-feature matches with velocyto's flank rule.

    Returns ``(read, feature, start, end)``: one row per segment and feature
    (a row of ``models.features``) and the segment's coordinates.
    """
    import cupy as cp

    index = models.index()
    g = models.device()
    rev = ((sub.flag & REVERSE) != 0).astype(cp.int64)
    ref2 = sub.ref.astype(cp.int64)[rec] * 2 + rev[rec]
    a = s + MIN_FLANK + 1  # a feature [fs, fe] intersects the segment when fs <= e - 6 and fe >= s + 6
    b = e - MIN_FLANK - 1
    lo, hi = cp.minimum(a, b), cp.maximum(a, b)
    q, f, _ = index.overlaps(names, ref2, lo - 1, hi)
    f = g["rows"][f]
    fs, fe = g["start"][f], g["end"][f]
    ok = (fs <= b[q]) & (fe >= a[q])
    q, f = q[ok], f[ok]
    return rec[q], f, s[q], e[q]


def _flanked(pos, s, e):
    """velocyto's ``*_overlaps_with_part_of``: ``pos`` is inside the segment, more than 5 bases from both ends."""
    return (s + MIN_FLANK < pos) & (e - MIN_FLANK > pos)


def _crosses_junction(models: TranscriptModels, f, s, e):
    """Segments that cross an intron's end into the next exon, or its start from the previous exon."""
    g = models.device()
    end_side = _flanked(g["end"][f], s, e) & _flanked(g["down_start"][f], s, e)
    start_side = _flanked(g["start"][f], s, e) & _flanked(g["up_end"][f], s, e)
    return end_side | start_side


def _classify(df):
    """velocyto's ``Permissive10X.count`` on molecules: one row per molecule with its transcripts' flags.

    Takes columns ``onlyexo``, ``onlyintron``, ``mixed`` (any transcript of
    that kind) and ``allspan`` (every transcript has a validated intron with a
    segment across one of its junctions). Returns the layer (0, 1, 2) or -1.
    """
    import cupy as cp

    oe, oi, mx, sp = (df[c].values.astype(cp.bool_) for c in ("onlyexo", "onlyintron", "mixed", "allspan"))
    spliced = oe & ~oi & ~mx
    unspliced = sp | (oi & ~oe) | (mx & ~oi & ~oe)
    ambiguous = oe & (oi | mx)
    return cp.where(spliced, 0, cp.where(unspliced, 1, cp.where(ambiguous, 2, -1))).astype(cp.int8)


def count_velocity(
    bam: str | Path,
    gtf: str | Path | TranscriptModels,
    cells: CellIndex | list[str] | str | Path,
    *,
    cell_tag: str = "CB",
    umi_tag: str = "UB",
    mask: str | Path | RepeatMask | None = None,
    quirks: bool = True,
    chunk_bytes: int | None = None,
) -> Velocity:
    """Count spliced, unspliced and ambiguous molecules per cell and gene, as ``velocyto run``.

    Parameters
    ----------
    bam
        A coordinate-sorted BAM file (velocyto's cell-sorted copy is not needed).
    gtf
        The GTF of the reference, or :class:`TranscriptModels` built from it.
    cells
        The cell barcodes; reads of other barcodes are skipped.
    mask
        Repeats (a GTF file, as ``velocyto run -m``, or a :class:`RepeatMask`):
        reads inside them are not counted.
    quirks
        Keep velocyto's quirks (see the module docs), for identical counts.
    """
    import cudf
    import cupy as cp

    from cubam import memory_pool

    t0 = time.perf_counter()
    header = BamHeader.read(bam)
    header.require_coordinate_sorted(bam)
    stats = dict.fromkeys(("records", "reads", "validated_introns", "molecules"), 0)
    with memory_pool():
        if not isinstance(cells, CellIndex):
            cells = CellIndex(cells) if isinstance(cells, list) else CellIndex.from_file(cells)
        models = gtf if isinstance(gtf, TranscriptModels) else TranscriptModels(gtf, quirks)
        if models.quirks != quirks:
            raise ValueError("The transcript models were built with another quirks setting")
        names = models.query_names(header.names)
        models.index().segments(names)
        g = models.device()
        if mask is not None and not isinstance(mask, RepeatMask):
            mask = RepeatMask(mask, quirks)
        if mask is not None:
            if mask.quirks != quirks:
                raise ValueError("The repeat mask was built with another quirks setting")
            mask_names = names if quirks else _match_names(header.names, mask.intervals["cs"].cat.categories)
            mask.index.segments(mask_names)
        tm_gene = cp.asarray(models.transcripts["gene"].to_numpy(np.int32))

        def reads(batch):
            """The reads velocyto counts: mapped, NH 1 (or no NH), with a cell barcode and a UMI."""
            ok = (batch.flag & UNMAPPED) == 0
            nh = batch.tag("NH")
            ok &= (nh.type == 0) | (nh.value == 1)
            cell = cells.lookup(batch.tag(cell_tag))
            umi = batch.tag(umi_tag)
            ok &= (cell >= 0) & (umi.type == ord("Z"))
            k = cp.flatnonzero(ok)
            sub = batch.take(k)
            return sub, cell[k], umi.hash()[k]

        # pass 1: introns crossed into an exon by an unspliced read
        validated = cp.zeros(len(models.features), cp.bool_)
        for batch in read_bam(bam, chunk_bytes):
            sub, _, _ = reads(batch)
            del batch
            rec, s, e, spliced = _read_segments(sub, quirks)
            read, feat, s, e = _matches(models, names, sub, rec, s, e)
            keep = g["intron"][feat] & ~spliced[read]
            feat, s, e = feat[keep], s[keep], e[keep]
            hit = _crosses_junction(models, feat, s, e)
            validated[feat[hit]] = True
        stats["validated_introns"] = int(validated.sum())

        # pass 2: transcripts of each read, then of each molecule
        parts, mol_parts = [], []
        for batch in read_bam(bam, chunk_bytes):
            stats["records"] += len(batch)
            sub, cell, umi = reads(batch)
            del batch
            n = len(sub)
            stats["reads"] += n
            if n == 0:
                continue
            rec, s, e, spliced = _read_segments(sub, quirks)
            if mask is not None:
                masked = mask.masked(mask_names, sub, rec, s, e)
                stats["masked"] = stats.get("masked", 0) + int(masked.sum())
                k = cp.flatnonzero(~masked[rec])
                rec, s, e = rec[k], s[k], e[k]
            read, feat, s, e = _matches(models, names, sub, rec, s, e)
            # a match needs a segment of more than 5 bases
            k = cp.flatnonzero((e - s) > MIN_FLANK)
            read, feat, s, e = read[k], feat[k], s[k], e[k]
            intron = g["intron"][feat]
            valid = intron & validated[feat]
            span = valid & _crosses_junction(models, feat, s, e)
            # a spliced read must start or end a segment within 6 bases of the feature it matches
            sense = (cp.abs(g["start"][feat] - s) <= SPLICE_INACCURACY) | (
                cp.abs(g["end"][feat] - e) <= SPLICE_INACCURACY
            )
            bad = spliced[read] & ~sense
            df = cudf.DataFrame(
                {
                    "read": read,
                    "tm": g["tm"][feat],
                    "e": ~intron,
                    "i": intron,
                    "v": valid,
                    "s": span,
                    "bad": bad,
                    "n": cp.ones(read.size, cp.int32),
                }
            )
            per = df.groupby(["read", "tm"], sort=False).agg(
                {"e": "max", "i": "max", "v": "max", "s": "max", "bad": "max", "n": "sum"}
            )
            per = per.reset_index()
            # keep the transcripts with the most matches, then drop those with a senseless splice
            best = per.groupby("read").agg({"n": "max"}).rename(columns={"n": "best"})
            per = per.merge(best, left_on="read", right_index=True)
            per = per[(per["n"] == per["best"]) & ~per["bad"]]
            rd = per["read"].values
            per["cell"] = cell[rd]
            per["umi"] = umi[rd]
            per["r"] = cp.ones(len(per), cp.int32)
            # reads with transcripts, per molecule
            used = cp.unique(rd)
            mol_parts.append(
                cudf.DataFrame({"cell": cell[used], "umi": umi[used], "R": cp.ones(used.size, cp.int32)})
                .groupby(["cell", "umi"])
                .agg({"R": "sum"})
                .reset_index()
                .to_pandas()
            )
            parts.append(per.groupby(["cell", "umi", "tm"]).agg(_MT_AGG).reset_index().to_pandas())
            del sub, df, per
        # A molecule lies in one cell: reduce the rows of a share of the cells at a time
        mt_all = pd.concat(parts, ignore_index=True) if parts else None
        mols_all = pd.concat(mol_parts, ignore_index=True) if mol_parts else None
        del parts, mol_parts
        outs = []
        for k in ("molecules", "no_shared_transcript", "multigene", "unclassified", *LAYERS):
            stats[k] = 0
        if mt_all is not None:
            free, _ = cp.cuda.runtime.memGetInfo()
            n_buckets = int(min(max(1, -(-mt_all.memory_usage().sum() * _BUCKET_FACTOR // free)), len(cells)))
            mt_bucket = mt_all["cell"].to_numpy() % n_buckets
            mol_bucket = mols_all["cell"].to_numpy() % n_buckets
            for b in range(n_buckets):
                mt = cudf.from_pandas(mt_all[mt_bucket == b])
                mols = cudf.from_pandas(mols_all[mol_bucket == b])
                outs.append(_molecules(mt, mols, tm_gene, stats))
            del mt_all, mols_all
        outs = [o for o in outs if len(o)]
        if outs:
            out = cudf.concat(outs, ignore_index=True).sort_values(["cell", "gene"]).reset_index(drop=True)
        else:
            out = cudf.DataFrame({k: cudf.Series([], dtype="int32") for k in ("cell", "gene", *LAYERS)})
    return Velocity(out, cells.barcodes, models.genes.copy(), stats, time.perf_counter() - t0)


def _molecules(mt, mols, tm_gene, stats: dict):
    """Counts per (cell, gene) of molecules, from rows per (cell, UMI, transcript) and reads per molecule."""
    import cupy as cp

    mt = mt.groupby(["cell", "umi", "tm"]).agg(_MT_AGG).reset_index()
    mols = mols.groupby(["cell", "umi"]).agg({"R": "sum"}).reset_index()
    stats["molecules"] += len(mols)
    # transcripts shared by every read of the molecule
    mt = mt.merge(mols, on=["cell", "umi"])
    mt = mt[mt["r"] == mt["R"]]
    mt["gene"] = tm_gene[mt["tm"].values]
    mt["onlyexo"] = mt["e"] & ~mt["i"]
    mt["onlyintron"] = mt["i"] & ~mt["e"]
    mt["mixed"] = mt["e"] & mt["i"] & ~mt["s"]
    mt["gmax"] = mt["gene"]
    m = mt.groupby(["cell", "umi"]).agg(
        {"gene": "min", "gmax": "max", "onlyexo": "max", "onlyintron": "max", "mixed": "max", "s": "min"}
    )
    m = m.rename(columns={"s": "allspan"}).reset_index()
    del mt
    stats["no_shared_transcript"] += len(mols) - len(m)
    multi = m["gene"] != m["gmax"]
    stats["multigene"] += int(multi.sum())
    m = m[~multi]
    layer = _classify(m)
    stats["unclassified"] += int((layer < 0).sum())
    for i, name in enumerate(LAYERS):
        m[name] = (layer == i).astype(cp.int32)
        stats[name] += int(m[name].sum())
    m = m[layer >= 0]
    out = m.groupby(["cell", "gene"]).agg(dict.fromkeys(LAYERS, "sum")).reset_index()
    out["cell"] = out["cell"].astype("int32")
    out["gene"] = out["gene"].astype("int32")
    return out
