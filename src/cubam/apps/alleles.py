"""
Per-cell allele counts at known sites, with the rules of cellsnp-lite (mode 1a).

The rules come from the source of cellsnp-lite 1.2.3 (``csp_fetch.c``,
``csp.c``, ``mplp.c``):

- A read counts at a site when it covers the site with an aligned base (a
  deletion or a reference skip at the site does not count).
- Read filters: mapped, MAPQ of at least ``min_mapq`` (20), none of the
  ``exclude_flags`` (unmapped, secondary, QC fail, and duplicates when there
  is no UMI), at least ``min_len`` (30) aligned bases, the cell tag in the
  barcode list, the UMI tag present. Paired reads that are not properly
  paired are skipped (``no_orphan``).
- Without UMIs every read counts, so both mates count where they overlap.
  With UMIs, one count per (site, cell, UMI): the base of the first read of
  the UMI in file order.
- Sites with more than two alleles, or a REF or ALT longer than one base,
  are skipped.
- Bases are A, C, G, T or other (N). A site is kept when the total count over
  all cells is at least ``min_count`` and the inferred minor allele has at
  least ``min_maf`` of it. REF and ALT come from the sites. AD is the ALT
  count, DP is REF + ALT, and OTH is the rest.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cubam.bam import (
    DUPLICATE,
    QC_FAIL,
    SECONDARY,
    UNMAPPED,
    BamHeader,
    read_bam,
)
from cubam.cells import CellIndex
from cubam.contigs import ref_ids
from cubam.filters import ReadFilter
from cubam.intervals import IntervalIndex

_BASE = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def read_sites(path: str | Path) -> pd.DataFrame:
    """Sites of a VCF file (plain, gzip or BGZF): ``chrom``, ``pos`` (1-based), ``ref``, ``alt``."""
    import cudf

    path = Path(path)
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        import gzip

        try:
            import cupy as cp

            from cubam.bgzf import inflate_chunks

            src = io.BytesIO(cp.concatenate(list(inflate_chunks(path))).get().tobytes())
        except ValueError:  # gzip, not BGZF
            with gzip.open(path, "rb") as fh:
                src = io.BytesIO(fh.read())
    else:
        src = str(path)
    df = cudf.read_csv(
        src,
        sep="\t",
        comment="#",
        header=None,
        usecols=[0, 1, 3, 4],
        names=["chrom", "pos", "ref", "alt"],
        dtype={"chrom": "str", "pos": "int64", "ref": "str", "alt": "str"},
    )
    return df.to_pandas()


def _base_code(alleles: pd.Series) -> np.ndarray:
    """``csp_mplp_base2int`` of the first letter: 0-3 for ACGT, 4 for N, -1 otherwise."""
    first = alleles.fillna("").str[:1].str.upper()
    return first.map(_BASE).fillna(-1).astype(np.int8).to_numpy()


def _infer_allele(bc):
    """``infer_allele`` of cellsnp-lite on an (n, 5) array: the two most frequent bases (ties: lower index)."""
    import cupy as cp

    k1 = cp.where(bc[:, 0] < bc[:, 1], 1, 0)
    k2 = 1 - k1
    m1 = cp.maximum(bc[:, 0], bc[:, 1])
    m2 = cp.minimum(bc[:, 0], bc[:, 1])
    for i in range(2, 5):
        c = bc[:, i]
        gt1 = c > m1
        gt2 = ~gt1 & (c > m2)
        m2, k2 = cp.where(gt1, m1, cp.where(gt2, c, m2)), cp.where(gt1, k1, cp.where(gt2, i, k2))
        m1, k1 = cp.where(gt1, c, m1), cp.where(gt1, i, k1)
    return k1, k2


def _infer_alt(bc, ref):
    """``infer_alt`` of cellsnp-lite: the most frequent base other than REF (first on ties)."""
    import cupy as cp

    masked = bc.copy()
    masked[cp.arange(bc.shape[0]), cp.maximum(ref, 0)] = -1
    k = cp.argmax(masked, axis=1)
    none = masked.max(axis=1) <= 0
    return cp.where(none, cp.where(ref == 0, 1, 0), k).astype(cp.int8)


@dataclass
class AlleleCounts:
    """Allele counts at the kept sites.

    ``sites`` holds the kept sites in input order (``chrom``, ``pos``,
    ``ref``, ``alt``, and the totals ``AD``, ``DP``, ``OTH``). ``counts`` is a
    cuDF DataFrame of ``site`` (0-based row of ``sites``), ``cell`` (0-based
    index into ``barcodes``), ``AD``, ``DP`` and ``OTH``.
    """

    sites: pd.DataFrame
    barcodes: list[str]
    counts: object
    stats: dict = field(default_factory=dict)
    seconds: float = 0.0

    def write_cellsnp(self, outdir: str | Path) -> None:
        """Write the files of cellsnp-lite: ``cellSNP.base.vcf``, ``cellSNP.samples.tsv`` and the AD/DP/OTH matrices."""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        s = self.sites
        with open(outdir / "cellSNP.base.vcf", "w") as fh:
            fh.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
            lines = (
                s["chrom"].astype(str)
                + "\t"
                + s["pos"].astype(str)
                + "\t.\t"
                + s["ref"]
                + "\t"
                + s["alt"]
                + "\t.\tPASS\tAD="
                + s["AD"].astype(str)
                + ";DP="
                + s["DP"].astype(str)
                + ";OTH="
                + s["OTH"].astype(str)
            )
            fh.write("\n".join(lines) + ("\n" if len(lines) else ""))
        (outdir / "cellSNP.samples.tsv").write_text("".join(b + "\n" for b in self.barcodes))
        c = self.counts.sort_values(["site", "cell"]).to_pandas()
        for name in ("AD", "DP", "OTH"):
            nz = c[c[name] > 0]
            with open(outdir / f"cellSNP.tag.{name}.mtx", "w") as fh:
                fh.write("%%MatrixMarket matrix coordinate integer general\n%\n")
                fh.write(f"{len(s)}\t{len(self.barcodes)}\t{len(nz)}\n")
                body = (nz["site"] + 1).astype(str) + "\t" + (nz["cell"] + 1).astype(str) + "\t" + nz[name].astype(str)
                fh.write("\n".join(body) + ("\n" if len(body) else ""))


def count_alleles(
    bam: str | Path,
    sites: str | Path | pd.DataFrame,
    barcodes: list[str] | str | Path,
    *,
    cell_tag: str = "CB",
    umi_tag: str | None = "UB",
    min_mapq: int = 20,
    min_len: int = 30,
    exclude_flags: int | None = None,
    no_orphan: bool = True,
    min_count: int = 20,
    min_maf: float = 0.0,
    chunk_bytes: int | None = None,
) -> AlleleCounts:
    """Count the alleles of each cell at known sites, as cellsnp-lite mode 1a.

    Parameters
    ----------
    bam
        A coordinate-sorted BAM file.
    sites
        A VCF file, or a DataFrame with ``chrom``, ``pos`` (1-based), ``ref``
        and ``alt``.
    barcodes
        The cell barcodes, or a file with one per line. Their order is the
        order of the cells in the output.
    umi_tag
        UMI tag (``UB`` for Cell Ranger GEX), or ``None`` to count every read
        (ATAC). cellsnp-lite ``--UMItag Auto`` is ``UB`` with a barcode list.
    exclude_flags
        Reads with any of these flags are skipped. Default: unmapped,
        secondary and QC fail, plus duplicates when ``umi_tag`` is ``None``.
    min_count, min_maf
        Site filters (cellsnp-lite ``--minCOUNT`` and ``--minMAF``).
    """
    import cudf
    import cupy as cp

    from cubam import memory_pool

    t0 = time.perf_counter()
    header = BamHeader.read(bam)
    header.require_coordinate_sorted(bam)
    s = read_sites(sites) if not isinstance(sites, pd.DataFrame) else sites.reset_index(drop=True)
    # cellsnp-lite skips records with more than two alleles, or a REF or ALT longer than one base
    ref_s, alt_s = s["ref"].fillna("").astype(str), s["alt"].fillna(".").astype(str)
    single = (ref_s.str.len() <= 1) & ~alt_s.str.contains(",", regex=False) & ((alt_s.str.len() <= 1) | (alt_s == "."))
    n_skipped = int((~single).sum())
    s = s[single.to_numpy()].reset_index(drop=True)
    if exclude_flags is None:
        exclude_flags = UNMAPPED | SECONDARY | QC_FAIL | (0 if umi_tag else DUPLICATE)

    read_filter = ReadFilter(min_mapq=min_mapq, exclude=exclude_flags, no_orphan=no_orphan)

    ref_id = ref_ids(s["chrom"].astype(str), header.names)
    found = ref_id >= 0
    key = (ref_id << 32) | (s["pos"].to_numpy(np.int64) - 1)
    order = np.argsort(np.where(found, key, np.iinfo(np.int64).max), kind="stable")
    order = order[found[order]]  # sorted row -> input row

    stats = dict.fromkeys(("records", "reads", "read_sites"), 0)
    with memory_pool():
        cells_index = CellIndex(barcodes) if isinstance(barcodes, list) else CellIndex.from_file(barcodes)
        site_key = cp.asarray(key[order])
        site_pos = cp.asarray((key[order] & 0xFFFFFFFF).astype(np.int32))
        # each site is a feature of one base, numbered in sorted order
        sorted_ref, sorted_pos = key[order] >> 32, key[order] & 0xFFFFFFFF
        chrom = pd.Categorical.from_codes(sorted_ref, categories=pd.Index(header.names))
        index = IntervalIndex(chrom, sorted_pos, sorted_pos + 1, np.arange(order.size))
        parts = []
        ordinal0 = 0
        for batch in read_bam(bam, chunk_bytes):
            n = len(batch)
            stats["records"] += n
            idx = cp.flatnonzero(read_filter.mask(batch))
            sub = batch.take(idx)
            del batch
            cell = cells_index.lookup(sub.tag(cell_tag))
            keep = (cell >= 0) & (sub.aligned_length() >= min_len)
            if umi_tag:
                ut = sub.tag(umi_tag)
                keep &= ut.type == ord("Z")
                umi = ut.hash()
            k = cp.flatnonzero(keep)
            sub = sub.take(k)
            cell = cell[k]
            ordinal = idx[k] + ordinal0
            ordinal0 += n
            if umi_tag:
                umi = umi[k]
            stats["reads"] += len(sub)
            if not len(sub):
                continue
            # (read, site) pairs: the sites in [pos, end) of each read
            rec, site, _ = index.overlaps(header.names, sub.ref, sub.pos, sub.end)
            if rec.size == 0:
                continue
            base = sub.base_at(rec, site_pos[site])
            good = cp.flatnonzero(base >= 0)
            rec, site, base = rec[good], site[good], base[good]
            stats["read_sites"] += int(rec.size)
            df = cudf.DataFrame({"site": site, "cell": cell[rec], "base": base})
            if umi_tag:
                df["umi"] = umi[rec]
                df["ordinal"] = ordinal[rec]
                # the first read of each UMI at a site gives its base
                df = df.sort_values(["site", "cell", "umi", "ordinal"]).drop_duplicates(
                    ["site", "cell", "umi"], keep="first"
                )
            else:
                df = df.groupby(["site", "cell", "base"], sort=False).size().rename("n").reset_index()
            parts.append(df)
        if not parts:
            raise ValueError("No read covers any site")
        df = cudf.concat(parts, ignore_index=True)
        del parts
        if umi_tag:
            df = df.sort_values(["site", "cell", "umi", "ordinal"])
            df = df.drop_duplicates(["site", "cell", "umi"], keep="first")
            df = df.groupby(["site", "cell", "base"], sort=False).size().rename("n").reset_index()
        else:
            df = df.groupby(["site", "cell", "base"], sort=False).agg({"n": "sum"}).reset_index()

        # Site totals per base, the site filters, and REF/ALT
        n_sites = int(site_key.size)
        bc = cp.zeros((n_sites, 5), cp.int64)
        cp.add.at(bc, (df["site"].values, df["base"].values.astype(cp.int64)), df["n"].values.astype(cp.int64))
        tc = bc.sum(axis=1)
        _, inf_alt = _infer_allele(bc)
        kept = (tc >= min_count) & (tc > 0) & ~(bc[cp.arange(n_sites), inf_alt] < tc * min_maf)
        ref = cp.asarray(_base_code(s["ref"])[order])
        alt = cp.asarray(_base_code(s["alt"])[order])
        inf_ref, inf_alt = _infer_allele(bc)
        no_ref = ref < 0
        ref = cp.where(no_ref, inf_ref, ref).astype(cp.int8)
        alt = cp.where(no_ref, inf_alt, cp.where(alt < 0, _infer_alt(bc, ref), alt)).astype(cp.int8)

        # Per (site, cell): AD, DP, OTH
        b = df["base"].values
        st = df["site"].values
        nn = df["n"].values.astype(cp.int64)
        is_alt = b == alt[st]
        is_ref = b == ref[st]
        cells = cudf.DataFrame(
            {
                "site": st,
                "cell": df["cell"].values,
                "AD": cp.where(is_alt, nn, 0),
                "DP": cp.where(is_alt | is_ref, nn, 0),
                "OTH": cp.where(is_alt | is_ref, 0, nn),
            }
        )
        cells = cells[kept[st]].groupby(["site", "cell"], sort=False).sum().reset_index()

        # Kept sites in input order, numbered from 0
        kept_rows = cp.flatnonzero(kept)
        input_row = cp.asarray(order)[kept_rows]
        rank = cp.argsort(input_row)
        new_index = cp.empty(n_sites, cp.int64)
        new_index[kept_rows[rank]] = cp.arange(kept_rows.size)
        cells["site"] = new_index[cells["site"].values]
        letters = np.array(list("ACGTN"))
        rows = input_row[rank].get()
        ad = bc[kept_rows[rank], alt[kept_rows[rank]].astype(cp.int64)]
        dp = ad + bc[kept_rows[rank], ref[kept_rows[rank]].astype(cp.int64)]
        out_sites = pd.DataFrame(
            {
                "chrom": s["chrom"].to_numpy()[rows],
                "pos": s["pos"].to_numpy()[rows],
                "ref": letters[ref[kept_rows[rank]].get()],
                "alt": letters[alt[kept_rows[rank]].get()],
                "AD": ad.get(),
                "DP": dp.get(),
                "OTH": (tc[kept_rows[rank]] - dp).get(),
            }
        )
        cells = cells.sort_values(["site", "cell"]).reset_index(drop=True)
    stats.update(sites=len(s) + n_skipped, sites_skipped=n_skipped, sites_on_contigs=int(found.sum()))
    stats["sites_kept"] = len(out_sites)
    return AlleleCounts(out_sites, cells_index.barcodes, cells, stats, time.perf_counter() - t0)
