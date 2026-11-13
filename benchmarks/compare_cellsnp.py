"""Count alleles with cubam and compare with the output of cellsnp-lite (mode 1a) on the same inputs.

    uv run --extra rapids-cu12 python benchmarks/compare_cellsnp.py BAM SITES.vcf BARCODES.tsv CELLSNP_DIR \\
        [--umi-tag UB|None] [--min-count 2]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cubam.apps.alleles import count_alleles


def read_cellsnp(d: Path):
    sites = pd.read_csv(d / "cellSNP.base.vcf", sep="\t", comment="#", header=None, usecols=[0, 1, 3, 4, 7])
    sites.columns = ["chrom", "pos", "ref", "alt", "info"]
    sites["chrom"] = sites["chrom"].astype(str)
    mats = {}
    for name in ("AD", "DP", "OTH"):
        m = pd.read_csv(d / f"cellSNP.tag.{name}.mtx", sep="\t", comment="%", header=None, skiprows=3)
        m.columns = ["site", "cell", name]
        m[["site", "cell"]] -= 1
        mats[name] = m.set_index(["site", "cell"])[name]
    counts = pd.concat(mats, axis=1).fillna(0).astype(np.int64).reset_index()
    return sites, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bam", type=Path)
    ap.add_argument("sites", type=Path)
    ap.add_argument("barcodes", type=Path)
    ap.add_argument("cellsnp", type=Path)
    ap.add_argument("--umi-tag", default="UB")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--write", type=Path)
    a = ap.parse_args()
    umi = None if a.umi_tag == "None" else a.umi_tag

    res = count_alleles(a.bam, a.sites, a.barcodes, umi_tag=umi, min_count=a.min_count)
    print(f"cubam: {len(res.sites):,} sites, {len(res.counts):,} (site, cell) entries in {res.seconds:.1f} s")
    print("stats", res.stats)
    if a.write:
        res.write_cellsnp(a.write)

    ref_sites, ref_counts = read_cellsnp(a.cellsnp)
    print(f"cellsnp-lite: {len(ref_sites):,} sites, {len(ref_counts):,} (site, cell) entries")
    key = ["chrom", "pos", "ref", "alt"]
    ours = res.sites.assign(chrom=res.sites["chrom"].astype(str))
    for df in (ours, ref_sites):
        df["dup"] = df.groupby(key).cumcount()
        df["row"] = np.arange(len(df))
    m = ours.merge(ref_sites, on=key + ["dup"], how="outer", suffixes=("", "_cs"), indicator=True)
    print("sites:", m["_merge"].value_counts().to_dict())
    for side in ("left_only", "right_only"):
        if (m["_merge"] == side).any():
            print(f"== {side} sites (sample)\n{m[m['_merge'] == side].head(8).to_string()}")
    both = m[m["_merge"] == "both"]
    info = both["info"].str.extract(r"AD=(\d+);DP=(\d+);OTH=(\d+)").astype(np.int64)
    same_totals = (info.values == both[["AD", "DP", "OTH"]].values).all(axis=1)
    print(f"site totals equal: {int(same_totals.sum()):,} of {len(both):,}")
    if not same_totals.all():
        print(both[~same_totals].head(8).to_string())

    # (site, cell) entries, on the sites both kept
    row_map = pd.Series(both["row_cs"].to_numpy(), index=both["row"].to_numpy())
    c = res.counts.to_pandas()
    c = c[c["site"].isin(row_map.index)]
    c["site"] = row_map.loc[c["site"]].to_numpy()
    rc = ref_counts[ref_counts["site"].isin(both["row_cs"])]
    mc = c.merge(rc, on=["site", "cell"], how="outer", suffixes=("", "_cs")).fillna(0)
    diff = (mc[["AD", "DP", "OTH"]].values != mc[["AD_cs", "DP_cs", "OTH_cs"]].values).any(axis=1)
    print(f"(site, cell) entries: {len(mc):,}, differing {int(diff.sum()):,}")
    if diff.any():
        bad = mc[diff].merge(both[["row_cs", "chrom", "pos", "ref", "alt"]], left_on="site", right_on="row_cs")
        print(bad.head(12).to_string())


if __name__ == "__main__":
    main()
