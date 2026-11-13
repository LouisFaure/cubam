"""Call fragments from a Cell Ranger ARC sample and compare them with its atac_fragments.tsv.gz.

uv run --extra rapids-cu12 python benchmarks/compare_cellranger.py <cellranger outs dir>
"""

import argparse
import gzip
import io
import time
from pathlib import Path

import cudf
import cupy as cp

from cubam import inflate_chunks
from cubam.apps.fragments import call_fragments


def primary_contigs(fragments: Path) -> list[str]:
    out = []
    with gzip.open(fragments, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if line.startswith("# primary_contig="):
                out.append(line.strip().split("=", 1)[1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outs", type=Path)
    ap.add_argument("--chunk-mb", type=int, default=256)
    a = ap.parse_args()
    bam = a.outs / "atac_possorted_bam.bam"
    ref = a.outs / "atac_fragments.tsv.gz"
    prim = primary_contigs(ref)
    print(f"{bam} ({bam.stat().st_size / 1e9:.2f} GB), {len(prim)} primary contigs")

    frags = call_fragments(bam, keep_chroms=prim or None, chunk_bytes=a.chunk_mb << 20)
    print(f"cubam: {len(frags):,} fragments in {frags.seconds:.2f} s")
    print("stats", frags.stats)
    t = frags.table
    key = t["ref"].astype("int64") * (1 << 32) + t["start"].astype("int64")
    print("sorted by position:", bool((key.diff().dropna() >= 0).all()))

    ts = time.perf_counter()
    cols = ["chrom", "start", "end", "barcode", "count"]
    # cuDF reads only the first gzip member of a BGZF file: inflate it with cubam
    text = cp.concatenate(list(inflate_chunks(ref)))
    cr = cudf.read_csv(io.BytesIO(text.get().tobytes()), sep="\t", comment="#", header=None, names=cols)
    print(f"Cell Ranger: {len(cr):,} fragments, read in {time.perf_counter() - ts:.1f} s")

    gpu = frags.to_frame()
    del frags
    # one chromosome at a time: a string merge of both tables at once does not fit on a 48 GB GPU
    n = {"both": 0, "only cubam": 0, "only Cell Ranger": 0, "count differs": 0}
    samples = []
    for chrom in gpu["chrom"].unique().to_arrow().to_pylist() + ["<cr only>"]:
        if chrom == "<cr only>":
            g_ = gpu.head(0)
            c_ = cr[~cr["chrom"].isin(gpu["chrom"].unique())]
        else:
            g_, c_ = gpu[gpu["chrom"] == chrom], cr[cr["chrom"] == chrom]
        m = g_.merge(c_, on=cols[:4], how="outer", suffixes=("_gpu", "_cr"))
        g, c = m["count_gpu"].notna(), m["count_cr"].notna()
        both = m[g & c]
        n["both"] += len(both)
        n["only cubam"] += int((g & ~c).sum())
        n["only Cell Ranger"] += int((~g & c).sum())
        n["count differs"] += int((both["count_gpu"] != both["count_cr"]).sum())
        if bool((g ^ c).any()):
            samples.append(m[g ^ c].head(5).to_pandas())
    print("rows: " + "  ".join(f"{k} {v:,}" for k, v in n.items()))
    for df in samples[:4]:
        print(df.to_string())


if __name__ == "__main__":
    main()
