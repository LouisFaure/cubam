"""Time each cubam app end to end (inputs read, outputs written) on one multiome sample.

    uv run --extra rapids-cu12 --extra anndata python benchmarks/speed_cubam.py \\
        --outs OUTS --gtf genes.gtf --sites snps.vcf --rna-barcodes bc_rna.tsv --atac-barcodes bc_atac.tsv \\
        --workdir WORKDIR

Cells get 8 groups (``cl0``..``cl7``, in turn). ``speed_baselines.sh`` runs
the CPU tools on the same inputs. ``--gpu-gb 8`` holds all but 8 GB of the
GPU first, as on an 8 GB GPU.

Each line of ``WORKDIR/cubam_times.tsv`` holds the step, the GPU size
(``all`` or ``--gpu-gb``), seconds, the peak GPU memory in use (CUDA context
included, polled every 20 ms) and the peak resident host memory, in GB. Run
one step per process (``--only``) for the peaks of that step alone.
"""

import argparse
import gzip
import resource
import shutil
import threading
import time
from pathlib import Path

import cupy as cp

from cubam.apps.alleles import count_alleles
from cubam.apps.coverage import coverage, fragment_coverage
from cubam.apps.features import count_features
from cubam.apps.fragments import call_fragments
from cubam.apps.split import split_bam
from cubam.apps.velocity import count_velocity
from cubam.cells import CellIndex, _read_barcodes
from cubam.intervals import IntervalIndex


def primary_contigs(fragments: Path) -> list[str]:
    out = []
    with gzip.open(fragments, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if line.startswith("# primary_contig="):
                out.append(line.strip().split("=", 1)[1])
    return out


class GpuPeak(threading.Thread):
    """Highest memory in use on the device while running (the GPU should be otherwise idle)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = 0, threading.Event()

    def run(self):
        cp.cuda.Device().use()
        while not self.stop.wait(0.02):
            free, total = cp.cuda.runtime.memGetInfo()
            self.peak = max(self.peak, total - free)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outs", type=Path, required=True, help="Cell Ranger ARC outs folder")
    ap.add_argument("--gtf", type=Path, required=True)
    ap.add_argument("--sites", type=Path, required=True, help="VCF of SNPs")
    ap.add_argument("--rna-barcodes", type=Path, required=True)
    ap.add_argument("--atac-barcodes", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="run only these steps")
    ap.add_argument("--gpu-gb", type=float, help="reserve the rest of the GPU, as on a GPU of this size")
    a = ap.parse_args()
    gex, atac = a.outs / "gex_possorted_bam.bam", a.outs / "atac_possorted_bam.bam"
    w = a.workdir
    w.mkdir(parents=True, exist_ok=True)
    rna_bc, atac_bc = _read_barcodes(a.rna_barcodes), _read_barcodes(a.atac_barcodes)
    rna_cells = CellIndex(rna_bc, labels=[f"cl{i % 8}" for i in range(len(rna_bc))])
    atac_cells = CellIndex(atac_bc, labels=[f"cl{i % 8}" for i in range(len(atac_bc))])
    prim = primary_contigs(a.outs / "atac_fragments.tsv.gz")
    cp.zeros(1).sum()  # CUDA context
    if a.gpu_gb:
        free, total = cp.cuda.runtime.memGetInfo()
        cp.cuda.runtime.malloc(int(total - a.gpu_gb * 1e9))  # held until exit, outside every pool

    def fragments():
        call_fragments(atac, keep_chroms=prim).write_tsv(w / "cubam_fragments.tsv")

    def alleles(bam, bc, umi):
        def run():
            res = count_alleles(bam, a.sites, bc, umi_tag=umi, min_count=2)
            res.write_cellsnp(w / f"cubam_alleles_{bam.name.split('_')[0]}")

        return run

    def rna_coverage():
        coverage(gex, rna_cells).write_bigwig(w / "cubam_cov_rna")

    def atac_coverage():
        fragment_coverage(atac, atac_cells, keep_chroms=prim).write_bigwig(w / "cubam_cov_atac", bin_size=10)

    def split(bam, cells):
        def run():
            out = w / f"cubam_split_{bam.name.split('_')[0]}"
            split_bam(bam, cells, out)
            shutil.rmtree(out)

        return run

    def features():
        genes = IntervalIndex.from_gtf(a.gtf, feature="exon", group_by="gene_id")
        res = count_features(gex, genes, rna_bc, umi_tag="UB", strand="sense")
        res.write_mtx(w / "cubam_features")

    def velocity():
        count_velocity(gex, a.gtf, rna_bc).write_h5ad(w / "cubam_velocity.h5ad")

    steps = {
        "fragments_atac": fragments,
        "alleles_gex": alleles(gex, rna_bc, "UB"),
        "alleles_atac": alleles(atac, atac_bc, None),
        "coverage_gex": rna_coverage,
        "coverage_atac": atac_coverage,
        "split_gex": split(gex, rna_cells),
        "split_atac": split(atac, atac_cells),
        "features_gex": features,
        "velocity_gex": velocity,
    }
    with open(w / "cubam_times.tsv", "a") as log:
        for name, run in steps.items():
            if a.only and name not in a.only:
                continue
            mon = GpuPeak()
            mon.start()
            t0 = time.perf_counter()
            run()
            cp.cuda.Device().synchronize()
            dt = time.perf_counter() - t0
            mon.stop.set()
            mon.join()
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # kB -> GB
            reserved = total - a.gpu_gb * 1e9 if a.gpu_gb else 0
            line = f"{name}\t{a.gpu_gb or 'all'}\t{dt:.1f}\t{(mon.peak - reserved) / 1e9:.1f}\t{rss:.1f}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
