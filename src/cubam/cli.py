"""
The ``cubam`` command: the applications of cubam, from the shell.

Each sub-command calls one application and writes its usual files::

    cubam fragments  atac.bam -o fragments.tsv.gz
    cubam coverage   gex.bam --barcodes clusters.csv -o tracks/
    cubam split      gex.bam --barcodes clusters.csv -o bams/
    cubam sort       reads.bam -o sorted.bam
    cubam count      gex.bam --features genes.gtf --barcodes barcodes.tsv.gz -o counts/
    cubam velocity   gex.bam --gtf genes.gtf --barcodes barcodes.tsv.gz -o velocity.h5ad
    cubam cellsnp    -s gex.bam -b barcodes.tsv.gz -R snps.vcf.gz -O cellsnp/

The options are the parameters of the Python functions, with the same
defaults. The GPU libraries are only imported once a command runs, so
``--help`` works on any machine.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
import re
import sys
import time
from pathlib import Path

_log = logging.getLogger("cubam")

#: First fields of a header line in a barcode file. An empty one is the index column of pandas or R.
_HEADER_NAMES = {
    "",
    "barcode",
    "barcodes",
    "cell",
    "cells",
    "cell_id",
    "cell_barcode",
    "spot",
    "spots",
    "spot_id",
    "bead",
    "beads",
    "index",
    "obs_names",
}


class CliError(Exception):
    """An error in the input of a command, reported without a traceback."""


# ── Inputs ──────────────────────────────────────────────────────────────────


def read_barcode_file(path: str | Path, group_column: str | None = None) -> tuple[list[str], list | None]:
    """Barcodes of a file, and the group of each one when the file has a second column.

    The file has one barcode per line (``barcodes.tsv.gz`` of Cell Ranger), or
    is a CSV or TSV table whose first column holds the barcodes. Its first line
    is a header when it starts with a name such as ``barcode``, ``cell`` or
    ``spot``, or with an empty field (the index column of pandas and R).
    ``group_column`` names (or numbers, from 1) the column of the groups; by
    default it is the second column. An empty group puts the barcode in no group.
    """
    path = Path(path)
    if not path.exists():
        raise CliError(f"no such barcode file: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as f:
        lines = [line for line in f if line.strip()]
    if not lines:
        raise CliError(f"{path} has no barcodes")
    sep = "\t" if "\t" in lines[0] else "," if "," in lines[0] else None
    rows = [[c.strip() for c in r] for r in csv.reader(lines, delimiter=sep)] if sep else [[x.strip()] for x in lines]
    header = rows.pop(0) if rows[0][0].lower() in _HEADER_NAMES else None
    if not rows:
        raise CliError(f"{path} has a header but no barcodes")
    barcodes = [r[0] for r in rows]

    if group_column is None:
        col = 1 if max(len(r) for r in rows) > 1 else None
    elif group_column.isdigit():
        col = int(group_column) - 1
    elif header is not None and group_column in header:
        col = header.index(group_column)
    else:
        found = f"its columns are {', '.join(header)}" if header else "it has no header line"
        raise CliError(f"no column {group_column!r} in {path}: {found}")
    if col is None:
        return barcodes, None
    if col == 0:
        raise CliError("the group column cannot be the barcode column")
    groups = [r[col] if col < len(r) and r[col] not in ("", "NA", "nan") else None for r in rows]
    return barcodes, groups


def _barcode_index(args, need_groups: bool):
    """The :class:`~cubam.cells.CellIndex` of ``--barcodes``, with one group named ``all`` when the file has none."""
    from cubam.cells import CellIndex

    barcodes, groups = read_barcode_file(args.barcodes, args.group_column)
    if groups is None and need_groups:
        _log.info("%s has no group column: one group, 'all', of its %d barcodes", args.barcodes, len(barcodes))
        groups = ["all"] * len(barcodes)
    return CellIndex(barcodes, labels=groups)


def _size(text: str) -> int:
    """Bytes of ``64M``, ``256MB``, ``1G`` or a plain number."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kKmMgG]?)[bB]?", text.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"not a size: {text!r} (e.g. 64M, 256MB, 1G)")
    return int(float(m.group(1)) * 1024 ** "_KMG".index(m.group(2).upper() or "_"))


def _names(text: str) -> list[str]:
    """Contig names, comma-separated."""
    return [n.strip() for n in text.split(",") if n.strip()]


def _optional_int(text: str) -> int | None:
    return None if text.lower() == "none" else int(text)


def _tag_or_none(text: str) -> str | None:
    return None if text.lower() == "none" else text


# ── Commands ────────────────────────────────────────────────────────────────


def _fragments(args) -> tuple[object, str]:
    from cubam.apps.fragments import call_fragments

    res = call_fragments(
        args.bam,
        keep_chroms=args.keep_chroms,
        drop_chroms=tuple(args.drop_chroms),
        barcode_tag=args.barcode_tag,
        min_mapq=args.min_mapq,
        shift=tuple(args.shift),
        max_mate_distance=args.max_mate_distance,
        require_proper_pair=args.require_proper_pair,
        require_representative=not args.keep_unrepresented,
        min_length=args.min_length,
        chunk_bytes=args.chunk_bytes,
    )
    res.write_tsv(args.output)
    hint = f"; index it with: tabix -p bed {args.output}" if str(args.output).endswith(".gz") else ""
    return res, f"{args.output}{hint}"


def _coverage(args) -> tuple[object, str]:
    from cubam.apps import coverage as cov

    cells = _barcode_index(args, need_groups=True)
    if args.atac:
        kw = {"keep_chroms": args.keep_chroms, "drop_chroms": tuple(args.drop_chroms), "barcode_tag": args.barcode_tag}
        if args.min_mapq is not None:
            kw["min_mapq"] = args.min_mapq
        res = cov.fragment_coverage(
            args.bam,
            cells,
            mode=args.mode,
            insertion_width=args.insertion_width,
            chunk_bytes=args.chunk_bytes,
            **kw,
        )
    else:
        from dataclasses import replace

        read_filter = cov.DEFAULT_READ_FILTER
        if args.min_mapq is not None:
            read_filter = replace(read_filter, min_mapq=args.min_mapq)
        res = cov.coverage(
            args.bam,
            cells,
            cell_tag=args.barcode_tag,
            read_filter=read_filter,
            strand=args.strand,
            split_deletions=args.split_deletions,
            keep_chroms=args.keep_chroms,
            drop_chroms=tuple(args.drop_chroms),
            chunk_bytes=args.chunk_bytes,
        )
    normalize = None if args.normalize == "none" else args.normalize
    paths = res.write_bigwig(args.output, normalize=normalize, bin_size=args.bin_size, prefix=args.prefix)
    return res, f"{len(paths)} bigWig files in {args.output}"


def _split(args) -> tuple[object, str]:
    from cubam.apps.split import split_bam

    cells = _barcode_index(args, need_groups=False)
    read_filter = None
    if args.min_mapq is not None:
        from cubam.filters import ReadFilter

        read_filter = ReadFilter(min_mapq=args.min_mapq)
    res = split_bam(
        args.bam,
        cells,
        args.output,
        by="cell" if args.per_barcode else None,
        cell_tag=args.barcode_tag,
        read_filter=read_filter,
        prefix=args.prefix,
        level=args.level,
        chunk_bytes=args.chunk_bytes,
    )
    return res, f"{len(res.paths)} BAM files in {args.output}; index them with samtools index"


def _sort(args) -> tuple[object, str]:
    from cubam.apps.sort import sort_bam

    by = "name" if args.name or args.name_bytes else "coordinate"
    res = sort_bam(
        args.bam,
        args.output,
        by=by,
        natural=not args.name_bytes,
        level=args.level,
        tmp_dir=args.tmp_dir,
        run_bytes=args.run_bytes,
        chunk_bytes=args.chunk_bytes,
    )
    hint = f"; index it with: samtools index {args.output}" if by == "coordinate" else ""
    return res, f"{args.output}{hint}"


def _features(args):
    """The ``--features`` of ``count``: an IntervalIndex of a GTF or BED file, or a tag name."""
    from cubam.intervals import IntervalIndex

    path = Path(args.features)
    name = path.name.lower().removesuffix(".gz")
    if path.exists():
        if name.endswith(".gtf"):
            return IntervalIndex.from_gtf(path, feature=args.feature_type, group_by=args.group_by)
        if name.endswith(".bed"):
            return IntervalIndex.from_bed(path)
        raise CliError(f"--features {path}: expected a .gtf or .bed file (optionally .gz)")
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9]", args.features):
        return args.features  # a tag, e.g. GX
    raise CliError(f"--features {args.features}: no such file, and not a two-letter tag such as GX")


def _count(args) -> tuple[object, str]:
    from cubam.apps.features import count_features

    barcodes, _ = read_barcode_file(args.barcodes)
    output = str(args.output)
    if output.endswith(".h5ad"):
        _require_anndata()
    read_filter = None
    if args.min_mapq is not None:
        from cubam.apps.features import _PRIMARY, DUPLICATE
        from cubam.filters import ReadFilter

        read_filter = ReadFilter(min_mapq=args.min_mapq, exclude=_PRIMARY | (0 if args.umi_tag else DUPLICATE))
    res = count_features(
        args.bam,
        _features(args),
        barcodes,
        cell_tag=args.barcode_tag,
        umi_tag=args.umi_tag,
        positions=args.positions,
        strand=args.strand,
        assign=args.assign,
        min_overlap=args.min_overlap,
        read_filter=read_filter,
        keep_chroms=args.keep_chroms,
        drop_chroms=tuple(args.drop_chroms),
        chunk_bytes=args.chunk_bytes,
    )
    if output.endswith(".h5ad"):
        res.to_anndata().write_h5ad(output)
        return res, output
    res.write_mtx(output)
    return res, f"{output}/ (matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz)"


def _velocity(args) -> tuple[object, str]:
    from cubam.apps.velocity import count_velocity

    if not str(args.output).endswith(".h5ad"):
        raise CliError(f"-o {args.output}: velocity writes an .h5ad file")
    _require_anndata()
    barcodes, _ = read_barcode_file(args.barcodes)
    res = count_velocity(
        args.bam,
        args.gtf,
        barcodes,
        cell_tag=args.barcode_tag,
        umi_tag=args.umi_tag,
        mask=args.mask,
        quirks=not args.no_quirks,
        chunk_bytes=args.chunk_bytes,
    )
    res.write_h5ad(args.output)
    return res, str(args.output)


def _require_anndata() -> None:
    try:
        import anndata  # noqa: F401
    except ImportError:
        raise CliError("writing .h5ad needs anndata: pip install 'cubam[anndata]'") from None


_CELLSNP_HELP = """\
usage: cubam cellsnp -s BAM -b BARCODES -R SITES.vcf -O OUTDIR [cellsnp-lite options]

Per-cell allele counts at known sites, as cellsnp-lite in mode 1a, on the GPU.

It takes the options of cellsnp-lite and writes the same files (cellSNP.base.vcf,
cellSNP.samples.tsv and the AD, DP and OTH matrices). Mode 1a runs on the GPU
with these options:

  -s, --samFile       one BAM file
  -b, --barcodeFile   the barcodes, one per line
  -R, --regionsVCF    the sites
  -O, --outDir        the output directory
  --minCOUNT, --minMAF, --UMItag, --cellTAG, --minMAPQ, --minLEN,
  --exclFLAG, --countORPHAN, -p/--nproc (ignored)

Any other mode or option runs the real cellsnp-lite: the one named by
CUBAM_CELLSNP_FALLBACK, or else the first cellsnp-lite on PATH that is not cubam.
A link to cubam named cellsnp-lite, put first on PATH, makes tools that call
cellsnp-lite (such as numbat's pileup_and_phase.R) run on the GPU:

  ln -s "$(command -v cubam)" ~/bin/cellsnp-lite
"""


def _run_cellsnp(argv: list[str]) -> int:
    from cubam import _cellsnp

    if not argv or argv[0] in ("-h", "--help"):
        print(_CELLSNP_HELP, end="")
        return 0
    return _cellsnp.main(argv)


# ── Parser ──────────────────────────────────────────────────────────────────


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("cubam")
    except PackageNotFoundError:
        return "unknown"


def _add_common(p: argparse.ArgumentParser, *, barcodes: str | None, chroms: bool = True) -> None:
    """Options shared by the commands: the BAM, the barcodes, the contigs and the chunk size."""
    p.add_argument("bam", type=Path, help="a coordinate-sorted BAM file")
    if barcodes is not None:
        p.add_argument("--barcodes", type=Path, required=True, metavar="FILE", help=barcodes)
    p.add_argument("--barcode-tag", default="CB", metavar="TAG", help="tag of the barcode of each read (default: CB)")
    if chroms:
        p.add_argument(
            "--keep-chroms",
            type=_names,
            metavar="NAMES",
            help="contigs to keep, comma-separated, e.g. chr1,chr2 (default: all)",
        )
        p.add_argument(
            "--drop-chroms",
            type=_names,
            default=[],
            metavar="NAMES",
            help="contigs to drop, comma-separated, e.g. chrM",
        )
    p.add_argument(
        "--chunk-bytes",
        type=_size,
        metavar="SIZE",
        help="compressed bytes read at a time, e.g. 64M (default: from the free GPU memory)",
    )


_GROUPED = (
    "the barcodes, and a group for each (a cluster, a sample): a CSV or TSV file with the barcode "
    "in the first column and the group in the second, or a list of barcodes (then one group, 'all')"
)
_GROUP_COLUMN = "the column of the groups in --barcodes: its name in the header line, or its number (default: 2)"
_LIST = "the barcodes, one per line (e.g. barcodes.tsv.gz of Cell Ranger), or a table with them in the first column"


def build_parser() -> argparse.ArgumentParser:
    """The parser of the ``cubam`` command."""
    parser = argparse.ArgumentParser(
        prog="cubam",
        description="GPU-accelerated BAM parsing into columns, and the applications built on it.",
        epilog="Run 'cubam <command> --help' for the options of a command.",
    )
    parser.add_argument("--version", action="version", version=f"cubam {_version()}")
    parser.add_argument("-v", "--verbose", action="store_true", help="print the statistics of the run and tracebacks")
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    # fragments
    p = sub.add_parser(
        "fragments",
        help="ATAC fragments of a paired-end BAM, as Cell Ranger",
        description="Call the ATAC fragments of a coordinate-sorted, paired-end BAM file, with the rules of "
        "Cell Ranger, and write them as fragments.tsv.gz (BGZF when the name ends in .gz).",
    )
    _add_common(p, barcodes=None)
    p.add_argument("-o", "--output", type=Path, required=True, help="the fragments file, e.g. fragments.tsv.gz")
    p.add_argument("--min-mapq", type=int, default=31, help="minimum MAPQ of both reads (default: 31)")
    p.add_argument(
        "--shift",
        type=int,
        nargs=2,
        default=[4, -5],
        metavar=("START", "END"),
        help="Tn5 shift of the start and end (default: 4 -5)",
    )
    p.add_argument(
        "--max-mate-distance",
        type=_optional_int,
        default=5000,
        metavar="N",
        help="maximum distance between the starts of the mates, or 'none' (default: 5000)",
    )
    p.add_argument("--require-proper-pair", action="store_true", help="keep properly paired reads only")
    p.add_argument(
        "--keep-unrepresented",
        action="store_true",
        help="keep duplicates whose representative pair fails the filters (Cell Ranger drops them)",
    )
    p.add_argument("--min-length", type=int, default=1, help="minimum fragment length (default: 1)")
    p.set_defaults(run=_fragments)

    # coverage
    p = sub.add_parser(
        "coverage",
        help="coverage tracks per group of barcodes, as bigWig",
        description="Coverage of the reads (or, with --atac, of the ATAC fragments) of each group of barcodes, "
        "written as one bigWig file per group.",
    )
    _add_common(p, barcodes=_GROUPED)
    p.add_argument("--group-column", metavar="COLUMN", help=_GROUP_COLUMN)
    p.add_argument("-o", "--output", type=Path, required=True, help="the directory of the bigWig files")
    p.add_argument("--prefix", default="", help="prefix of the file names, before the group")
    p.add_argument(
        "--normalize",
        choices=("cpm", "cells", "none"),
        default="cpm",
        help="per million reads (or fragments), per barcode, or raw counts (default: cpm)",
    )
    p.add_argument("--bin-size", type=int, metavar="N", help="average over bins of N bases (default: base resolution)")
    p.add_argument("--min-mapq", type=int, help="minimum MAPQ (default: 30, or 31 with --atac)")
    p.add_argument("--strand", choices=("+", "-"), help="reads aligned to this strand only (default: both)")
    p.add_argument("--split-deletions", action="store_true", help="leave deletions uncovered, as introns")
    p.add_argument("--atac", action="store_true", help="call fragments from the BAM and cover them")
    p.add_argument(
        "--mode",
        choices=("insertions", "fragments"),
        default="insertions",
        help="with --atac: the Tn5 insertions at both ends, or whole fragments (default: insertions)",
    )
    p.add_argument("--insertion-width", type=int, default=1, help="with --atac: width of an insertion (default: 1)")
    p.set_defaults(run=_coverage)

    # split
    p = sub.add_parser(
        "split",
        help="one BAM file per group of barcodes, or per barcode",
        description="Split a BAM file into one BAM file per group of barcodes, or per barcode with "
        "--per-barcode. Each output keeps the order of the input and its header.",
    )
    _add_common(p, barcodes=_GROUPED.replace("then one group, 'all'", "then one file per barcode"), chroms=False)
    p.add_argument("--group-column", metavar="COLUMN", help=_GROUP_COLUMN)
    p.add_argument("-o", "--output", type=Path, required=True, help="the directory of the BAM files")
    p.add_argument("--per-barcode", action="store_true", help="one file per barcode, even when there are groups")
    p.add_argument("--prefix", default="", help="prefix of the file names, before the group or barcode")
    p.add_argument("--min-mapq", type=int, help="keep reads with at least this MAPQ (default: every read)")
    p.add_argument(
        "--level", type=int, default=2, choices=range(6), metavar="0-5", help="nvCOMP Deflate level (default: 2)"
    )
    p.set_defaults(run=_split)

    # sort
    p = sub.add_parser(
        "sort",
        help="sort a BAM file by coordinate or by read name, as samtools sort",
        description="Sort a BAM file by coordinate, or by read name with -n or -N, in the order of "
        "samtools sort: the output holds the same records in the same order.",
    )
    p.add_argument("bam", type=Path, help="a BAM file, in any order")
    p.add_argument("-o", "--output", type=Path, required=True, help="the sorted BAM file")
    order = p.add_mutually_exclusive_group()
    order.add_argument("-n", dest="name", action="store_true", help="sort by read name, numbers by value (natural)")
    order.add_argument("-N", dest="name_bytes", action="store_true", help="sort by read name, byte by byte")
    p.add_argument(
        "-l", "--level", type=int, default=2, choices=range(6), metavar="0-5", help="nvCOMP Deflate level (default: 2)"
    )
    p.add_argument(
        "-T",
        "--tmp-dir",
        type=Path,
        metavar="DIR",
        help="directory of the temporary files, when the input is sorted in several runs (default: that of -o)",
    )
    p.add_argument(
        "--run-bytes",
        type=_size,
        metavar="SIZE",
        help="bytes of records sorted at a time on the GPU, e.g. 4G (default: 8G, less on a GPU under 24 GB)",
    )
    p.add_argument(
        "--chunk-bytes",
        type=_size,
        metavar="SIZE",
        help="compressed bytes read at a time, e.g. 64M (default: from the free GPU memory)",
    )
    p.set_defaults(run=_sort)

    # count
    p = sub.add_parser(
        "count",
        help="counts per barcode and feature (genes, peaks, repeats)",
        description="Count the reads or UMIs of each barcode on the features of a GTF or BED file "
        "(genes, peaks, repeats), or per value of a tag (GX in a Cell Ranger BAM), with the rules of "
        "featureCounts. Writes a 10x MTX directory, or an .h5ad file.",
    )
    _add_common(p, barcodes=_LIST)
    p.add_argument(
        "--features",
        required=True,
        metavar="FILE|TAG",
        help="a GTF or BED file (optionally .gz), or the tag that names the feature of each read, e.g. GX",
    )
    p.add_argument("-o", "--output", type=Path, required=True, help="a directory (10x MTX), or a file ending in .h5ad")
    p.add_argument(
        "--feature-type", default="exon", help="GTF rows to count on: exon, or gene for gene bodies (default: exon)"
    )
    p.add_argument(
        "--group-by", default="gene_id", help="GTF attribute that groups rows into features (default: gene_id)"
    )
    p.add_argument(
        "--umi-tag",
        type=_tag_or_none,
        default="UB",
        metavar="TAG",
        help="UMI tag, or 'none' to count reads (default: UB)",
    )
    p.add_argument(
        "--positions",
        choices=("blocks", "five_prime", "three_prime"),
        default="blocks",
        help="what of a read is matched: its aligned blocks, or the aligned base at one end (default: blocks)",
    )
    p.add_argument(
        "--strand",
        choices=("sense", "antisense"),
        help="reads on the same strand as the feature (a 10x 3' library), or the opposite (default: both)",
    )
    p.add_argument(
        "--assign",
        choices=("unique", "any", "largest"),
        default="unique",
        help="a read matching several features counts for none, all, or the one with most bases (default: unique)",
    )
    p.add_argument("--min-overlap", type=int, default=1, help="minimum bases of a read on a feature (default: 1)")
    p.add_argument("--min-mapq", type=int, help="minimum MAPQ (default: 30)")
    p.set_defaults(run=_count)

    # velocity
    p = sub.add_parser(
        "velocity",
        help="spliced, unspliced and ambiguous counts, as velocyto",
        description="Count the spliced, unspliced and ambiguous molecules of each barcode and gene, as "
        "'velocyto run' does, and write them as an .h5ad file with three layers (for scVelo).",
    )
    _add_common(p, barcodes=_LIST, chroms=False)
    p.add_argument("--gtf", type=Path, required=True, help="the GTF file of the reference")
    p.add_argument("-o", "--output", type=Path, required=True, help="the .h5ad file")
    p.add_argument("--mask", type=Path, help="a GTF of repeats to skip, as velocyto run -m")
    p.add_argument("--umi-tag", default="UB", metavar="TAG", help="UMI tag (default: UB)")
    p.add_argument(
        "--no-quirks", action="store_true", help="count where the reads are, instead of velocyto's exact counts"
    )
    p.set_defaults(run=_velocity)

    # cellsnp: its arguments are those of cellsnp-lite, parsed by cubam._cellsnp
    sub.add_parser(
        "cellsnp",
        help="allele counts at known sites, a drop-in for cellsnp-lite (mode 1a)",
        add_help=False,
    )
    return parser


#: Statistics of a run worth a place in its summary line, in this order.
_SUMMARY_STATS = ("records", "reads", "fragments", "kept", "assigned", "molecules")


def _summary(res, seconds: float) -> str:
    stats = getattr(res, "stats", {}) or {}
    counts = [f"{stats[k]:,} {k}" for k in _SUMMARY_STATS if isinstance(stats.get(k), int)]
    return "; ".join([", ".join(counts), f"{seconds:.1f} s"] if counts else [f"{seconds:.1f} s"])


def main(argv: list[str] | None = None) -> int:
    """Run the ``cubam`` command (or ``cubam cellsnp``, when called as ``cellsnp-lite``)."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if Path(sys.argv[0]).name == "cellsnp-lite":
        from cubam import _cellsnp

        return _cellsnp.main(argv)
    if argv[:1] == ["cellsnp"]:
        return _run_cellsnp(argv[1:])

    parser = build_parser()
    args, rest = parser.parse_known_args(argv)
    if args.command == "cellsnp":  # after global options, e.g. cubam -v cellsnp ...
        return _run_cellsnp(rest)
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="cubam: %(message)s")
    t0 = time.time()
    try:
        res, wrote = args.run(args)
    except (CliError, FileNotFoundError, ValueError) as e:
        if args.verbose:
            raise
        print(f"cubam {args.command}: error: {e}", file=sys.stderr)
        return 1
    _log.info("%s: wrote %s (%s)", args.command, wrote, _summary(res, time.time() - t0))
    if args.verbose:
        _log.debug("statistics: %s", getattr(res, "stats", {}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
