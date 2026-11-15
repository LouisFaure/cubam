import gzip
import importlib.util

import pandas as pd
import pytest
from bamwriter import Read, sort_key, write_bam

from cubam.cli import CliError, _size, build_parser, main, read_barcode_file

gpu = pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="needs cupy and a GPU")

REFS = [("chr1", 1_000_000), ("chrM", 16_569)]
BCS = ["ACGTACGTACGTACGT-1", "TTTTCCCCGGGGAAAA-1", "GGGGAAAACCCCTTTT-1"]


# ── Barcode files (no GPU) ──────────────────────────────────────────────────


def test_barcode_list(tmp_path):
    p = tmp_path / "barcodes.tsv.gz"
    with gzip.open(p, "wt") as f:
        f.write("\n".join(BCS) + "\n\n")
    assert read_barcode_file(p) == (BCS, None)


@pytest.mark.parametrize(
    "text",
    [
        "Barcode,Cluster\n{0},1\n{1},2\n{2},\n",  # Cell Ranger clusters.csv
        ",leiden\n{0},1\n{1},2\n{2},\n",  # pandas: obs[["leiden"]].to_csv()
        '"","x"\n"{0}","1"\n"{1}","2"\n"{2}",NA\n',  # R: write.csv(Idents(obj))
        "{0}\t1\n{1}\t2\n{2}\t\n",  # TSV without a header line
    ],
)
def test_barcode_table(tmp_path, text):
    p = tmp_path / "clusters.csv"
    p.write_text(text.format(*BCS))
    assert read_barcode_file(p) == (BCS, ["1", "2", None])


def test_group_column(tmp_path):
    p = tmp_path / "meta.csv"
    p.write_text(f"barcode,sample,cluster\n{BCS[0]},a,1\n{BCS[1]},b,2\n")
    assert read_barcode_file(p, "cluster")[1] == ["1", "2"]
    assert read_barcode_file(p, "2")[1] == ["a", "b"]
    with pytest.raises(CliError, match="no column 'leiden'.*barcode, sample, cluster"):
        read_barcode_file(p, "leiden")
    with pytest.raises(CliError, match="no such barcode file"):
        read_barcode_file(tmp_path / "missing.tsv")


def test_sizes():
    assert _size("64M") == 64 << 20
    assert _size("256MB") == 256 << 20
    assert _size("1G") == 1 << 30
    assert _size("1000") == 1000


def test_help_of_every_command(capsys):
    for cmd in ("fragments", "coverage", "split", "count", "velocity", "markdup"):
        with pytest.raises(SystemExit) as e:
            build_parser().parse_args([cmd, "--help"])
        assert e.value.code == 0
    assert main(["cellsnp", "--help"]) == 0
    assert "cellsnp-lite" in capsys.readouterr().out


def test_input_errors(tmp_path, capsys):
    bc = tmp_path / "bc.tsv"
    bc.write_text("\n".join(BCS) + "\n")
    assert main(["velocity", "x.bam", "--gtf", "g.gtf", "--barcodes", str(bc), "-o", "out.loom"]) == 1
    assert "writes an .h5ad file" in capsys.readouterr().err
    assert main(["count", "x.bam", "--features", "genes.txt", "--barcodes", str(bc), "-o", "o"]) == 1
    assert "no such file, and not a two-letter tag" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["fragments", "x.bam", "-o", "f.tsv.gz", "--no-such-option"])


# ── Commands (GPU) ──────────────────────────────────────────────────────────


def pairs(n=30):
    """Read pairs of the barcodes in ``BCS``, 50 bp reads 150 bp apart."""
    reads = []
    for i in range(n):
        bc, pos = BCS[i % len(BCS)], 10_000 + 1_000 * i
        tags = [("CB", "Z", bc), ("UB", "Z", f"{'ACGT'[i % 4] * 10}"), ("GX", "Z", f"g{i % 2}")]
        reads.append(Read(f"r{i}", 0x1 | 0x2 | 0x20 | 0x40, 0, pos, 60, "50M", 0, pos + 150, 200, tags=tags))
        reads.append(Read(f"r{i}", 0x1 | 0x2 | 0x10 | 0x80, 0, pos + 150, 60, "50M", 0, pos, -200, tags=tags))
    return sorted(reads, key=sort_key)


@gpu
def test_fragments(tmp_path):
    from cubam.apps.fragments import call_fragments

    bam = write_bam(tmp_path / "a.bam", REFS, pairs())
    out = tmp_path / "fragments.tsv.gz"
    assert main(["fragments", str(bam), "-o", str(out), "--chunk-bytes", "1M"]) == 0
    got = pd.read_csv(out, sep="\t", header=None)
    want = call_fragments(bam).to_frame().to_pandas()
    assert len(got) == len(want) == 30
    assert got[3].tolist() == want["barcode"].astype(str).tolist()


@gpu
def test_count_split_coverage(tmp_path):
    bam = write_bam(tmp_path / "a.bam", REFS, pairs())
    groups = tmp_path / "clusters.csv"
    groups.write_text("barcode,cluster\n" + "".join(f"{b},c{i % 2}\n" for i, b in enumerate(BCS)))

    assert main(["count", str(bam), "--features", "GX", "--barcodes", str(groups), "-o", str(tmp_path / "mtx")]) == 0
    assert {p.name for p in (tmp_path / "mtx").iterdir()} == {"matrix.mtx.gz", "barcodes.tsv.gz", "features.tsv.gz"}

    bed = tmp_path / "peaks.bed"
    bed.write_text("chr1\t9000\t20000\tpeak1\n")
    args = ["--barcodes", str(groups), "--umi-tag", "none", "-o", str(tmp_path / "bed")]
    assert main(["count", str(bam), "--features", str(bed), *args]) == 0

    assert main(["split", str(bam), "--barcodes", str(groups), "-o", str(tmp_path / "bams")]) == 0
    assert {p.name for p in (tmp_path / "bams").glob("*.bam")} == {"c0.bam", "c1.bam"}

    pytest.importorskip("pyBigWig")
    assert main(["coverage", str(bam), "--barcodes", str(groups), "-o", str(tmp_path / "bw")]) == 0
    assert {p.name for p in (tmp_path / "bw").glob("*.bw")} == {"c0.bw", "c1.bw"}
    assert main(["coverage", str(bam), "--barcodes", str(groups), "--atac", "-o", str(tmp_path / "atac")]) == 0
    assert {p.name for p in (tmp_path / "atac").glob("*.bw")} == {"c0.bw", "c1.bw"}
