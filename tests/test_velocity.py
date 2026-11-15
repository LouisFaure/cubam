"""Velocity counts against velocyto 0.17 (``velocyto run``, Permissive10X) on synthetic data.

``tests/data/velocyto_expected.json`` holds velocyto's counts for the data of
:func:`make_data`. After changing :func:`make_data`, make it again with
``python tests/test_velocity.py <dir>`` (needs pixi, which installs velocyto).
"""

import json
import random
from pathlib import Path

import pytest
from bamwriter import Read, parse_cigar, sort_key, write_bam

from cubam.apps import velocity as velocity_mod
from cubam.apps.velocity import RepeatMask, count_velocity, segments
from cubam.bam import REVERSE, read_bam

EXPECTED = Path(__file__).parent / "data" / "velocyto_expected.json"
REFS = [("chr1", 1_300_000), ("chr2", 100_000), ("chrM", 16_569)]
CELLS = ["AAACCCAAGAAACACT-1", "AAACCCAAGAAACCAT-1", "AAACCCAAGAAACCCA-1", "AAACCCAAGAAACCCG-1"]
OTHER_CELL = "TTTTTTTTTTTTTTTT-1"


def make_genes(rng):
    """Genes as {gene_id: (chrom, strand, [transcript exon lists])}, exons 1-based inclusive."""
    genes = {}

    def gene(gid, chrom, strand, start, n_exons):
        exons, x = [], start
        for _ in range(n_exons):
            ln = rng.randint(60, 300)
            exons.append((x, x + ln - 1))
            x += ln + rng.randint(150, 2500)
        txs = [exons]
        for _ in range(rng.randint(0, 2)):
            sub = [e for e in exons if rng.random() < 0.7] or exons[:1]
            if rng.random() < 0.3:  # an alternative last exon end
                s, e = sub[-1]
                sub = sub[:-1] + [(s, e + rng.randint(20, 200))]
            txs.append(sub)
        genes[gid] = (chrom, strand, txs)

    starts = sorted(rng.sample(range(1_000, 180_000, 500), 12))
    for i, s in enumerate(starts):
        gene(f"G{i:02d}", "chr1", rng.choice("+-"), s, rng.randint(1, 6))
    for i, s in enumerate(sorted(rng.sample(range(1_000, 80_000, 500), 4))):
        gene(f"H{i:02d}", "chr2", rng.choice("+-"), s, rng.randint(1, 5))
    # an intron longer than 1 Mb, with a gene inside it
    genes["LONG"] = ("chr1", "+", [[(200_001, 200_200), (1_250_001, 1_250_300), (1_251_001, 1_251_400)]])
    gene("NESTED", "chr1", "+", 600_000, 3)
    genes["MTGENE"] = ("chrM", "+", [[(1_001, 2_500)]])
    return genes


def write_gtf(path, genes):
    lines = []
    for gid, (chrom, strand, txs) in genes.items():
        for t, exons in enumerate(txs):
            n = len(exons)
            for k, (s, e) in enumerate(exons):
                num = k + 1 if strand == "+" else n - k
                attr = f'gene_id "{gid}"; transcript_id "{gid}.{t}"; gene_name "{gid}n"; exon_number "{num}";'
                lines.append(f"{chrom}\ttest\texon\t{s}\t{e}\t.\t{strand}\t.\t{attr}")
    path.write_text("\n".join(lines) + "\n")


def write_repeats(path, rng, genes):
    """Repeats over the genes, some within 5 bases of each other, some inside others."""
    lines = []
    for chrom, strand, txs in rng.sample(list(genes.values()), 12):
        if chrom == "chrM":
            continue
        lo, hi = min(e[0] for t in txs for e in t), max(e[1] for t in txs for e in t)
        for _ in range(rng.randint(1, 4)):
            s = rng.randint(lo - 300, hi)
            e = s + rng.randint(50, 600)
            st = strand if rng.random() < 0.8 else rng.choice("+-")
            lines.append((chrom, s, e, st))
            if rng.random() < 0.3:  # a neighbour within 5 bases, or a repeat inside this one
                s2 = e + rng.randint(0, 5) if rng.random() < 0.5 else s + 10
                lines.append((chrom, s2, s2 + rng.randint(20, 80), st))
    rng.shuffle(lines)
    path.write_text(
        "".join(
            f'{c}\trmsk\texon\t{s}\t{e}\t.\t{st}\t.\tgene_id "R{i}"; transcript_id "R{i}";\n'
            for i, (c, s, e, st) in enumerate(lines)
        )
    )


def decorate(rng, blocks):
    """A CIGAR from aligned blocks [(start0, len)] (gaps become N), with clips and small indels."""
    ops = []
    for i, (s, ln) in enumerate(blocks):
        if i:
            ops.append(f"{s - (blocks[i - 1][0] + blocks[i - 1][1])}N")
        r = rng.random()
        if ln > 30 and r < 0.12:
            a = rng.randint(10, ln - 10)
            ops += [f"{a}M", f"{rng.randint(1, 3)}{rng.choice('ID')}", f"{ln - a}M"]
        elif ln > 30 and r < 0.17:
            a = rng.randint(10, ln - 10)
            ops += [f"{a}M", f"{rng.randint(4, 9)}D", f"{ln - a}M"]
        elif ln > 30 and r < 0.19:
            a = rng.randint(10, ln - 10)
            ops += [f"{a}M", "2X", f"{ln - a - 2}M"]
        else:
            ops.append(f"{ln}M")
    if rng.random() < 0.25:
        ops.insert(0, f"{rng.randint(1, 30)}S")
    if rng.random() < 0.2:
        ops.append(f"{rng.randint(1, 30)}S")
    return "".join(ops)


def read_blocks(rng, chrom_len, kind, exons, ln):
    """Aligned blocks of one read of a molecule of the given kind."""
    if kind == "splice" and len(exons) > 1:
        k = rng.randrange(len(exons) - 1)
        (_, e1), (s2, _) = exons[k], exons[k + 1]
        a = rng.randint(8, ln - 8)
        jit = rng.choice([0, 0, 0, 2, -3, 9, -12])
        return [(e1 - a + jit, a), (s2 - 1 + jit, ln - a)]
    if kind == "intron" and len(exons) > 1:
        k = rng.randrange(len(exons) - 1)
        lo, hi = exons[k][1], exons[k + 1][0] - 1 - ln
        if hi > lo:
            return [(rng.randint(lo, hi), ln)]
    if kind == "span" and len(exons) > 1:
        k = rng.randrange(len(exons) - 1)
        b = rng.choice([exons[k][1], exons[k + 1][0] - 1])
        return [(b - rng.randint(3, ln - 3), ln)]
    if kind == "around":
        lo = max(0, exons[0][0] - 600)
        return [(rng.randint(lo, min(chrom_len - ln, exons[-1][1] + 600)), ln)]
    s, e = rng.choice(exons)
    return [(rng.randint(s - 1 - ln // 3, max(s - 1 - ln // 3, e - ln + ln // 3)), ln)]


def make_reads(rng, genes):
    refs = {n: i for i, (n, _) in enumerate(REFS)}
    lens = dict(REFS)
    reads, n = [], 0
    for cell in [*CELLS, OTHER_CELL]:
        for _ in range(160):
            gid = rng.choice(list(genes))
            chrom, strand, txs = genes[gid]
            exons = rng.choice(txs)
            kind = rng.choice(["exon", "exon", "splice", "splice", "intron", "intron", "span", "around"])
            umi = "".join(rng.choices("ACGT", k=5))  # few UMIs: (cell, UMI) collisions across genes
            rev = (strand == "-") != (rng.random() < 0.1)
            for _ in range(rng.choice([1, 1, 2, 3])):
                if rng.random() < 0.2:  # reads of a molecule need not agree
                    kind = rng.choice(["exon", "splice", "intron", "span"])
                ln = rng.randint(40, 91)
                blocks = [(max(0, s), ln_) for s, ln_ in read_blocks(rng, lens[chrom], kind, exons, ln)]
                tags = [("CB", "Z", cell)]
                if rng.random() > 0.03:
                    tags.append(("UB", "Z", umi))
                tags.append(("NH", "i", 1 if rng.random() > 0.05 else 2))
                cigar = decorate(rng, blocks)
                reads.append(Read(f"r{n}", REVERSE if rev else 0, refs[chrom], blocks[0][0], 255, cigar, tags=tags))
                n += 1
    return sorted(reads, key=sort_key)


def make_data(d: Path):
    rng = random.Random(7)
    genes = make_genes(rng)
    write_gtf(d / "genes.gtf", genes)
    reads = make_reads(rng, genes)
    write_bam(d / "reads.bam", REFS, reads, block=3000)
    write_repeats(d / "repeats.gtf", rng, genes)
    (d / "barcodes.tsv").write_text("\n".join(CELLS) + "\n")
    return reads


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("vel")
    reads = make_data(d)
    return d, reads


def py_segments(read):
    """velocyto's parse_cigar_tuple (counter.py), 1-based inclusive segments."""
    code = {"M": 0, "I": 1, "D": 2, "N": 3, "S": 4, "H": 5, "P": 6, "=": 7, "X": 8}
    cig = [(code[op], n) for op, n in parse_cigar(read.cigar)]
    segs, holes, skip, pos = [], set(), False, read.pos + 1
    p = pos
    for i, (op, ln) in enumerate(cig):
        if op == 0:
            segs.append((p, p + ln - 1))
            p += ln
        elif op == 3:
            skip = True
            p += ln
        elif op in (1, 2):
            if ln <= 3 and 0 < i < len(cig) - 1 and cig[i + 1][0] == 0 and cig[i - 1][0] == 0:
                holes.add(len(segs) - 1)
            if op == 2:
                p += ln
        elif op == 4:
            p += ln
    for a, b in enumerate(sorted(holes)):
        segs[b - a] = (segs.pop(b - a)[0], segs[b - a][1])
    return segs, skip


def test_segments_match_velocyto_parser(data):
    d, reads = data
    got = {}
    for batch in read_bam(d / "reads.bam", 4000):
        rec, s, e, spliced = segments(batch)
        names = batch.names().to_arrow().to_pylist()
        rec, s, e, spliced = rec.get(), s.get(), e.get(), spliced.get()
        for i, nm in enumerate(names):
            got[nm] = ([], bool(spliced[i]))
        for r, a, b in zip(rec, s, e, strict=True):
            got[names[r]][0].append((int(a), int(b)))
    exp = {r.name: py_segments(r) for r in reads}
    assert got == exp


def counts(res):
    c = res.counts.to_pandas()
    gid = res.genes["gene_id"].to_numpy()
    out = {}
    for r in c.itertuples():
        out.setdefault(res.barcodes[r.cell], {})[gid[r.gene]] = [r.spliced, r.unspliced, r.ambiguous]
    return out


@pytest.mark.parametrize("mask", [False, True])
def test_counts_match_velocyto(data, monkeypatch, mask):
    d, _ = data
    if mask:  # molecules reduced a few cells at a time
        monkeypatch.setattr(velocity_mod, "_BUCKET_FACTOR", 1 << 40)
    m = d / "repeats.gtf" if mask else None
    res = count_velocity(d / "reads.bam", d / "genes.gtf", d / "barcodes.tsv", mask=m, chunk_bytes=5000)
    assert counts(res) == json.loads(EXPECTED.read_text())["mask" if mask else "plain"]
    if mask:
        assert res.stats["masked"] > 0
    s = res.stats
    assert s["molecules"] == s["no_shared_transcript"] + s["multigene"] + s["unclassified"] + sum(
        s[k] for k in ("spliced", "unspliced", "ambiguous")
    )


def test_without_quirks_counts_chrM(data):
    d, _ = data
    q = counts(count_velocity(d / "reads.bam", d / "genes.gtf", d / "barcodes.tsv"))
    nq = counts(count_velocity(d / "reads.bam", d / "genes.gtf", d / "barcodes.tsv", quirks=False))
    assert not any("MTGENE" in g for g in q.values())
    assert any("MTGENE" in g for g in nq.values())


@pytest.mark.parametrize("n_repeats", [1, 3])
def test_repeat_mask_empty_after_quirks(data, tmp_path, n_repeats):
    d, _ = data
    path = tmp_path / "single_strand.gtf"
    path.write_text(
        "".join(
            f'chr1\trmsk\texon\t{10 + i * 200}\t{100 + i * 200}\t.\t+\t.\tgene_id "R{i}";\n' for i in range(n_repeats)
        )
    )
    mask = RepeatMask(path)
    assert mask.intervals.empty
    assert mask.index.n_intervals == 0
    assert len(RepeatMask(path, quirks=False).intervals) == n_repeats
    res = count_velocity(d / "reads.bam", d / "genes.gtf", CELLS, mask=mask)
    assert counts(res) == json.loads(EXPECTED.read_text())["plain"]
    assert res.stats["masked"] == 0


def test_anndata_and_h5ad(data, tmp_path):
    d, _ = data
    res = count_velocity(d / "reads.bam", d / "genes.gtf", d / "barcodes.tsv")
    anndata = pytest.importorskip("anndata")
    import pandas as pd
    import scipy.sparse as sp

    a = res.to_anndata()
    assert a.shape == (len(CELLS), len(res.genes))
    for k in ("spliced", "unspliced", "ambiguous"):
        assert int(a.layers[k].sum()) == res.stats[k]
    path = tmp_path / "v.h5ad"
    res.write_h5ad(path)
    saved = anndata.read_h5ad(path)
    assert saved.obs_names.tolist() == CELLS
    a.strings_to_categoricals()
    pd.testing.assert_frame_equal(saved.var, a.var)
    assert sp.issparse(saved.X)
    assert (saved.X != a.X).nnz == 0
    # Newer AnnData versions also expose X as layers[None].
    layer_names = {k for k in saved.layers if k is not None}
    assert layer_names == {"spliced", "unspliced", "ambiguous"}
    for k in layer_names:
        assert sp.issparse(saved.layers[k])
        assert (saved.layers[k] != a.layers[k]).nnz == 0


if __name__ == "__main__":  # write velocyto's counts for make_data into EXPECTED
    import subprocess
    import sys

    import h5py

    d = Path(sys.argv[1])
    d.mkdir(parents=True, exist_ok=True)
    make_data(d)
    full = {c.split("-")[0]: c for c in CELLS}
    expected = {}
    for key, opt in (("plain", ""), ("mask", "-m repeats.gtf")):
        subprocess.run(
            "pixi exec -s velocyto.py -s samtools velocyto run -b barcodes.tsv "
            f"-o {key} -e s {opt} reads.bam genes.gtf",
            shell=True,
            check=True,
            cwd=d,
        )
        with h5py.File(d / key / "s.loom") as f:
            cells = [c.decode().split(":")[1][:-1] for c in f["col_attrs/CellID"][:]]
            gids = [g.decode() for g in f["row_attrs/Accession"][:]]
            layers = [f[f"layers/{k}"][:] for k in ("spliced", "unspliced", "ambiguous")]
        out = expected[key] = {}
        for j, c in enumerate(cells):
            for i, g in enumerate(gids):
                v = [int(x[i, j]) for x in layers]
                if any(v):
                    out.setdefault(full[c], {})[g] = v
        print(key, sum(map(sum, (v for g in out.values() for v in g.values()))), "molecules")
    EXPECTED.parent.mkdir(exist_ok=True)
    EXPECTED.write_text(json.dumps(expected, indent=1, sort_keys=True) + "\n")
