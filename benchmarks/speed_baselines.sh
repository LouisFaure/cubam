#!/usr/bin/env bash
# CPU tools on the same inputs as speed_cubam.py, one after the other, with $THREADS threads.
#
#   pixi exec -s sinto -s "setuptools<81" -s samtools -s bedtools -s ucsc-bedgraphtobigwig \
#       -s subread -s cellsnp-lite -- bash benchmarks/speed_baselines.sh \
#       OUTS GENES.gtf SNPS.vcf BC_RNA.tsv BC_ATAC.tsv WORKDIR [STEP...]
#
# Times (seconds) go to WORKDIR/baseline_times.tsv. Cells get the same 8 groups
# as in speed_cubam.py (cl0..cl7, in turn).
set -euo pipefail
OUTS=$1 GTF=$2 VCF=$3 BC_RNA=$4 BC_ATAC=$5 W=$6
shift 6
STEPS=${*:-features_gex coverage_gex split_gex sort_gex alleles_atac fragments_atac split_atac alleles_gex}
THREADS=${THREADS:-6}
GEX=$OUTS/gex_possorted_bam.bam ATAC=$OUTS/atac_possorted_bam.bam
mkdir -p "$W/base"
B=$W/base
awk '{print $1"\tcl"(NR-1)%8}' "$BC_RNA" >"$B/groups_rna.tsv"
awk '{print $1"\tcl"(NR-1)%8}' "$BC_ATAC" >"$B/groups_atac.tsv"

timed() {  # name command...
    local name=$1 t0 t1
    shift
    t0=$(date +%s.%N)
    "$@"
    t1=$(date +%s.%N)
    printf '%s\t%.1f\n' "$name" "$(echo "$t1 - $t0" | bc)" | tee -a "$W/baseline_times.tsv"
}

features_gex() {  # bulk counts per gene only: no cells, no UMIs
    featureCounts -T "$THREADS" -a "$GTF" -t exon -g gene_id -s 1 -M -o "$B/featurecounts.txt" "$GEX"
}

coverage_group() {  # one group: reads as cubam keeps them, bedGraph, then bigWig
    local g=$1
    cut -f1 <(awk -v g="$g" '$2==g' "$B/groups_rna.tsv") >"$B/cov/$g.txt"
    samtools view -u -F 0xF04 -q 30 -D "CB:$B/cov/$g.txt" "$GEX" |
        bedtools genomecov -ibam stdin -bg -split >"$B/cov/$g.bg"
    bedGraphToBigWig "$B/cov/$g.bg" "$B/cov/chrom.sizes" "$B/cov/$g.bw"
    rm "$B/cov/$g.bg"
}
export -f coverage_group
coverage_gex() {
    mkdir -p "$B/cov"
    samtools view -H "$GEX" | awk -F'\t' '$1=="@SQ"{sub("SN:","",$2); sub("LN:","",$3); print $2"\t"$3}' >"$B/cov/chrom.sizes"
    export B GEX
    printf 'cl%d\n' 0 1 2 3 4 5 6 7 | xargs -P "$THREADS" -I{} bash -c 'coverage_group {}'
}

split_gex() {
    mkdir -p "$B/split_gex"
    sinto filterbarcodes -b "$GEX" -c "$B/groups_rna.tsv" -p "$THREADS" --outdir "$B/split_gex"
    rm -r "$B/split_gex"
}
split_atac() {
    mkdir -p "$B/split_atac"
    sinto filterbarcodes -b "$ATAC" -c "$B/groups_atac.tsv" -p "$THREADS" --outdir "$B/split_atac"
    rm -r "$B/split_atac"
}

sort_gex() {
    samtools sort -n -@ "$THREADS" -o "$B/sort_gex.bam" "$GEX"
    rm "$B/sort_gex.bam"
}

fragments_atac() {  # unsorted, uncompressed output, as cubam writes it here
    sinto fragments -b "$ATAC" -f "$B/sinto_fragments.tsv" -p "$THREADS"
}

alleles_gex() {
    cellsnp-lite -s "$GEX" -b "$BC_RNA" -R "$VCF" -O "$B/cellsnp_gex" -p "$THREADS" --minCOUNT 2 --UMItag UB
}
alleles_atac() {
    cellsnp-lite -s "$ATAC" -b "$BC_ATAC" -R "$VCF" -O "$B/cellsnp_atac" -p "$THREADS" --minCOUNT 2 --UMItag None
}

for s in $STEPS; do
    timed "$s" "$s"
done
