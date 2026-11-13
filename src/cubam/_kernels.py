"""CUDA kernels of cubam, compiled once per process."""

from __future__ import annotations

import numpy as np

SOURCE = r"""
extern "C" {

__device__ __forceinline__ unsigned int u32(const unsigned char* b, long long o) {
    return (unsigned int)b[o] | ((unsigned int)b[o + 1] << 8) | ((unsigned int)b[o + 2] << 16)
         | ((unsigned int)b[o + 3] << 24);
}
__device__ __forceinline__ unsigned int u16(const unsigned char* b, long long o) {
    return (unsigned int)b[o] | ((unsigned int)b[o + 1] << 8);
}

// Append offset o to out when r is true: one atomic per warp
__device__ __forceinline__ void warp_append(bool r, long long o, long long* out, unsigned long long cap,
                                            unsigned long long* count) {
    unsigned int mask = __ballot_sync(0xffffffffu, r);
    if (mask == 0) return;
    unsigned int lane = threadIdx.x & 31, leader = __ffs(mask) - 1;
    unsigned long long base = 0;
    if (lane == leader) base = atomicAdd(count, (unsigned long long)__popc(mask));
    base = __shfl_sync(0xffffffffu, base, leader);
    if (r) {
        unsigned long long k = base + __popc(mask & ((1u << lane) - 1));
        if (k < cap) out[k] = o;
    }
}

// dst[dst_off[i] : +size[i]] = src[src_off[i] : +size[i]], one warp per span
__global__ void copy_spans(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
                           const long long* __restrict__ src_off, const long long* __restrict__ dst_off,
                           const long long* __restrict__ size, long long n) {
    long long w = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= n) return;
    const unsigned char* s = src + src_off[w];
    unsigned char* d = dst + dst_off[w];
    for (long long i = lane; i < size[w]; i += 32) d[i] = s[i];
}

// ------------------------------------------------------------------ BGZF

// Complete BGZF blocks: the fixed header fields (XLEN 6, one BC subfield) and a block that fits
__global__ void bgzf_candidates(const unsigned char* b, long long n, long long* out,
                                unsigned long long cap, unsigned long long* count) {
    long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    bool r = o + 18 <= n && b[o] == 31 && b[o + 1] == 139 && b[o + 2] == 8 && b[o + 3] == 4
             && b[o + 10] == 6 && b[o + 11] == 0 && b[o + 12] == 66 && b[o + 13] == 67
             && b[o + 14] == 2 && b[o + 15] == 0;
    if (r) {
        long long bs = (long long)u16(b, o + 16) + 1;
        r = bs >= 26 && o + bs <= n;
    }
    warp_append(r, o, out, cap, count);
}

__global__ void bgzf_sizes(const unsigned char* b, const long long* start, long long m,
                           long long* bsize, long long* isize) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    long long s = start[i], bs = (long long)u16(b, s + 16) + 1;
    bsize[i] = bs;
    isize[i] = (long long)u32(b, s + bs - 4);
}

// ------------------------------------------------------------------ BAM records

__device__ int reg2bin(int beg, int end) {
    --end;
    if (beg >> 14 == end >> 14) return ((1 << 15) - 1) / 7 + (beg >> 14);
    if (beg >> 17 == end >> 17) return ((1 << 12) - 1) / 7 + (beg >> 17);
    if (beg >> 20 == end >> 20) return ((1 << 9) - 1) / 7 + (beg >> 20);
    if (beg >> 23 == end >> 23) return ((1 << 6) - 1) / 7 + (beg >> 23);
    if (beg >> 26 == end >> 26) return ((1 << 3) - 1) / 7 + (beg >> 26);
    return 0;
}

__device__ __forceinline__ bool ref_consuming(unsigned int op) {
    return op == 0 || op == 2 || op == 3 || op == 7 || op == 8;  // M D N = X
}

// A complete, structurally valid BAM record starts at byte o
__device__ bool is_record(const unsigned char* b, long long n, long long o, int n_ref) {
    if (o + 36 > n) return false;
    int bs = (int)u32(b, o);
    if (bs < 33 || bs > (1 << 22) || o + 4 + bs > n) return false;
    int ref = (int)u32(b, o + 4), pos = (int)u32(b, o + 8);
    int nref = (int)u32(b, o + 24), npos = (int)u32(b, o + 28);
    if (ref < -1 || ref >= n_ref || pos < -1 || nref < -1 || nref >= n_ref || npos < -1) return false;
    unsigned int lrn = b[o + 12], ncig = u16(b, o + 16), flag = u16(b, o + 18);
    unsigned int lseq = u32(b, o + 20);
    if (lrn < 2 || flag >= 4096 || lseq > (1u << 22)) return false;
    if (32LL + lrn + 4LL * ncig + (lseq + 1) / 2 + lseq > bs) return false;
    long long nm = o + 36;
    if (b[nm + lrn - 1] != 0) return false;
    for (unsigned int i = 0; i + 1 < lrn; i++) {
        unsigned char c = b[nm + i];
        if (c < 33 || c > 126 || c == '@') return false;
    }
    long long cg = nm + lrn, rlen = 0;
    for (unsigned int i = 0; i < ncig; i++) {
        unsigned int v = u32(b, cg + 4LL * i), op = v & 15;
        if (op > 8) return false;
        if (ref_consuming(op)) rlen += v >> 4;
    }
    if (pos >= 0) {
        int end = pos + (int)(rlen > 0 ? rlen : 1);
        if (u16(b, o + 14) != (unsigned int)reg2bin(pos, end)) return false;
    }
    return true;
}

__global__ void record_candidates(const unsigned char* b, long long n, int n_ref, long long* out,
                                  unsigned long long cap, unsigned long long* count) {
    long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    warp_append(o < n && is_record(b, n, o, n_ref), o, out, cap, count);
}

__global__ void record_ends(const unsigned char* b, const long long* off, long long m, long long* nxt) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) nxt[i] = off[i] + 4 + (long long)u32(b, off[i]);
}

// Fixed fields, and from the CIGAR the alignment end and the soft clips at both ends
__global__ void decode_fixed(const unsigned char* b, const long long* off, long long n,
                             int* ref, int* pos, int* end, unsigned char* mapq, unsigned short* flag,
                             int* nref, int* npos, int* tlen, int* l_seq,
                             unsigned short* clip_left, unsigned short* clip_right) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long o = off[i];
    int p = (int)u32(b, o + 8);
    ref[i] = (int)u32(b, o + 4);
    pos[i] = p;
    mapq[i] = b[o + 13];
    flag[i] = (unsigned short)u16(b, o + 18);
    nref[i] = (int)u32(b, o + 24);
    npos[i] = (int)u32(b, o + 28);
    tlen[i] = (int)u32(b, o + 32);
    l_seq[i] = (int)u32(b, o + 20);
    unsigned int lrn = b[o + 12], ncig = u16(b, o + 16);
    long long cg = o + 36 + lrn, rlen = 0;
    unsigned int cl = 0, cr = 0;
    for (unsigned int k = 0; k < ncig; k++) {
        unsigned int v = u32(b, cg + 4LL * k), op = v & 15, len = v >> 4;
        if (ref_consuming(op)) rlen += len;
        if (op == 4) { if (k == 0) cl = len; else cr = len; }
    }
    end[i] = p + (int)rlen;
    clip_left[i] = (unsigned short)(cl > 65535 ? 65535 : cl);
    clip_right[i] = (unsigned short)(cr > 65535 ? 65535 : cr);
}

// FNV-1a 64 of the read name
__global__ void name_hash(const unsigned char* b, const long long* off, long long n, unsigned long long* h) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long o = off[i];
    unsigned int lrn = b[o + 12];
    unsigned long long x = 1469598103934665603ULL;
    for (unsigned int k = 0; k + 1 < lrn; k++) { x ^= b[o + 36 + k]; x *= 1099511628211ULL; }
    h[i] = x;
}

// Locate an auxiliary tag. type 0 = missing, 255 = malformed aux data, otherwise the type
// character. Z/H: value offset and length (without the NUL). Integer types: the value.
__global__ void find_tag(const unsigned char* b, const long long* off, long long n,
                         unsigned char t0, unsigned char t1,
                         unsigned char* type, long long* voff, int* vlen, long long* ival) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long o = off[i];
    long long rec_end = o + 4 + (long long)u32(b, o);
    unsigned int lrn = b[o + 12], ncig = u16(b, o + 16);
    unsigned int lseq = u32(b, o + 20);
    long long a = o + 36 + lrn + 4LL * ncig + (lseq + 1) / 2 + lseq;
    unsigned char ty_out = 0;
    long long vo = 0, iv = 0;
    int vl = 0;
    while (a + 3 <= rec_end) {
        unsigned char ty = b[a + 2];
        long long v = a + 3, nx;
        switch (ty) {
            case 'A': case 'c': case 'C': nx = v + 1; break;
            case 's': case 'S': nx = v + 2; break;
            case 'i': case 'I': case 'f': nx = v + 4; break;
            case 'Z': case 'H': nx = v; while (nx < rec_end && b[nx] != 0) nx++; nx++; break;
            case 'B': {
                if (v + 5 > rec_end) { nx = -1; break; }
                unsigned char sub = b[v];
                long long cnt = (long long)u32(b, v + 1);
                int sz = (sub == 'c' || sub == 'C') ? 1 : (sub == 's' || sub == 'S') ? 2 : 4;
                nx = v + 5 + cnt * sz;
                break;
            }
            default: nx = -1;
        }
        if (nx < 0 || nx > rec_end) { ty_out = 255; break; }
        if (b[a] == t0 && b[a + 1] == t1) {
            ty_out = ty;
            vo = v;
            switch (ty) {
                case 'c': iv = (signed char)b[v]; break;
                case 'C': case 'A': iv = b[v]; break;
                case 's': iv = (short)u16(b, v); break;
                case 'S': iv = u16(b, v); break;
                case 'i': iv = (int)u32(b, v); break;
                case 'I': iv = u32(b, v); break;
                case 'Z': case 'H': vl = (int)(nx - 1 - v); break;
                case 'B': vl = (int)(nx - v); break;
                default: break;
            }
            break;
        }
        a = nx;
    }
    type[i] = ty_out;
    voff[i] = vo;
    vlen[i] = vl;
    ival[i] = iv;
}

// Pack a string value [ACGT]{1,24}(-[0-9]{1,3})? into 64 bits: 2 bits per base,
// the length at bit 48, the suffix + 1 at bit 53 (0 = no suffix). ok = 0 when it does not fit.
__global__ void pack_acgt(const unsigned char* b, const long long* voff, const int* vlen, long long n,
                          unsigned long long* out, unsigned char* ok) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long k = voff[i], e = voff[i] + vlen[i];
    unsigned long long code = 0, suffix = 0;
    unsigned int nb = 0;
    bool good = vlen[i] > 0;
    for (; good && k < e && b[k] != '-'; k++) {
        unsigned char c = b[k];
        unsigned int x = c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : c == 'T' ? 3 : 4;
        if (x == 4 || nb == 24) good = false;
        else code |= (unsigned long long)x << (2 * nb++);
    }
    if (good && k < e) {
        k++;
        unsigned int nd = 0;
        for (; k < e; k++, nd++) {
            if (b[k] < '0' || b[k] > '9' || nd == 3) { good = false; break; }
            suffix = suffix * 10 + (b[k] - '0');
        }
        if (nd == 0) good = false;
        suffix += 1;
    }
    if (nb == 0) good = false;
    out[i] = good ? (code | ((unsigned long long)nb << 48) | (suffix << 53)) : 0;
    ok[i] = good;
}

// Unpack codes of pack_acgt into chars at offsets
__global__ void unpack_acgt(const unsigned long long* code, const long long* dst_off, long long n,
                            unsigned char* chars) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long c = code[i];
    unsigned int nb = (unsigned int)((c >> 48) & 31), suffix = (unsigned int)(c >> 53);
    unsigned char* d = chars + dst_off[i];
    for (unsigned int k = 0; k < nb; k++) d[k] = "ACGT"[(c >> (2 * k)) & 3];
    if (suffix) {
        unsigned int s = suffix - 1, nd = s >= 100 ? 3 : s >= 10 ? 2 : 1;
        d[nb] = '-';
        for (unsigned int k = 0; k < nd; k++) { d[nb + nd - k] = '0' + s % 10; s /= 10; }
    }
}
}
"""

_module = None


def kernel(name: str):
    """A kernel of :data:`SOURCE`."""
    import cupy as cp

    global _module
    if _module is None:
        _module = cp.RawModule(code=SOURCE, options=("-std=c++14",))
    return _module.get_function(name)


def launch(name: str, n: int, *args, threads_per_item: int = 1) -> None:
    """Launch ``name`` over ``n`` items (``threads_per_item`` = 32 for one warp per item)."""
    if n:
        t = n * threads_per_item
        kernel(name)(((t + 255) // 256,), (256,), args)


def copy_spans(src, dst, src_off, dst_off, size) -> None:
    """``dst[dst_off[i]:][:size[i]] = src[src_off[i]:][:size[i]]`` on the GPU."""
    import cupy as cp

    n = int(size.size)
    launch(
        "copy_spans",
        n,
        src,
        dst,
        cp.asarray(src_off, cp.int64),
        cp.asarray(dst_off, cp.int64),
        cp.asarray(size, cp.int64),
        np.int64(n),
        threads_per_item=32,
    )


def append_offsets(name: str, data, cap: int, *args):
    """Run a candidate kernel that appends offsets (``warp_append``); returns them sorted."""
    import cupy as cp

    n = int(data.size)
    out = cp.empty(cap, cp.int64)
    count = cp.zeros(1, cp.uint64)
    launch(name, n, data, np.int64(n), *args, out, np.uint64(cap), count)
    m = int(count[0])
    if m > cap:
        raise ValueError("Too many candidates: the data is corrupt")
    return cp.sort(out[:m])


def chain(cand, nxt):
    """Indices of the candidates on the chain from the first one, or None for all of them.

    ``nxt[i]`` is where the item at ``cand[i]`` ends, and ``cand[0]`` must be
    0. When every candidate ends where the next one starts (the usual case)
    the chain is all of them. Otherwise pointer jumping from the first keeps
    the chain and drops spurious candidates.
    """
    import cupy as cp

    m = int(cand.size)
    first_ok, chained = (int(x) for x in cp.stack([cand[0] == 0, (nxt[:-1] == cand[1:]).all()]).get())
    if not first_ok:
        raise ValueError("No valid block or record where one was expected: the data is corrupt")
    if chained:
        return None
    idx = cp.searchsorted(cand, nxt)
    hit = (idx < m) & (cand[cp.minimum(idx, m - 1)] == nxt)
    jump = cp.concatenate([cp.where(hit, idx, m).astype(cp.int64), cp.asarray([m], cp.int64)])
    mark = cp.zeros(m + 1, cp.bool_)
    mark[0] = True
    span = 1
    while span < m:
        mark[jump[cp.flatnonzero(mark)]] = True
        jump = jump[jump]
        span *= 2
    return cp.flatnonzero(mark[:m])
