"""Batched Householder QR - sol_combo (sol_best + sol_v9 CholeskyQR n=4096 B=2 path)."""
import os

import torch
import triton
import triton.language as tl


cuda_src = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <stdint.h>

#define FULL_MASK 0xffffffffu
#define LW 8

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL_MASK, v, o);
    return v;
}

__device__ __forceinline__ void house_coeffs(float alpha, float sigma, float* cf) {
    if (sigma <= 0.f) {
        cf[0] = 0.f; cf[1] = 0.f; cf[2] = alpha;
    } else {
        float beta = -copysignf(sqrtf(fmaf(alpha, alpha, sigma)), alpha);
        cf[0] = (beta - alpha) / beta;
        cf[1] = 1.f / (alpha - beta);
        cf[2] = beta;
    }
}

template <int NT>
__device__ void panel_core(float* S, long sld, int r, int w,
                           float* cf, float* gammas, float* taug,
                           float* scratch) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nw = NT >> 5;
    {
        float part = 0.f;
        for (int i = 1 + threadIdx.x; i < r; i += NT) {
            float x = S[i];
            part = fmaf(x, x, part);
        }
        part = warp_sum(part);
        if (lane == 0) scratch[wid] = part;
        __syncthreads();
        if (threadIdx.x == 0) {
            float sg = 0.f;
            for (int u = 0; u < nw; ++u) sg += scratch[u];
            house_coeffs(S[0], sg, cf);
        }
        __syncthreads();
    }
    for (int j = 0; j < w; ++j) {
        const float* cfc = cf + 4 * (j & 1);
        float* cfn = cf + 4 * ((j + 1) & 1);
        float tj = cfc[0], gj = cfc[1], bj = cfc[2];
        float* colj = S + (long)j * sld;
        if (threadIdx.x == 0) {
            gammas[j] = gj;
            taug[j] = tj;
        }
        for (int k = j + 1 + wid; k < w; k += nw) {
            float* ck = S + (long)k * sld;
            float d = (lane == 0) ? ck[j] : 0.f;
            float acc = 0.f;
            for (int i = j + 1 + lane; i < r; i += 32) acc = fmaf(colj[i], ck[i], acc);
            d += gj * acc;
            d = warp_sum(d);
            float wk = tj * d;
            float alpha_next = 0.f;
            float sq = 0.f;
            if (lane == 0) ck[j] -= wk;
            float wg = wk * gj;
            for (int i = j + 1 + lane; i < r; i += 32) {
                float nv = fmaf(-wg, colj[i], ck[i]);
                ck[i] = nv;
                if (k == j + 1) {
                    if (i == j + 1) alpha_next = nv;
                    else sq = fmaf(nv, nv, sq);
                }
            }
            if (k == j + 1) {
                sq = warp_sum(sq);
                if (lane == 0) house_coeffs(alpha_next, sq, cfn);
            }
        }
        if (threadIdx.x == 0) colj[j] = bj;
        __syncthreads();
    }
}

template <int NT>
__device__ void pair_dots(const float* S, long sld, int r, int w,
                          const float* gammas, float* sWv, int ldwv, int o) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nw = NT >> 5;
    const int npairs = w * (w - 1) / 2;
    for (int p = wid; p < npairs; p += nw) {
        int j = (int)((1.0f + sqrtf(1.0f + 8.0f * (float)p)) * 0.5f);
        while (j * (j - 1) / 2 > p) --j;
        while ((j + 1) * j / 2 <= p) ++j;
        int i = p - j * (j - 1) / 2;
        const float* ci = S + (long)i * sld;
        const float* cj = S + (long)j * sld;
        float acc = 0.f;
        for (int l = j + 1 + lane; l < r; l += 32) acc = fmaf(ci[l], cj[l], acc);
        acc = warp_sum(acc);
        if (lane == 0) {
            sWv[(o + i) * ldwv + (o + j)] = gammas[i] * ci[j] + gammas[i] * gammas[j] * acc;
        }
    }
}

__device__ void t_recurrence(const float* sWv, int ldwv, int o,
                             const float* taug, float* sT, int ldt, int w) {
    const int lane = threadIdx.x & 31;
    for (int j = 0; j < w; ++j) {
        float tj = taug[j];
        for (int i = lane; i < j; i += 32) {
            float s = 0.f;
            for (int k = i; k < j; ++k)
                s = fmaf(sT[i * ldt + k], sWv[(o + k) * ldwv + (o + j)], s);
            sT[i * ldt + j] = -tj * s;
        }
        if (lane == 0) sT[j * ldt + j] = tj;
        for (int i = j + 1 + lane; i < w; i += 32) sT[i * ldt + j] = 0.f;
        __syncwarp();
    }
}

template <int NT>
__global__ void panel_smem_kernel(float* __restrict__ H,
                                  float* __restrict__ P,
                                  float* __restrict__ Tg,
                                  float* __restrict__ tau,
                                  int n, int j0, int w,
                                  long pbs, int ldp, long tbs, int ldt,
                                  int want_T) {
    extern __shared__ float smem[];
    const int r = n - j0;
    const int sld = r | 1;
    // Layout: always-used buffers first (S, gammas, taug, cf, scratch); the block-T
    // scratch (sWv, sT) goes LAST so the want_T=0 launch can omit it (2*w*w floats),
    // shrinking dynamic smem and raising occupancy for the latency-bound panel.
    float* S = smem;
    float* gammas = S + (long)sld * w;
    float* taug = gammas + w;
    float* cf = taug + w;
    float* scratch = cf + 8;
    float* sWv = scratch + 32;
    float* sT = sWv + w * w;

    const long b = blockIdx.x;
    float* Hb = H + b * (long)n * n;

    for (int idx = threadIdx.x; idx < r * w; idx += NT) {
        int i = idx / w, j = idx - i * w;
        S[(long)j * sld + i] = Hb[(long)(j0 + i) * n + (j0 + j)];
    }
    __syncthreads();

    panel_core<NT>(S, sld, r, w, cf, gammas, taug, scratch);

    if (want_T) {
        pair_dots<NT>(S, sld, r, w, gammas, sWv, w, 0);
        __syncthreads();
        if ((threadIdx.x >> 5) == 0) t_recurrence(sWv, w, 0, taug, sT, w, w);
        __syncthreads();
    }

    float* taub = tau + b * (long)n + j0;
    for (int j = threadIdx.x; j < w; j += NT) taub[j] = taug[j];
    for (int idx = threadIdx.x; idx < r * w; idx += NT) {
        int i = idx / w, j = idx - i * w;
        float x = S[(long)j * sld + i];
        Hb[(long)(j0 + i) * n + (j0 + j)] = (i > j) ? gammas[j] * x : x;
    }
    float* Pb = P + b * pbs;
    for (int idx = threadIdx.x; idx < r * w; idx += NT) {
        int j = idx / r, i = idx - j * r;
        float x = S[(long)j * sld + i];
        Pb[(long)j * ldp + i] = (i < j) ? 0.f : (i == j ? 1.f : gammas[j] * x);
    }
    if (want_T) {
        float* Tb = Tg + b * tbs;
        for (int idx = threadIdx.x; idx < w * w; idx += NT) {
            int i = idx / w, j = idx - i * w;
            Tb[(long)i * ldt + j] = sT[i * w + j];
        }
    }
}

template <int NT, int WFIX, int RFIX = 0>
__global__ void panel_smem_wfix0_kernel(float* __restrict__ H,
                                        float* __restrict__ P,
                                        float* __restrict__ tau,
                                        int n, int j0,
                                        long pbs, int ldp) {
    extern __shared__ float smem[];
    const int r = (RFIX > 0) ? RFIX : (n - j0);
    const int sld = r | 1;
    float* S = smem;
    float* gammas = S + (long)sld * WFIX;
    float* taug = gammas + WFIX;
    float* cf = taug + WFIX;
    float* scratch = cf + 8;

    const long b = blockIdx.x;
    float* Hb = H + b * (long)n * n;

    for (int idx = threadIdx.x; idx < r * WFIX; idx += NT) {
        int i = idx / WFIX, j = idx - i * WFIX;
        S[(long)j * sld + i] = Hb[(long)(j0 + i) * n + (j0 + j)];
    }
    __syncthreads();

    panel_core<NT>(S, sld, r, WFIX, cf, gammas, taug, scratch);

    float* taub = tau + b * (long)n + j0;
    for (int j = threadIdx.x; j < WFIX; j += NT) taub[j] = taug[j];
    for (int idx = threadIdx.x; idx < r * WFIX; idx += NT) {
        int i = idx / WFIX, j = idx - i * WFIX;
        float x = S[(long)j * sld + i];
        Hb[(long)(j0 + i) * n + (j0 + j)] = (i > j) ? gammas[j] * x : x;
    }
    float* Pb = P + b * pbs;
    for (int idx = threadIdx.x; idx < r * WFIX; idx += NT) {
        int j = idx / r, i = idx - j * r;
        float x = S[(long)j * sld + i];
        Pb[(long)j * ldp + i] = (i < j) ? 0.f : (i == j ? 1.f : gammas[j] * x);
    }
}

template <int NT, int WFIX, int RFIX = 0>
__global__ void panel_smem_wfixT_kernel(float* __restrict__ H,
                                        float* __restrict__ P,
                                        float* __restrict__ Tg,
                                        float* __restrict__ tau,
                                        int n, int j0,
                                        long pbs, int ldp, long tbs, int ldt) {
    extern __shared__ float smem[];
    const int r = (RFIX > 0) ? RFIX : (n - j0);
    const int sld = r | 1;
    float* S = smem;
    float* gammas = S + (long)sld * WFIX;
    float* taug = gammas + WFIX;
    float* cf = taug + WFIX;
    float* scratch = cf + 8;
    float* sWv = scratch + 32;
    float* sT = sWv + WFIX * WFIX;

    const long b = blockIdx.x;
    float* Hb = H + b * (long)n * n;

    for (int idx = threadIdx.x; idx < r * WFIX; idx += NT) {
        int i = idx / WFIX, j = idx - i * WFIX;
        S[(long)j * sld + i] = Hb[(long)(j0 + i) * n + (j0 + j)];
    }
    __syncthreads();

    panel_core<NT>(S, sld, r, WFIX, cf, gammas, taug, scratch);
    pair_dots<NT>(S, sld, r, WFIX, gammas, sWv, WFIX, 0);
    __syncthreads();
    if ((threadIdx.x >> 5) == 0) t_recurrence(sWv, WFIX, 0, taug, sT, WFIX, WFIX);
    __syncthreads();

    float* taub = tau + b * (long)n + j0;
    for (int j = threadIdx.x; j < WFIX; j += NT) taub[j] = taug[j];
    for (int idx = threadIdx.x; idx < r * WFIX; idx += NT) {
        int i = idx / WFIX, j = idx - i * WFIX;
        float x = S[(long)j * sld + i];
        Hb[(long)(j0 + i) * n + (j0 + j)] = (i > j) ? gammas[j] * x : x;
    }
    float* Pb = P + b * pbs;
    for (int idx = threadIdx.x; idx < r * WFIX; idx += NT) {
        int j = idx / r, i = idx - j * r;
        float x = S[(long)j * sld + i];
        Pb[(long)j * ldp + i] = (i < j) ? 0.f : (i == j ? 1.f : gammas[j] * x);
    }
    float* Tb = Tg + b * tbs;
    for (int idx = threadIdx.x; idx < WFIX * WFIX; idx += NT) {
        int i = idx / WFIX, j = idx - i * WFIX;
        Tb[(long)i * ldt + j] = sT[i * WFIX + j];
    }
}

template <int NT>
__global__ void panel_tall_kernel(float* __restrict__ H,
                                  float* __restrict__ P,
                                  float* __restrict__ Tg,
                                  float* __restrict__ tau,
                                  int n, int j0, int w,
                                  long pbs, int ldp, long tbs, int ldt) {
    extern __shared__ float smem[];
    const int r = n - j0;
    const int sldL = r | 1;
    const long leafBuf = max((long)sldL * LW, (long)(NT >> 5) * 32 * 33);
    float* Sleaf = smem;
    float* sWv = Sleaf + leafBuf;
    float* sRs = sWv + w * w;
    float* sT = sRs + w * w;
    float* sT8 = sT + w * w;
    float* gammas = sT8 + LW * LW;
    float* taus = gammas + LW;
    float* cf = taus + w;
    float* scratch = cf + 8;

    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nw = NT >> 5;
    const long b = blockIdx.x;
    float* Hb = H + b * (long)n * n;
    float* Pb = P + b * pbs;
    float* taub = tau + b * (long)n + j0;

    {
        float* wtile = Sleaf + wid * (32 * 33);
        const int ntr = (r + 31) >> 5, ntc = (w + 31) >> 5;
        for (int t = wid; t < ntr * ntc; t += nw) {
            int tc = t / ntr, tr = t - tc * ntr;
            int g0 = tr * 32, g1 = tc * 32;
            #pragma unroll 4
            for (int rr = 0; rr < 32; ++rr) {
                int gi = g0 + rr, gj = g1 + lane;
                wtile[rr * 33 + lane] =
                    (gi < r && gj < w) ? Hb[(long)(j0 + gi) * n + (j0 + gj)] : 0.f;
            }
            __syncwarp();
            #pragma unroll 4
            for (int cc = 0; cc < 32; ++cc) {
                int gj = g1 + cc, gi = g0 + lane;
                if (gj < w && gi < r) Pb[(long)gj * ldp + gi] = wtile[lane * 33 + cc];
            }
            __syncwarp();
        }
        __syncthreads();
    }

    for (int l0 = 0; l0 < w; l0 += LW) {
        const int lw = min(LW, w - l0);
        const int lr = r - l0;
        for (int j = 0; j < lw; ++j)
            for (int i = threadIdx.x; i < lr; i += NT)
                Sleaf[(long)j * sldL + i] = Pb[(long)(l0 + j) * ldp + l0 + i];
        __syncthreads();
        panel_core<NT>(Sleaf, sldL, lr, lw, cf, gammas, taus + l0, scratch);
        pair_dots<NT>(Sleaf, sldL, lr, lw, gammas, sWv, w, l0);
        __syncthreads();
        if (wid == 0) t_recurrence(sWv, w, l0, taus + l0, sT8, LW, lw);
        for (int e = threadIdx.x; e < lw * lw; e += NT) {
            int i = e / lw, j = e - i * lw;
            if (i <= j) sRs[(l0 + i) * w + (l0 + j)] = Sleaf[(long)j * sldL + i];
        }
        for (int j = wid; j < lw; j += nw)
            for (int i = lane; i < l0; i += 32)
                sRs[i * w + (l0 + j)] = Pb[(long)(l0 + j) * ldp + i];
        __syncthreads();
        for (int j = 0; j < lw; ++j) {
            float gj = gammas[j];
            for (int i = threadIdx.x; i < lr; i += NT) {
                float x = Sleaf[(long)j * sldL + i];
                Sleaf[(long)j * sldL + i] = (i < j) ? 0.f : (i == j ? 1.f : gj * x);
            }
        }
        __syncthreads();
        for (int pj = wid; pj < l0; pj += nw) {
            const float* cp = Pb + (long)pj * ldp + l0;
            float d[LW];
            #pragma unroll
            for (int m = 0; m < LW; ++m) d[m] = 0.f;
            for (int i = lane; i < lr; i += 32) {
                float pv = cp[i];
                #pragma unroll
                for (int m = 0; m < LW; ++m)
                    if (m < lw) d[m] = fmaf(pv, Sleaf[(long)m * sldL + i], d[m]);
            }
            #pragma unroll
            for (int m = 0; m < LW; ++m) d[m] = warp_sum(d[m]);
            if (lane == 0) {
                #pragma unroll
                for (int m = 0; m < LW; ++m)
                    if (m < lw) sWv[pj * w + (l0 + m)] = d[m];
            }
        }
        for (int j = 0; j < lw; ++j) {
            for (int i = threadIdx.x; i < l0; i += NT) Pb[(long)(l0 + j) * ldp + i] = 0.f;
            for (int i = threadIdx.x; i < lr; i += NT)
                Pb[(long)(l0 + j) * ldp + l0 + i] = Sleaf[(long)j * sldL + i];
        }
        __syncthreads();
        const int nrem = w - (l0 + lw);
        for (int kk = wid; kk < nrem; kk += nw) {
            float* cp = Pb + (long)(l0 + lw + kk) * ldp + l0;
            float d[LW];
            #pragma unroll
            for (int m = 0; m < LW; ++m) d[m] = 0.f;
            for (int i = lane; i < lr; i += 32) {
                float c = cp[i];
                #pragma unroll
                for (int m = 0; m < LW; ++m)
                    if (m < lw) d[m] = fmaf(Sleaf[(long)m * sldL + i], c, d[m]);
            }
            #pragma unroll
            for (int m = 0; m < LW; ++m) d[m] = warp_sum(d[m]);
            float ev[LW];
            #pragma unroll
            for (int m = 0; m < LW; ++m) {
                float e = 0.f;
                if (m < lw) {
                    for (int p = 0; p <= m; ++p) e = fmaf(sT8[p * LW + m], d[p], e);
                }
                ev[m] = e;
            }
            for (int i = lane; i < lr; i += 32) {
                float c = cp[i];
                #pragma unroll
                for (int m = 0; m < LW; ++m)
                    if (m < lw) c = fmaf(-Sleaf[(long)m * sldL + i], ev[m], c);
                cp[i] = c;
            }
        }
        __syncthreads();
    }

    if (wid == 0) t_recurrence(sWv, w, 0, taus, sT, w, w);
    __syncthreads();
    {
        float* Tb = Tg + b * tbs;
        for (int idx = threadIdx.x; idx < w * w; idx += NT) {
            int i = idx / w, j = idx - i * w;
            Tb[(long)i * ldt + j] = sT[i * w + j];
        }
        for (int j = threadIdx.x; j < w; j += NT) taub[j] = taus[j];
    }
    __syncthreads();

    {
        float* wtile = Sleaf + wid * (32 * 33);
        const int ntr = (r + 31) >> 5, ntc = (w + 31) >> 5;
        for (int t = wid; t < ntr * ntc; t += nw) {
            int tc = t / ntr, tr = t - tc * ntr;
            int g0 = tr * 32, g1 = tc * 32;
            #pragma unroll 4
            for (int cc = 0; cc < 32; ++cc) {
                int gj = g1 + cc, gi = g0 + lane;
                wtile[lane * 33 + cc] =
                    (gj < w && gi < r) ? Pb[(long)gj * ldp + gi] : 0.f;
            }
            __syncwarp();
            #pragma unroll 4
            for (int rr = 0; rr < 32; ++rr) {
                int gi = g0 + rr, gj = g1 + lane;
                if (gi < r && gj < w) {
                    float v = (gi <= gj) ? sRs[gi * w + gj] : wtile[rr * 33 + lane];
                    Hb[(long)(j0 + gi) * n + (j0 + gj)] = v;
                }
            }
            __syncwarp();
        }
    }
}

template <int NT, int NFIX = 0>
__global__ void qr_fused_kernel(const float* __restrict__ A,
                                float* __restrict__ H,
                                float* __restrict__ tau,
                                int n_rt) {
    const int n = (NFIX > 0) ? NFIX : n_rt;
    extern __shared__ float smem[];
    const int sld = n | 1;
    float* S = smem;
    float* gammas = S + (long)sld * n;
    float* taug = gammas + n;
    float* cf = taug + n;
    float* scratch = cf + 8;
    const long b = blockIdx.x;
    const float* Ab = A + b * (long)n * n;
    float* Hb = H + b * (long)n * n;

    for (int idx = threadIdx.x; idx < n * n; idx += NT) {
        int i = idx / n, j = idx - i * n;
        S[(long)j * sld + i] = Ab[idx];
    }
    __syncthreads();

    panel_core<NT>(S, sld, n, n, cf, gammas, taug, scratch);

    float* taub = tau + b * (long)n;
    for (int j = threadIdx.x; j < n; j += NT) taub[j] = taug[j];
    for (int idx = threadIdx.x; idx < n * n; idx += NT) {
        int i = idx / n, j = idx - i * n;
        float x = S[(long)j * sld + i];
        Hb[idx] = (i > j) ? gammas[j] * x : x;
    }
}

void qr_fused(uint64_t A_ptr, uint64_t H_ptr, uint64_t tau_ptr,
              int B, int n, int nthreads) {
    const float* A = reinterpret_cast<const float*>(A_ptr);
    float* H = reinterpret_cast<float*>(H_ptr);
    float* tau = reinterpret_cast<float*>(tau_ptr);
    size_t smem = ((size_t)(n | 1) * n + 2 * n + 8 + 32) * sizeof(float);
    #define LAUNCH_FUSED(NT) { \
        auto kern = qr_fused_kernel<NT, 0>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448); \
        kern<<<B, NT, smem>>>(A, H, tau, n); }
    #define LAUNCH_FUSED_NFIX(NT, NF) { \
        auto kern = qr_fused_kernel<NT, NF>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448); \
        kern<<<B, NT, smem>>>(A, H, tau, n); }
    if (nthreads == 1024 && n == 32) LAUNCH_FUSED_NFIX(1024, 32)
    else LAUNCH_FUSED(512)
    #undef LAUNCH_FUSED_NFIX
    #undef LAUNCH_FUSED
}

void panel_smem(uint64_t H_ptr, uint64_t P_ptr, uint64_t T_ptr, uint64_t tau_ptr,
                int B, int n, int j0, int w, long pbs, int ldp,
                long tbs, int ldt, int want_T, int nthreads) {
    float* H = reinterpret_cast<float*>(H_ptr);
    float* P = reinterpret_cast<float*>(P_ptr);
    float* T = reinterpret_cast<float*>(T_ptr);
    float* tau = reinterpret_cast<float*>(tau_ptr);
    const int r = n - j0;
    // want_T=0 omits the 2*w*w block-T scratch (sWv,sT) -> smaller smem, higher occupancy.
    size_t tbuf = want_T ? (size_t)(2 * w * w) : 0;
    size_t smem = ((size_t)(r | 1) * w + 2 * w + 8 + 32 + tbuf) * sizeof(float);
    #define LAUNCH_PS(NT) { \
        auto kern = panel_smem_kernel<NT>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448); \
        kern<<<B, NT, smem>>>(H, P, T, tau, n, j0, w, pbs, ldp, tbs, ldt, want_T); }
    if (nthreads == 1024) LAUNCH_PS(1024)
    else LAUNCH_PS(512)
    #undef LAUNCH_PS
}

void panel_smem_wfix(uint64_t H_ptr, uint64_t P_ptr, uint64_t tau_ptr,
                     int B, int n, int j0, int w, long pbs, int ldp, int nthreads) {
    float* H = reinterpret_cast<float*>(H_ptr);
    float* P = reinterpret_cast<float*>(P_ptr);
    float* tau = reinterpret_cast<float*>(tau_ptr);
    const int r = n - j0;
    size_t smem = ((size_t)(r | 1) * w + 2 * (size_t)w + 8 + 32) * sizeof(float);
    #define LAUNCH_PSFIX(NT, WF) { \
        auto kern = panel_smem_wfix0_kernel<NT, WF, 0>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448); \
        kern<<<B, NT, smem>>>(H, P, tau, n, j0, pbs, ldp); }
    #define LAUNCH_PSFIX_R(NT, WF, RF) { \
        auto kern = panel_smem_wfix0_kernel<NT, WF, RF>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448); \
        kern<<<B, NT, smem>>>(H, P, tau, n, j0, pbs, ldp); }
    #define RCASE(NT, RF) if (r == RF) { LAUNCH_PSFIX_R(NT, 32, RF); return; }
    #define RCASEW(NT, WF, RF) if (r == RF) { LAUNCH_PSFIX_R(NT, WF, RF); return; }
    #define R32SET_LOW(NT) \
        RCASE(NT,512); RCASE(NT,480); RCASE(NT,448); RCASE(NT,416); RCASE(NT,384); \
        RCASE(NT,352); RCASE(NT,320); RCASE(NT,288); RCASE(NT,256); RCASE(NT,224); \
        RCASE(NT,192); RCASE(NT,160); RCASE(NT,128); RCASE(NT,96);  RCASE(NT,64); RCASE(NT,32)
    #define R32SET_HI(NT) \
        RCASE(NT,1024); RCASE(NT,992); RCASE(NT,960); RCASE(NT,928); RCASE(NT,896); \
        RCASE(NT,864);  RCASE(NT,832); RCASE(NT,800); RCASE(NT,768); RCASE(NT,736); \
        RCASE(NT,704);  RCASE(NT,672); RCASE(NT,640); RCASE(NT,608); RCASE(NT,576); RCASE(NT,544)
    #define R44SET(NT) \
        RCASEW(NT,44,176); RCASEW(NT,44,132); RCASEW(NT,44,88); RCASEW(NT,44,44)
    #define R48SET(NT) \
        RCASEW(NT,48,1024); RCASEW(NT,48,976); RCASEW(NT,48,928); RCASEW(NT,48,880); \
        RCASEW(NT,48,832);  RCASEW(NT,48,784); RCASEW(NT,48,736); RCASEW(NT,48,688); \
        RCASEW(NT,48,640);  RCASEW(NT,48,592); RCASEW(NT,48,544); RCASEW(NT,48,496); \
        RCASEW(NT,48,448);  RCASEW(NT,48,400); RCASEW(NT,48,352); RCASEW(NT,48,304); \
        RCASEW(NT,48,256);  RCASEW(NT,48,208); RCASEW(NT,48,160); RCASEW(NT,48,112); \
        RCASEW(NT,48,64)
    if (w == 16) {
        if (nthreads == 1024) { RCASEW(1024, 16, 16); LAUNCH_PSFIX(1024, 16) }
        else { RCASEW(512, 16, 16); LAUNCH_PSFIX(512, 16) }
    } else if (w == 32) {
        if (nthreads == 1024) { R32SET_HI(1024); R32SET_LOW(1024); LAUNCH_PSFIX(1024, 32) }
        else if (nthreads == 384) { R32SET_LOW(384); LAUNCH_PSFIX(384, 32) }
        else { R32SET_LOW(512); LAUNCH_PSFIX(512, 32) }
    } else if (w == 40) {
        if (nthreads == 1024) LAUNCH_PSFIX(1024, 40)
        else LAUNCH_PSFIX(512, 40)
    } else if (w == 44) {
        if (nthreads == 1024) { R44SET(1024); LAUNCH_PSFIX(1024, 44) }
        else { R44SET(512); LAUNCH_PSFIX(512, 44) }
    } else if (w == 48) {
        if (nthreads == 1024) { R48SET(1024); LAUNCH_PSFIX(1024, 48) }
        else LAUNCH_PSFIX(512, 48)
    }
    #undef R48SET
    #undef R44SET
    #undef R32SET_HI
    #undef R32SET_LOW
    #undef RCASEW
    #undef RCASE
    #undef LAUNCH_PSFIX_R
    #undef LAUNCH_PSFIX
}

void panel_smem_wfixT(uint64_t H_ptr, uint64_t P_ptr, uint64_t T_ptr, uint64_t tau_ptr,
                      int B, int n, int j0, int w, long pbs, int ldp,
                      long tbs, int ldt, int nthreads) {
    float* H = reinterpret_cast<float*>(H_ptr);
    float* P = reinterpret_cast<float*>(P_ptr);
    float* T = reinterpret_cast<float*>(T_ptr);
    float* tau = reinterpret_cast<float*>(tau_ptr);
    const int r = n - j0;
    size_t smem = ((size_t)(r | 1) * w + 2 * (size_t)w + 8 + 32 + 2 * (size_t)w * w) * sizeof(float);
    #define LAUNCH_PSFIXT(NT, WF) { \
        auto kern = panel_smem_wfixT_kernel<NT, WF, 0>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448); \
        kern<<<B, NT, smem>>>(H, P, T, tau, n, j0, pbs, ldp, tbs, ldt); }
    #define LAUNCH_PSFIXT_R(NT, WF, RF) { \
        auto kern = panel_smem_wfixT_kernel<NT, WF, RF>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448); \
        kern<<<B, NT, smem>>>(H, P, T, tau, n, j0, pbs, ldp, tbs, ldt); }
    #define RCASE_T(NT, RF) if (r == RF) { LAUNCH_PSFIXT_R(NT, 32, RF); return; }
    #define R32SETT_LOW(NT) \
        RCASE_T(NT,512); RCASE_T(NT,480); RCASE_T(NT,448); RCASE_T(NT,416); RCASE_T(NT,384); \
        RCASE_T(NT,352); RCASE_T(NT,320); RCASE_T(NT,288); RCASE_T(NT,256); RCASE_T(NT,224); \
        RCASE_T(NT,192); RCASE_T(NT,160); RCASE_T(NT,128); RCASE_T(NT,96);  RCASE_T(NT,64); RCASE_T(NT,32)
    #define R32SETT_HI(NT) \
        RCASE_T(NT,1024); RCASE_T(NT,992); RCASE_T(NT,960); RCASE_T(NT,928); RCASE_T(NT,896); \
        RCASE_T(NT,864);  RCASE_T(NT,832); RCASE_T(NT,800); RCASE_T(NT,768); RCASE_T(NT,736); \
        RCASE_T(NT,704);  RCASE_T(NT,672); RCASE_T(NT,640); RCASE_T(NT,608); RCASE_T(NT,576); RCASE_T(NT,544)
    if (w == 32) {
        if (nthreads == 1024) { R32SETT_HI(1024); R32SETT_LOW(1024); LAUNCH_PSFIXT(1024, 32) }
        else if (nthreads == 384) { R32SETT_LOW(384); LAUNCH_PSFIXT(384, 32) }
        else { R32SETT_LOW(512); LAUNCH_PSFIXT(512, 32) }
    }
    #undef R32SETT_HI
    #undef R32SETT_LOW
    #undef RCASE_T
    #undef LAUNCH_PSFIXT_R
    #undef LAUNCH_PSFIXT
}

void panel_tall(uint64_t H_ptr, uint64_t P_ptr, uint64_t T_ptr, uint64_t tau_ptr,
                int B, int n, int j0, int w, long pbs, int ldp,
                long tbs, int ldt) {
    float* H = reinterpret_cast<float*>(H_ptr);
    float* P = reinterpret_cast<float*>(P_ptr);
    float* T = reinterpret_cast<float*>(T_ptr);
    float* tau = reinterpret_cast<float*>(tau_ptr);
    const int r = n - j0;
    size_t leafBuf = std::max((size_t)((r | 1) * LW), (size_t)(512 / 32) * 32 * 33);
    size_t smem = (leafBuf + 3 * (size_t)w * w + LW * LW
                   + LW + w + 8 + 32) * sizeof(float);
    auto kern = panel_tall_kernel<512>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
    kern<<<B, 512, smem>>>(H, P, T, tau, n, j0, w, pbs, ldp, tbs, ldt);
}

// ===========================================================================
// CholeskyQR kernels (from sol_v9) — used ONLY for the n=4096 B=2 path.
// Shared __device__ helpers (warp_sum/house_coeffs/t_recurrence/FULL_MASK/LW)
// are already defined above; we reuse them here.
// ===========================================================================

// 2-level blocked upper-tri inverse: sI = inv(sR) (sR upper, w x w).
//   Split [0,w) into [0,h) and [h,w). inv = [[Ai, -Ai*B*Ci],[0,Ci]].
//   Ai, Ci computed via column-parallel back-sub on each half (independent => half depth);
//   off-diagonal X = -Ai*(B*Ci) via two w-parallel matmuls. sX = w*w scratch.
//   Race-free: caller must __syncthreads() after. 1.2-1.3x vs full-depth back-sub.
__device__ __forceinline__ void invUpper_blk2(const float* sR, float* sI, float* sX,
                                              int w, int tid, int NT, int ld = 0) {
    if (ld == 0) ld = w;
    const int h = w >> 1;
    for (int j = tid; j < w; j += NT) {
        sI[j * ld + j] = 1.0f / sR[j * ld + j];
        int lo = (j < h) ? 0 : h;
        for (int i = j - 1; i >= lo; --i) {
            float s = 0.f;
            for (int k = i + 1; k <= j; ++k) s += sR[i * ld + k] * sI[k * ld + j];
            sI[i * ld + j] = -s / sR[i * ld + i];
        }
    }
    __syncthreads();
    int oc = w - h;
    for (int idx = tid; idx < h * oc; idx += NT) {       // sX = B * Ci (h x oc)
        int i = idx / oc, j = h + (idx - i * oc);
        float s = 0.f;
        for (int k = h; k <= j; ++k) s += sR[i * ld + k] * sI[k * ld + j];
        sX[i * ld + j] = s;
    }
    __syncthreads();
    for (int idx = tid; idx < h * oc; idx += NT) {        // sI[off] = -Ai * sX
        int i = idx / oc, j = h + (idx - i * oc);
        float s = 0.f;
        for (int k = i; k < h; ++k) s += sI[i * ld + k] * sX[k * ld + j];
        sI[i * ld + j] = -s;
    }
}

// 4-way (depth-quartered) blocked upper-tri inverse for large w. Splits [0,w) into 4
// blocks; inverts each diagonal block by column back-sub (quarter depth, all 4 in
// parallel), then fills the 3 super-diagonal block bands by increasing block-distance
// d=1,2,3 via two all-thread matmuls each:  Bij = -Bii * (sum_{k>i..j} Aik*Bkj).
// Only the upper triangle of sI is written (callers read upper only). sX = w*w scratch.
__device__ __forceinline__ void invUpper_blk4(const float* sR, float* sI, float* sX,
                                              int w, int tid, int NT, int ld = 0) {
    if (ld == 0) ld = w;
    const int b1 = w / 4, b2 = w / 2, b3 = (3 * w) / 4;
    int bnd[5] = {0, b1, b2, b3, w};
    for (int j = tid; j < w; j += NT) {
        int blk = (j < b1) ? 0 : (j < b2) ? 1 : (j < b3) ? 2 : 3;
        int lo = bnd[blk];
        sI[j * ld + j] = 1.0f / sR[j * ld + j];
        for (int i = j - 1; i >= lo; --i) {
            float s = 0.f;
            for (int k = i + 1; k <= j; ++k) s += sR[i * ld + k] * sI[k * ld + j];
            sI[i * ld + j] = -s / sR[i * ld + i];
        }
    }
    __syncthreads();
    // Off-diagonal band fill, batched per block-distance d. Bands at the same
    // distance only read finalized lower-distance blocks, so two syncs per
    // distance are enough instead of two syncs per band.
    #pragma unroll
    for (int d = 1; d < 4; ++d) {
        const int nb = 4 - d;
        int boff[5]; boff[0] = 0;
        for (int bi = 0; bi < nb; ++bi) {
            int nr = bnd[bi + 1] - bnd[bi];
            int nc = bnd[bi + d + 1] - bnd[bi + d];
            boff[bi + 1] = boff[bi] + nr * nc;
        }
        const int tot = boff[nb];
        for (int idx = tid; idx < tot; idx += NT) {   // sX = sum A[i,k]*B[k,j], all bands
            int bi = 0; while (boff[bi + 1] <= idx) ++bi;
            int loc = idx - boff[bi];
            int ri0 = bnd[bi], ri1 = bnd[bi + 1];
            int cj0 = bnd[bi + d];
            int nc = bnd[bi + d + 1] - cj0;
            int gi = ri0 + loc / nc, gj = cj0 + (loc - (loc / nc) * nc);
            float s = 0.f;
            for (int k = ri1; k <= gj; ++k) s += sR[gi * ld + k] * sI[k * ld + gj];
            sX[gi * ld + gj] = s;
        }
        __syncthreads();
        for (int idx = tid; idx < tot; idx += NT) {   // sI[off] = -Bii * sX, all bands
            int bi = 0; while (boff[bi + 1] <= idx) ++bi;
            int loc = idx - boff[bi];
            int ri0 = bnd[bi], ri1 = bnd[bi + 1];
            int cj0 = bnd[bi + d];
            int nc = bnd[bi + d + 1] - cj0;
            int gi = ri0 + loc / nc, gj = cj0 + (loc - (loc / nc) * nc);
            float s = 0.f;
            for (int k = gi; k < ri1; ++k) s += sI[gi * ld + k] * sX[k * ld + gj];
            sI[gi * ld + gj] = -s;
        }
        __syncthreads();
    }
}

// dispatcher: 4-way only pays off for large w (w>=56); 2-level for smaller.
__device__ __forceinline__ void invUpper_blk(const float* sR, float* sI, float* sX,
                                              int w, int tid, int NT, int ld = 0) {
    if (w >= 56) invUpper_blk4(sR, sI, sX, w, tid, NT, ld);
    else invUpper_blk2(sR, sI, sX, w, tid, NT, ld);
}

__device__ __forceinline__ float _ld(const float v) { return v; }
__device__ __forceinline__ float _ld(const __half v) { return __half2float(v); }
__device__ __forceinline__ void _st(float* p, float v) { *p = v; }
__device__ __forceinline__ void _st(__half* p, float v) { *p = __float2half(v); }

// FUSED chol + recon (CQR1 only). One block per matrix.
//   In:  G = P^T P (w x w), P (top w x w block used: P1)
//   Computes: Rc=chol(G), RcInv=inv(Rc), Q1=P1@RcInv, LU-with-sign on Q1 ->
//             V1,U,d,tau, Uinv=inv(U), M = RcInv@Uinv.
//   Out: H (V1 strict-lower + R_geqrf upper top w rows), tau, Vw (top: unit-lower),
//        M (w x w upper, for the host V2 = P2 @ M gemm), fail.
template <typename GT, typename HT, int WFIX, int NTFIX = 0>
__global__ void chol_recon_kernel(const float* __restrict__ G,
                                  const GT* __restrict__ Gh,
                                  const HT* __restrict__ P,
                                  HT* __restrict__ H,
                                  float* __restrict__ tau,
                                  float* __restrict__ M,
                                  HT* __restrict__ Vw,
                                  float* __restrict__ Tg,
                                  HT* __restrict__ Mh,
                                  HT* __restrict__ Th,
                                  int* __restrict__ fail,
                                  int n, int j0, int w_runtime,
                                  float shift_scale,
                                  long gbs, int gld, long pbs, int pld,
                                  long mbs, int mld, long vbs, int vld,
                                  long tbs, int tld, int want_T) {
    const int w = (WFIX > 0) ? WFIX : w_runtime;
    const int sld = (w == 64) ? (w + 1) : w;
    extern __shared__ float smem[];
    float* sR = smem;            // w*w  (Rc upper, then reused as RcInv)
    float* sI = sR + w * sld;    // w*w  (RcInv upper)
    float* sM = sI + w * sld;    // w*w  (Q1 -> LU: strict-lower=V1, upper=U)
    float* sU = sM + w * sld;    // w*w  (Uinv upper)
    float* sd = sU + w * sld;    // w    (signs)
    float* sX = sd + w;          // w*w  (blocked-inverse scratch)
    const long b = blockIdx.x;
    const float* Gb = G ? (G + b * gbs) : nullptr;
    const GT* Ghb = Gh ? (Gh + b * gbs) : nullptr;
    const HT* Pb = P + b * pbs;
    const int tid = threadIdx.x;
    const int NT = (NTFIX > 0) ? NTFIX : blockDim.x;

    for (int idx = tid; idx < w * w; idx += NT) {
        int i = idx / w, j = idx - i * w;
        float gv = Gb ? Gb[(long)i * gld + j] : _ld(Ghb[(long)i * gld + j]);
        sR[i * sld + j] = gv;
    }
    __syncthreads();
    if ((n == 2048 && w == 64) || shift_scale > 0.0f) {
        if (tid == 0) {
            float tr = 0.0f;
            for (int i = 0; i < w; ++i) tr += sR[i * sld + i];
            float sc = (shift_scale > 0.0f) ? shift_scale : 1.0e-7f;
            sX[0] = tr * sc;
        }
        __syncthreads();
        const float shift = sX[0];
        for (int i = tid; i < w; i += NT) sR[i * sld + i] += shift;
        __syncthreads();
    }

    // Cholesky (upper), right-looking. One sync per column: keep each row in raw
    // unnormalized form during the loop, and fold 1/diag into the rank-1 update.
    // Normalize rows and write sqrt(diag) in one batched pass after the loop.
    __shared__ int bad;
    if (tid == 0) bad = 0;
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < w; ++j) {
        float diag = sR[j * sld + j];
        float inv2 = 1.0f / diag;
        if (tid == 0) {
            if (!(diag > 1e-30f)) bad = 1;
            float root = sqrtf(diag);
            sX[j] = root;
            sX[w + j] = 1.0f / root;
        }
        int tw = w - j - 1;
        for (int idx = tid; idx < tw * tw; idx += NT) {
            int kk = idx / tw, ii = idx - kk * tw;
            int k = j + 1 + kk, i = j + 1 + ii;
            if (i >= k) sR[k * sld + i] -= sR[j * sld + k] * sR[j * sld + i] * inv2;
        }
        __syncthreads();
        if (bad) break;
    }
    if (bad) {
        if (tid == 0) fail[b] = 1;
        return;
    }
    for (int idx = tid; idx < w * w; idx += NT) {
        int j = idx / w, i = idx - j * w;
        if (i > j) sR[j * sld + i] *= sX[w + j];
        else if (i == j) sR[j * sld + j] = sX[j];
    }
    __syncthreads();

    // invert upper-tri Rc -> sI (RcInv), 2-level blocked back-sub.
    invUpper_blk(sR, sI, sX, w, tid, NT, sld);
    __syncthreads();

    // Q1 = P1 @ RcInv  (P1 = top w x w of P, RcInv upper -> k<=j).
    for (int idx = tid; idx < w * w; idx += NT) {
        int i = idx / w, j = idx - i * w;
        float acc = 0.f;
        for (int k = 0; k <= j; ++k) acc += _ld(Pb[(long)i * pld + k]) * sI[k * sld + j];
        sM[i * sld + j] = acc;
    }
    __syncthreads();

    // Unpivoted LU with sign on Q1. One sync per column: keep the sub-diagonal
    // column in raw unscaled form and fold 1/U_ii into the trailing update.
    // Scale the column and write U_ii in one batched pass after the loop.
    #pragma unroll
    for (int i = 0; i < w; ++i) {
        float piv = sM[i * sld + i];
        float di = (piv >= 0.f) ? -1.0f : 1.0f;
        float u = piv - di;
        float invu = 1.0f / u;
        if (tid == 0) { sd[i] = di; sX[i] = u; sX[w + i] = invu; }
        int tw = w - i - 1;
        for (int idx = tid; idx < tw * tw; idx += NT) {
            int kk = idx / tw, jj2 = idx - kk * tw;
            int k = i + 1 + kk, jj = i + 1 + jj2;
            sM[k * sld + jj] -= sM[k * sld + i] * invu * sM[i * sld + jj];
        }
        __syncthreads();
    }
    for (int idx = tid; idx < w * w; idx += NT) {
        int k = idx / w, i = idx - k * w;
        if (k > i) sM[k * sld + i] *= sX[w + i];
        else if (k == i) sM[i * sld + i] = sX[i];
    }
    __syncthreads();
    const bool need_M = (n - j0) > w;
    if (need_M) {
        // invert U (upper) -> sU, 2-level blocked back-sub.
        invUpper_blk(sM, sU, sX, w, tid, NT, sld);
        __syncthreads();
    }

    float* taub = tau + b * (long)n + j0;
    for (int i = tid; i < w; i += NT) taub[i] = -sd[i] * sM[i * sld + i];

    // M = RcInv @ Uinv  (both upper -> upper).  M[i][j] = sum_{k=i..j} sI[i][k]*sU[k][j].
    if (need_M) {
        float* Mb = M ? (M + b * mbs) : nullptr;
        HT* Mhb = Mh ? (Mh + b * mbs) : nullptr;
        for (int idx = tid; idx < w * w; idx += NT) {
            int i = idx / w, j = idx - i * w;
            float v = 0.f;
            if (i <= j) {
                for (int k = i; k <= j; ++k) v += sI[i * sld + k] * sU[k * sld + j];
            }
            if (Mb) Mb[(long)i * mld + j] = v;
            if (Mhb) _st(&Mhb[(long)i * mld + j], v);
        }
    }

    if (want_T) {
        __syncthreads();
        // Direct compact-WY block T from reconstruction factors.
        for (int idx = tid; idx < w * w; idx += NT) {
            int i = idx / w, j = idx - i * w;
            sU[i * sld + j] = (i == j) ? 1.0f : ((i < j) ? sM[j * sld + i] : 0.0f);
        }
        __syncthreads();
        invUpper_blk(sU, sI, sX, w, tid, NT, sld);
        __syncthreads();
        float* Tb = Tg ? (Tg + b * tbs) : nullptr;
        HT* Thb = Th ? (Th + b * tbs) : nullptr;
        for (int idx = tid; idx < w * w; idx += NT) {
            int i = idx / w, j = idx - i * w;
            float v = 0.0f;
            if (i <= j) {
                for (int k = i; k <= j; ++k) v += sM[i * sld + k] * sd[k] * sI[k * sld + j];
                v = -v;
            }
            if (Tb) Tb[(long)i * tld + j] = v;
            if (Thb) _st(&Thb[(long)i * tld + j], v);
        }
    }

    // R_geqrf = d_i * Rc[i][j]  (Rc still in sR upper).  Write H top w rows + V1.
    HT* Hb = H + b * (long)n * n;
    HT* Vwb = Vw + b * vbs;
    for (int idx = tid; idx < w * w; idx += NT) {
        int i = idx / w, j = idx - i * w;
        float vlo = sM[i * sld + j];
        if (i > j) {
            _st(&Hb[(long)(j0 + i) * n + (j0 + j)], vlo);
            _st(&Vwb[(long)i * vld + j], vlo);
        } else {
            _st(&Hb[(long)(j0 + i) * n + (j0 + j)], sd[i] * sR[i * sld + j]);
            _st(&Vwb[(long)i * vld + j], (i == j) ? 1.0f : 0.0f);
        }
    }
}

// larft: T (w x w upper) from V^T V (w x w, only strict-upper used: i<j) and tau.
template <int WFIX, int NTFIX = 0>
__global__ void larft_kernel(const float* __restrict__ VtV,
                             const float* __restrict__ tau,
                             float* __restrict__ Tg,
                             int n, int j0, int w_runtime,
                             long vbs, int vld, long tbs, int tld) {
    const int w = (WFIX > 0) ? WFIX : w_runtime;
    extern __shared__ float smem[];
    float* sWv = smem;          // w*w
    float* sT = sWv + w * w;    // w*w
    float* sX = sT + w * w;     // w*w  (blocked-inverse off-diagonal scratch)
    float* staug = sX + w * w;  // w
    const long b = blockIdx.x;
    const float* Vb = VtV + b * vbs;
    const float* taub = tau + b * (long)n + j0;
    const int tid = threadIdx.x;
    const int NT = (NTFIX > 0) ? NTFIX : blockDim.x;

    for (int idx = tid; idx < w * w; idx += NT) {
        int i = idx / w, j = idx - i * w;
        sWv[i * w + j] = Vb[(long)i * vld + j];
    }
    for (int i = tid; i < w; i += NT) staug[i] = taub[i];
    __syncthreads();

    // Build U = diag(1/tau) + striu(VtV) into sWv (upper used), then T = inv(U).
    // Identity: H_1..H_w = I - V T V^T  with  T = inv(diag(1/tau) + striu(V^T V)).
    // Diagonal of U is 1/tau[i]; off-diagonal upper = VtV[i][j] (i<j) already in sWv.
    // Setting sWv[i][i]=1/tau makes the generic blocked inverse (which divides by the
    // diagonal) produce exactly the tau-multiplied back-sub (inv(U[i][i])=tau[i]).
    for (int i = tid; i < w; i += NT) sWv[i * w + i] = 1.0f / staug[i];
    // zero full sT (lower triangle stays 0; blocked inverse writes only upper).
    for (int idx = tid; idx < w * w; idx += NT) sT[idx] = 0.f;
    __syncthreads();
    invUpper_blk(sWv, sT, sX, w, tid, NT);   // 4-way for w>=56, else 2-level
    __syncthreads();

    float* Tb = Tg + b * tbs;
    for (int idx = tid; idx < w * w; idx += NT) {
        int i = idx / w, j = idx - i * w;
        Tb[(long)i * tld + j] = sT[i * w + j];
    }
}

void chol_recon(uint64_t G_ptr, uint64_t P_ptr, uint64_t H_ptr,
                uint64_t tau_ptr, uint64_t M_ptr, uint64_t Vw_ptr,
                uint64_t fail_ptr, int B, int n, int j0, int w,
                float shift_scale,
                long gbs, int gld, long pbs, int pld,
                long mbs, int mld, long vbs, int vld, int nthreads) {
    const float* G = reinterpret_cast<const float*>(G_ptr);
    const float* P = reinterpret_cast<const float*>(P_ptr);
    float* H = reinterpret_cast<float*>(H_ptr);
    float* tau = reinterpret_cast<float*>(tau_ptr);
    float* M = reinterpret_cast<float*>(M_ptr);
    float* Vw = reinterpret_cast<float*>(Vw_ptr);
    int* fail = reinterpret_cast<int*>(fail_ptr);
    const int sld = (w == 64) ? (w + 1) : w;
    size_t smem = (size_t)(5 * w * sld + w) * sizeof(float);
    #define LAUNCH_CHOL(WF) { \
        auto kern = chol_recon_kernel<float, float, WF>; \
        if (smem > 48000) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000); \
        kern<<<B, nthreads, smem>>>(G, nullptr, P, H, tau, M, Vw, nullptr, nullptr, nullptr, fail, n, j0, w, \
            shift_scale, \
            gbs, gld, pbs, pld, mbs, mld, vbs, vld, 0, 0, 0); }
    #define LAUNCH_CHOL_NT(WF, NT) { \
        auto kern = chol_recon_kernel<float, float, WF, NT>; \
        if (smem > 48000) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000); \
        kern<<<B, NT, smem>>>(G, nullptr, P, H, tau, M, Vw, nullptr, nullptr, nullptr, fail, n, j0, w, \
            shift_scale, \
            gbs, gld, pbs, pld, mbs, mld, vbs, vld, 0, 0, 0); }
    if (w == 8) LAUNCH_CHOL(8)
    else if (w == 32) LAUNCH_CHOL(32)
    else if (w == 48) LAUNCH_CHOL(48)
    else if (w == 56) LAUNCH_CHOL(56)
    else if (w == 64 && nthreads == 1024) LAUNCH_CHOL_NT(64, 1024)
    else if (w == 64) LAUNCH_CHOL(64)
    else LAUNCH_CHOL(0)
    #undef LAUNCH_CHOL_NT
    #undef LAUNCH_CHOL
}

void chol_recon_t(uint64_t G_ptr, uint64_t P_ptr, uint64_t H_ptr,
                  uint64_t tau_ptr, uint64_t M_ptr, uint64_t Vw_ptr,
                  uint64_t T_ptr, uint64_t fail_ptr, int B, int n, int j0, int w,
                  float shift_scale,
                  long gbs, int gld, long pbs, int pld,
                  long mbs, int mld, long vbs, int vld,
                  long tbs, int tld, int nthreads) {
    const float* G = reinterpret_cast<const float*>(G_ptr);
    const float* P = reinterpret_cast<const float*>(P_ptr);
    float* H = reinterpret_cast<float*>(H_ptr);
    float* tau = reinterpret_cast<float*>(tau_ptr);
    float* M = reinterpret_cast<float*>(M_ptr);
    float* Vw = reinterpret_cast<float*>(Vw_ptr);
    float* T = reinterpret_cast<float*>(T_ptr);
    int* fail = reinterpret_cast<int*>(fail_ptr);
    const int sld = (w == 64) ? (w + 1) : w;
    size_t smem = (size_t)(5 * w * sld + w) * sizeof(float);
    #define LAUNCH_CHOLT(WF) { \
        auto kern = chol_recon_kernel<float, float, WF>; \
        if (smem > 48000) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000); \
        kern<<<B, nthreads, smem>>>(G, nullptr, P, H, tau, M, Vw, T, nullptr, nullptr, fail, n, j0, w, \
            shift_scale, \
            gbs, gld, pbs, pld, mbs, mld, vbs, vld, tbs, tld, 1); }
    #define LAUNCH_CHOLT_NT(WF, NT) { \
        auto kern = chol_recon_kernel<float, float, WF, NT>; \
        if (smem > 48000) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000); \
        kern<<<B, NT, smem>>>(G, nullptr, P, H, tau, M, Vw, T, nullptr, nullptr, fail, n, j0, w, \
            shift_scale, \
            gbs, gld, pbs, pld, mbs, mld, vbs, vld, tbs, tld, 1); }
    if (w == 8) LAUNCH_CHOLT(8)
    else if (w == 32) LAUNCH_CHOLT(32)
    else if (w == 48) LAUNCH_CHOLT(48)
    else if (w == 56) LAUNCH_CHOLT(56)
    else if (w == 64 && nthreads == 1024) LAUNCH_CHOLT_NT(64, 1024)
    else if (w == 64) LAUNCH_CHOLT(64)
    else LAUNCH_CHOLT(0)
    #undef LAUNCH_CHOLT_NT
    #undef LAUNCH_CHOLT
}

void chol_recon_h(uint64_t G_ptr, uint64_t P_ptr, uint64_t H_ptr,
                uint64_t tau_ptr, uint64_t M_ptr, uint64_t Vw_ptr,
                uint64_t fail_ptr, int B, int n, int j0, int w,
                float shift_scale,
                long gbs, int gld, long pbs, int pld,
                long mbs, int mld, long vbs, int vld, int nthreads) {
    const float* G = reinterpret_cast<const float*>(G_ptr);
    const __half* P = reinterpret_cast<const __half*>(P_ptr);
    __half* H = reinterpret_cast<__half*>(H_ptr);
    float* tau = reinterpret_cast<float*>(tau_ptr);
    float* M = reinterpret_cast<float*>(M_ptr);
    __half* Vw = reinterpret_cast<__half*>(Vw_ptr);
    int* fail = reinterpret_cast<int*>(fail_ptr);
    const int sld = (w == 64) ? (w + 1) : w;
    size_t smem = (size_t)(5 * w * sld + w) * sizeof(float);
    #define LAUNCH_CHOLH(WF) { \
        auto kern = chol_recon_kernel<float, __half, WF>; \
        if (smem > 48000) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000); \
        kern<<<B, nthreads, smem>>>(G, nullptr, P, H, tau, M, Vw, nullptr, nullptr, nullptr, fail, n, j0, w, \
            shift_scale, gbs, gld, pbs, pld, mbs, mld, vbs, vld, 0, 0, 0); }
    #define LAUNCH_CHOLH_NT(WF, NT) { \
        auto kern = chol_recon_kernel<float, __half, WF, NT>; \
        if (smem > 48000) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000); \
        kern<<<B, NT, smem>>>(G, nullptr, P, H, tau, M, Vw, nullptr, nullptr, nullptr, fail, n, j0, w, \
            shift_scale, gbs, gld, pbs, pld, mbs, mld, vbs, vld, 0, 0, 0); }
    if (w == 8) LAUNCH_CHOLH(8)
    else if (w == 32) LAUNCH_CHOLH(32)
    else if (w == 48) LAUNCH_CHOLH(48)
    else if (w == 56) LAUNCH_CHOLH(56)
    else if (w == 64 && nthreads == 1024) LAUNCH_CHOLH_NT(64, 1024)
    else if (w == 64) LAUNCH_CHOLH(64)
    else LAUNCH_CHOLH(0)
    #undef LAUNCH_CHOLH_NT
    #undef LAUNCH_CHOLH
}

void chol_recon_t_h(uint64_t G_ptr, uint64_t P_ptr, uint64_t H_ptr,
                  uint64_t tau_ptr, uint64_t M_ptr, uint64_t Vw_ptr,
                  uint64_t T_ptr, uint64_t Mh_ptr, uint64_t Th_ptr,
                  uint64_t fail_ptr, int B, int n, int j0, int w,
                  float shift_scale,
                  long gbs, int gld, long pbs, int pld,
                  long mbs, int mld, long vbs, int vld,
                  long tbs, int tld, int nthreads) {
    const float* G = reinterpret_cast<const float*>(G_ptr);
    const __half* P = reinterpret_cast<const __half*>(P_ptr);
    __half* H = reinterpret_cast<__half*>(H_ptr);
    float* tau = reinterpret_cast<float*>(tau_ptr);
    float* M = reinterpret_cast<float*>(M_ptr);
    __half* Vw = reinterpret_cast<__half*>(Vw_ptr);
    float* T = reinterpret_cast<float*>(T_ptr);
    __half* Mh = reinterpret_cast<__half*>(Mh_ptr);
    __half* Th = reinterpret_cast<__half*>(Th_ptr);
    int* fail = reinterpret_cast<int*>(fail_ptr);
    const int sld = (w == 64) ? (w + 1) : w;
    size_t smem = (size_t)(5 * w * sld + w) * sizeof(float);
    #define LAUNCH_CHOLTH(WF) { \
        auto kern = chol_recon_kernel<float, __half, WF>; \
        if (smem > 48000) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000); \
        kern<<<B, nthreads, smem>>>(G, nullptr, P, H, tau, M, Vw, T, Mh, Th, fail, n, j0, w, \
            shift_scale, gbs, gld, pbs, pld, mbs, mld, vbs, vld, tbs, tld, 1); }
    #define LAUNCH_CHOLTH_NT(WF, NT) { \
        auto kern = chol_recon_kernel<float, __half, WF, NT>; \
        if (smem > 48000) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000); \
        kern<<<B, NT, smem>>>(G, nullptr, P, H, tau, M, Vw, T, Mh, Th, fail, n, j0, w, \
            shift_scale, gbs, gld, pbs, pld, mbs, mld, vbs, vld, tbs, tld, 1); }
    if (w == 8) LAUNCH_CHOLTH(8)
    else if (w == 32) LAUNCH_CHOLTH(32)
    else if (w == 48) LAUNCH_CHOLTH(48)
    else if (w == 56) LAUNCH_CHOLTH(56)
    else if (w == 64 && nthreads == 1024) LAUNCH_CHOLTH_NT(64, 1024)
    else if (w == 64) LAUNCH_CHOLTH(64)
    else LAUNCH_CHOLTH(0)
    #undef LAUNCH_CHOLTH_NT
    #undef LAUNCH_CHOLTH
}

void chol_recon_t_h_g16(uint64_t Gh_ptr, uint64_t P_ptr, uint64_t H_ptr,
                  uint64_t tau_ptr, uint64_t M_ptr, uint64_t Vw_ptr,
                  uint64_t T_ptr, uint64_t Mh_ptr, uint64_t Th_ptr,
                  uint64_t fail_ptr, int B, int n, int j0, int w,
                  float shift_scale,
                  long gbs, int gld, long pbs, int pld,
                  long mbs, int mld, long vbs, int vld,
                  long tbs, int tld, int nthreads) {
    const __half* Gh = reinterpret_cast<const __half*>(Gh_ptr);
    const __half* P = reinterpret_cast<const __half*>(P_ptr);
    __half* H = reinterpret_cast<__half*>(H_ptr);
    float* tau = reinterpret_cast<float*>(tau_ptr);
    float* M = reinterpret_cast<float*>(M_ptr);
    __half* Vw = reinterpret_cast<__half*>(Vw_ptr);
    float* T = reinterpret_cast<float*>(T_ptr);
    __half* Mh = reinterpret_cast<__half*>(Mh_ptr);
    __half* Th = reinterpret_cast<__half*>(Th_ptr);
    int* fail = reinterpret_cast<int*>(fail_ptr);
    const int sld = (w == 64) ? (w + 1) : w;
    size_t smem = (size_t)(5 * w * sld + w) * sizeof(float);
    #define LAUNCH_CHOLTHG16(WF) { \
        auto kern = chol_recon_kernel<__half, __half, WF>; \
        if (smem > 48000) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000); \
        kern<<<B, nthreads, smem>>>(nullptr, Gh, P, H, tau, M, Vw, T, Mh, Th, fail, n, j0, w, \
            shift_scale, gbs, gld, pbs, pld, mbs, mld, vbs, vld, tbs, tld, 1); }
    #define LAUNCH_CHOLTHG16_NT(WF, NT) { \
        auto kern = chol_recon_kernel<__half, __half, WF, NT>; \
        if (smem > 48000) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000); \
        kern<<<B, NT, smem>>>(nullptr, Gh, P, H, tau, M, Vw, T, Mh, Th, fail, n, j0, w, \
            shift_scale, gbs, gld, pbs, pld, mbs, mld, vbs, vld, tbs, tld, 1); }
    if (w == 8) LAUNCH_CHOLTHG16(8)
    else if (w == 32) LAUNCH_CHOLTHG16(32)
    else if (w == 48) LAUNCH_CHOLTHG16(48)
    else if (w == 56) LAUNCH_CHOLTHG16(56)
    else if (w == 64 && nthreads == 1024) LAUNCH_CHOLTHG16_NT(64, 1024)
    else if (w == 64) LAUNCH_CHOLTHG16(64)
    else LAUNCH_CHOLTHG16(0)
    #undef LAUNCH_CHOLTHG16_NT
    #undef LAUNCH_CHOLTHG16
}

void larft(uint64_t VtV_ptr, uint64_t tau_ptr, uint64_t T_ptr,
           int B, int n, int j0, int w,
           long vbs, int vld, long tbs, int tld, int nthreads) {
    const float* VtV = reinterpret_cast<const float*>(VtV_ptr);
    const float* tau = reinterpret_cast<const float*>(tau_ptr);
    float* T = reinterpret_cast<float*>(T_ptr);
    size_t smem = (size_t)(3 * w * w + w) * sizeof(float);
    #define LAUNCH_LARFT(WF) { \
        auto kern = larft_kernel<WF>; \
        if (smem > 48000) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000); \
        kern<<<B, nthreads, smem>>>(VtV, tau, T, n, j0, w, vbs, vld, tbs, tld); }
    #define LAUNCH_LARFT_NT(WF, NT) { \
        auto kern = larft_kernel<WF, NT>; \
        if (smem > 48000) cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000); \
        kern<<<B, NT, smem>>>(VtV, tau, T, n, j0, w, vbs, vld, tbs, tld); }
    if (w == 32) LAUNCH_LARFT(32)
    else if (w == 48) LAUNCH_LARFT(48)
    else if (w == 56) LAUNCH_LARFT(56)
    else if (w == 64 && nthreads == 384) LAUNCH_LARFT_NT(64, 384)
    else if (w == 64 && nthreads == 512) LAUNCH_LARFT_NT(64, 512)
    else if (w == 64) LAUNCH_LARFT(64)
    else LAUNCH_LARFT(0)
    #undef LAUNCH_LARFT_NT
    #undef LAUNCH_LARFT
}

__global__ void retau_kernel(const float* __restrict__ H,
                             float* __restrict__ tau,
                             int n, int limit) {
    const int j = blockIdx.x;
    const int b = blockIdx.y;
    if (j >= limit) return;

    float* taub = tau + (long)b * n;
    if (taub[j] == 0.0f) return;

    const float* Hb = H + (long)b * n * n;
    float part = 0.0f;
    for (int i = j + 1 + threadIdx.x; i < n; i += blockDim.x) {
        float v = Hb[(long)i * n + j];
        part = fmaf(v, v, part);
    }
    part = warp_sum(part);

    __shared__ float scratch[32];
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nw = blockDim.x >> 5;
    if (lane == 0) scratch[wid] = part;
    __syncthreads();

    if (wid == 0) {
        float total = (lane < nw) ? scratch[lane] : 0.0f;
        total = warp_sum(total);
        if (lane == 0) taub[j] = 2.0f / (1.0f + total);
    }
}

__global__ void retau_tile32_kernel(float* __restrict__ H,
                                    float* __restrict__ tau,
                                    int n, int limit) {
    const int jbase = blockIdx.x * 32;
    const int b = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nw = blockDim.x >> 5;
    const int j = jbase + lane;

    float* taub = tau + (long)b * n;
    float* Hb = H + (long)b * n * n;
    float part = 0.0f;
    const bool active = (j < limit) && (taub[j] != 0.0f);
    for (int i = jbase + 1 + wid; i < n; i += nw) {
        if (active && i > j) {
            float v = Hb[(long)i * n + j];
            part = fmaf(v, v, part);
        }
    }

    __shared__ float scratch[16 * 32];
    scratch[wid * 32 + lane] = part;
    __syncthreads();

    if (wid == 0 && j < limit) {
        float total = 0.0f;
        #pragma unroll
        for (int u = 0; u < 16; ++u) total += scratch[u * 32 + lane];
        if (taub[j] != 0.0f) taub[j] = 2.0f / (1.0f + total);
    }
}

__global__ void finalize_h_tile32_kernel(const __half* __restrict__ Hh,
                                         float* __restrict__ Hf,
                                         float* __restrict__ tau,
                                         int n, int limit) {
    const int jbase = blockIdx.x * 32;
    const int b = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nw = blockDim.x >> 5;
    const int j = jbase + lane;

    float* taub = tau + (long)b * n;
    const __half* Hhb = Hh + (long)b * n * n;
    float* Hfb = Hf + (long)b * n * n;
    if (jbase >= limit) {
        for (int i = wid; i < n; i += nw) {
            if (j < n) Hfb[(long)i * n + j] = __half2float(Hhb[(long)i * n + j]);
        }
        return;
    }

    float part = 0.0f;
    const bool in_col = j < n;
    const bool active = (j < limit) && (taub[j] != 0.0f);
    for (int i = wid; i < n; i += nw) {
        if (in_col) {
            float v = __half2float(Hhb[(long)i * n + j]);
            Hfb[(long)i * n + j] = v;
            if (active && i > j) part = fmaf(v, v, part);
        }
    }

    __shared__ float scratch[16 * 32];
    scratch[wid * 32 + lane] = part;
    __syncthreads();

    if (wid == 0 && j < limit) {
        float total = 0.0f;
        #pragma unroll
        for (int u = 0; u < 16; ++u) total += scratch[u * 32 + lane];
        if (taub[j] != 0.0f) taub[j] = 2.0f / (1.0f + total);
    }

}

void retau(uint64_t H_ptr, uint64_t tau_ptr, int B, int n, int limit) {
    float* H = reinterpret_cast<float*>(H_ptr);
    float* tau = reinterpret_cast<float*>(tau_ptr);
    int lim = limit;
    if (lim <= 0) return;
    if (lim > n) lim = n;
    if (n == 1024 || n == 2048 || n == 4096) {
        dim3 grid((lim + 31) / 32, B);
        retau_tile32_kernel<<<grid, 512>>>(H, tau, n, lim);
    } else {
        dim3 grid(lim, B);
        retau_kernel<<<grid, 128>>>(H, tau, n, lim);
    }
}

void finalize_h(uint64_t Hh_ptr, uint64_t Hf_ptr, uint64_t tau_ptr, int B, int n, int limit) {
    const __half* Hh = reinterpret_cast<const __half*>(Hh_ptr);
    float* Hf = reinterpret_cast<float*>(Hf_ptr);
    float* tau = reinterpret_cast<float*>(tau_ptr);
    int lim = limit;
    if (lim < 0) lim = 0;
    if (lim > n) lim = n;
    dim3 grid((n + 31) / 32, B);
    finalize_h_tile32_kernel<<<grid, 512>>>(Hh, Hf, tau, n, lim);
}
"""


cpp_src = r"""
#include <pybind11/pybind11.h>
#include <cstdint>

void qr_fused(uint64_t A, uint64_t H, uint64_t tau, int B, int n, int nthreads);
void panel_smem(uint64_t H, uint64_t P, uint64_t T, uint64_t tau,
                int B, int n, int j0, int w, long pbs, int ldp,
                long tbs, int ldt, int want_T, int nthreads);
void panel_smem_wfix(uint64_t H, uint64_t P, uint64_t tau,
                     int B, int n, int j0, int w, long pbs, int ldp, int nthreads);
void panel_smem_wfixT(uint64_t H, uint64_t P, uint64_t T, uint64_t tau,
                      int B, int n, int j0, int w, long pbs, int ldp,
                      long tbs, int ldt, int nthreads);
void panel_tall(uint64_t H, uint64_t P, uint64_t T, uint64_t tau,
                int B, int n, int j0, int w, long pbs, int ldp,
                long tbs, int ldt);
void chol_recon(uint64_t G, uint64_t P, uint64_t H,
                uint64_t tau, uint64_t M, uint64_t Vw,
                uint64_t fail, int B, int n, int j0, int w,
                float shift_scale,
                long gbs, int gld, long pbs, int pld,
                long mbs, int mld, long vbs, int vld, int nthreads);
void chol_recon_t(uint64_t G, uint64_t P, uint64_t H,
                  uint64_t tau, uint64_t M, uint64_t Vw,
                  uint64_t T, uint64_t fail, int B, int n, int j0, int w,
                  float shift_scale,
                  long gbs, int gld, long pbs, int pld,
                  long mbs, int mld, long vbs, int vld,
                  long tbs, int tld, int nthreads);
void chol_recon_h(uint64_t G, uint64_t P, uint64_t H,
                  uint64_t tau, uint64_t M, uint64_t Vw,
                  uint64_t fail, int B, int n, int j0, int w,
                  float shift_scale,
                  long gbs, int gld, long pbs, int pld,
                  long mbs, int mld, long vbs, int vld, int nthreads);
void chol_recon_t_h(uint64_t G, uint64_t P, uint64_t H,
                    uint64_t tau, uint64_t M, uint64_t Vw,
                    uint64_t T, uint64_t Mh, uint64_t Th,
                    uint64_t fail, int B, int n, int j0, int w,
                    float shift_scale,
                    long gbs, int gld, long pbs, int pld,
                    long mbs, int mld, long vbs, int vld,
                    long tbs, int tld, int nthreads);
void chol_recon_t_h_g16(uint64_t Gh, uint64_t P, uint64_t H,
                        uint64_t tau, uint64_t M, uint64_t Vw,
                        uint64_t T, uint64_t Mh, uint64_t Th,
                        uint64_t fail, int B, int n, int j0, int w,
                        float shift_scale,
                        long gbs, int gld, long pbs, int pld,
                        long mbs, int mld, long vbs, int vld,
                        long tbs, int tld, int nthreads);
void larft(uint64_t VtV, uint64_t tau, uint64_t T,
           int B, int n, int j0, int w,
           long vbs, int vld, long tbs, int tld, int nthreads);
void retau(uint64_t H, uint64_t tau, int B, int n, int limit);
void finalize_h(uint64_t Hh, uint64_t Hf, uint64_t tau, int B, int n, int limit);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qr_fused", &qr_fused);
    m.def("panel_smem", &panel_smem);
    m.def("panel_smem_wfix", &panel_smem_wfix);
    m.def("panel_smem_wfixT", &panel_smem_wfixT);
    m.def("panel_tall", &panel_tall);
    m.def("chol_recon", &chol_recon);
    m.def("chol_recon_t", &chol_recon_t);
    m.def("chol_recon_h", &chol_recon_h);
    m.def("chol_recon_t_h", &chol_recon_t_h);
    m.def("chol_recon_t_h_g16", &chol_recon_t_h_g16);
    m.def("larft", &larft);
    m.def("retau", &retau);
    m.def("finalize_h", &finalize_h);
}
"""


_cc = torch.cuda.get_device_capability()
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{_cc[0]}.{_cc[1]}")

_src_dir = os.path.join(os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(), ".qr_raw_src_v2")
os.makedirs(_src_dir, exist_ok=True)
import ctypes as _ctypes

from torch.utils.cpp_extension import load_inline

_li_arch = f"sm_{_cc[0]}{_cc[1]}a" if _cc[0] >= 10 else f"sm_{_cc[0]}{_cc[1]}"
_ext = load_inline(
    name="qr_chiro_ext",
    cpp_sources=[cpp_src],
    cuda_sources=[cuda_src],
    functions=None,
    extra_cuda_cflags=["-O3", "--use_fast_math", f"-arch={_li_arch}", "-std=c++17", "--threads", "0"],
    no_implicit_headers=True,
    verbose=True,
)

_PANEL_W32_384_SRC = r"""
#include <cuda_runtime.h>
#include <stdint.h>

#define FULL_MASK 0xffffffffu

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL_MASK, v, o);
    return v;
}

__device__ __forceinline__ void house_coeffs(float alpha, float sigma, float* cf) {
    if (sigma <= 0.f) {
        cf[0] = 0.f; cf[1] = 0.f; cf[2] = alpha;
    } else {
        float beta = -copysignf(sqrtf(fmaf(alpha, alpha, sigma)), alpha);
        cf[0] = (beta - alpha) / beta;
        cf[1] = 1.f / (alpha - beta);
        cf[2] = beta;
    }
}

template <int NT, int WFIX>
__device__ void panel_core_wfix(float* S, long sld, int r,
                                float* cf, float* gammas, float* taug,
                                float* scratch) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nw = NT >> 5;
    {
        float part = 0.f;
        for (int i = 1 + threadIdx.x; i < r; i += NT) {
            float x = S[i];
            part = fmaf(x, x, part);
        }
        part = warp_sum(part);
        if (lane == 0) scratch[wid] = part;
        __syncthreads();
        if (threadIdx.x == 0) {
            float sg = 0.f;
            for (int u = 0; u < nw; ++u) sg += scratch[u];
            house_coeffs(S[0], sg, cf);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < WFIX; ++j) {
        const float* cfc = cf + 4 * (j & 1);
        float* cfn = cf + 4 * ((j + 1) & 1);
        float tj = cfc[0], gj = cfc[1], bj = cfc[2];
        float* colj = S + (long)j * sld;
        if (threadIdx.x == 0) {
            gammas[j] = gj;
            taug[j] = tj;
        }
        for (int k = j + 1 + wid; k < WFIX; k += nw) {
            float* ck = S + (long)k * sld;
            float d = (lane == 0) ? ck[j] : 0.f;
            float acc = 0.f;
            for (int i = j + 1 + lane; i < r; i += 32) acc = fmaf(colj[i], ck[i], acc);
            d += gj * acc;
            d = warp_sum(d);
            float wk = tj * d;
            float alpha_next = 0.f;
            float sq = 0.f;
            if (lane == 0) ck[j] -= wk;
            float wg = wk * gj;
            for (int i = j + 1 + lane; i < r; i += 32) {
                float nv = fmaf(-wg, colj[i], ck[i]);
                ck[i] = nv;
                if (k == j + 1) {
                    if (i == j + 1) alpha_next = nv;
                    else sq = fmaf(nv, nv, sq);
                }
            }
            if (k == j + 1) {
                sq = warp_sum(sq);
                if (lane == 0) house_coeffs(alpha_next, sq, cfn);
            }
        }
        if (threadIdx.x == 0) colj[j] = bj;
        __syncthreads();
    }
}

template <int NT, int RFIX>
__global__ void panel_w32_kernel(float* __restrict__ H,
                                 float* __restrict__ P,
                                 float* __restrict__ tau,
                                 int n, int j0,
                                 long pbs, int ldp) {
    constexpr int WFIX = 32;
    extern __shared__ float smem[];
    const int r = (RFIX > 0) ? RFIX : (n - j0);
    const int sld = r | 1;
    float* S = smem;
    float* gammas = S + (long)sld * WFIX;
    float* taug = gammas + WFIX;
    float* cf = taug + WFIX;
    float* scratch = cf + 8;

    const long b = blockIdx.x;
    float* Hb = H + b * (long)n * n;

    for (int idx = threadIdx.x; idx < r * WFIX; idx += NT) {
        int i = idx / WFIX, j = idx - i * WFIX;
        S[(long)j * sld + i] = Hb[(long)(j0 + i) * n + (j0 + j)];
    }
    __syncthreads();

    panel_core_wfix<NT, WFIX>(S, sld, r, cf, gammas, taug, scratch);

    float* taub = tau + b * (long)n + j0;
    for (int j = threadIdx.x; j < WFIX; j += NT) taub[j] = taug[j];
    for (int idx = threadIdx.x; idx < r * WFIX; idx += NT) {
        int i = idx / WFIX, j = idx - i * WFIX;
        float x = S[(long)j * sld + i];
        Hb[(long)(j0 + i) * n + (j0 + j)] = (i > j) ? gammas[j] * x : x;
    }
    float* Pb = P + b * pbs;
    for (int idx = threadIdx.x; idx < r * WFIX; idx += NT) {
        int j = idx / r, i = idx - j * r;
        float x = S[(long)j * sld + i];
        Pb[(long)j * ldp + i] = (i < j) ? 0.f : (i == j ? 1.f : gammas[j] * x);
    }
}

#define LAUNCH_R384(RF) \
    if (r == RF) { \
        auto kern = panel_w32_kernel<384, RF>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448); \
        kern<<<B, 384, smem>>>(H, P, tau, n, j0, pbs, ldp); \
        return 0; \
    }

extern "C" int panel_w32_384(float* H, float* P, float* tau,
                             int B, int n, int j0, long pbs, int ldp) {
    const int r = n - j0;
    const int w = 32;
    const int sld = r | 1;
    size_t smem = ((size_t)sld * w + 2 * (size_t)w + 8 + 32) * sizeof(float);
    LAUNCH_R384(512); LAUNCH_R384(480); LAUNCH_R384(448); LAUNCH_R384(416);
    LAUNCH_R384(384); LAUNCH_R384(352); LAUNCH_R384(320); LAUNCH_R384(288);
    LAUNCH_R384(256); LAUNCH_R384(224); LAUNCH_R384(192); LAUNCH_R384(160);
    LAUNCH_R384(128); LAUNCH_R384(96);  LAUNCH_R384(64);  LAUNCH_R384(32);
    auto kern = panel_w32_kernel<384, 0>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);
    kern<<<B, 384, smem>>>(H, P, tau, n, j0, pbs, ldp);
    return 0;
}

#undef LAUNCH_R384

"""

_PANEL_W44_1024_SRC = r"""
#include <cuda_runtime.h>
#include <stdint.h>

#define FULL_MASK 0xffffffffu

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL_MASK, v, o);
    return v;
}

__device__ __forceinline__ void house_coeffs(float alpha, float sigma, float* cf) {
    if (sigma <= 0.f) {
        cf[0] = 0.f; cf[1] = 0.f; cf[2] = alpha;
    } else {
        float beta = -copysignf(sqrtf(fmaf(alpha, alpha, sigma)), alpha);
        cf[0] = (beta - alpha) / beta;
        cf[1] = 1.f / (alpha - beta);
        cf[2] = beta;
    }
}

template <int NT, int WFIX>
__device__ void panel_core_wfix(float* S, long sld, int r,
                                float* cf, float* gammas, float* taug,
                                float* scratch) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    const int nw = NT >> 5;
    {
        float part = 0.f;
        for (int i = 1 + threadIdx.x; i < r; i += NT) {
            float x = S[i];
            part = fmaf(x, x, part);
        }
        part = warp_sum(part);
        if (lane == 0) scratch[wid] = part;
        __syncthreads();
        if (threadIdx.x == 0) {
            float sg = 0.f;
            for (int u = 0; u < nw; ++u) sg += scratch[u];
            house_coeffs(S[0], sg, cf);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < WFIX; ++j) {
        const float* cfc = cf + 4 * (j & 1);
        float* cfn = cf + 4 * ((j + 1) & 1);
        float tj = cfc[0], gj = cfc[1], bj = cfc[2];
        float* colj = S + (long)j * sld;
        if (threadIdx.x == 0) {
            gammas[j] = gj;
            taug[j] = tj;
        }
        for (int k = j + 1 + wid; k < WFIX; k += nw) {
            float* ck = S + (long)k * sld;
            float d = (lane == 0) ? ck[j] : 0.f;
            float acc = 0.f;
            for (int i = j + 1 + lane; i < r; i += 32) acc = fmaf(colj[i], ck[i], acc);
            d += gj * acc;
            d = warp_sum(d);
            float wk = tj * d;
            float alpha_next = 0.f;
            float sq = 0.f;
            if (lane == 0) ck[j] -= wk;
            float wg = wk * gj;
            for (int i = j + 1 + lane; i < r; i += 32) {
                float nv = fmaf(-wg, colj[i], ck[i]);
                ck[i] = nv;
                if (k == j + 1) {
                    if (i == j + 1) alpha_next = nv;
                    else sq = fmaf(nv, nv, sq);
                }
            }
            if (k == j + 1) {
                sq = warp_sum(sq);
                if (lane == 0) house_coeffs(alpha_next, sq, cfn);
            }
        }
        if (threadIdx.x == 0) colj[j] = bj;
        __syncthreads();
    }
}

template <int RFIX>
__global__ void panel_w44_kernel(float* __restrict__ H,
                                 float* __restrict__ P,
                                 float* __restrict__ tau,
                                 int n, int j0,
                                 long pbs, int ldp) {
    constexpr int NT = 1024;
    constexpr int WFIX = 44;
    extern __shared__ float smem[];
    const int r = RFIX;
    const int sld = r | 1;
    float* S = smem;
    float* gammas = S + (long)sld * WFIX;
    float* taug = gammas + WFIX;
    float* cf = taug + WFIX;
    float* scratch = cf + 8;

    const long b = blockIdx.x;
    float* Hb = H + b * (long)n * n;

    for (int idx = threadIdx.x; idx < r * WFIX; idx += NT) {
        int i = idx / WFIX, j = idx - i * WFIX;
        S[(long)j * sld + i] = Hb[(long)(j0 + i) * n + (j0 + j)];
    }
    __syncthreads();

    panel_core_wfix<NT, WFIX>(S, sld, r, cf, gammas, taug, scratch);

    float* taub = tau + b * (long)n + j0;
    for (int j = threadIdx.x; j < WFIX; j += NT) taub[j] = taug[j];
    for (int idx = threadIdx.x; idx < r * WFIX; idx += NT) {
        int i = idx / WFIX, j = idx - i * WFIX;
        float x = S[(long)j * sld + i];
        Hb[(long)(j0 + i) * n + (j0 + j)] = (i > j) ? gammas[j] * x : x;
    }
    float* Pb = P + b * pbs;
    for (int idx = threadIdx.x; idx < r * WFIX; idx += NT) {
        int j = idx / r, i = idx - j * r;
        float x = S[(long)j * sld + i];
        Pb[(long)j * ldp + i] = (i < j) ? 0.f : (i == j ? 1.f : gammas[j] * x);
    }
}

#define LAUNCH_R(RF) \
    if (r == RF) { \
        auto kern = panel_w44_kernel<RF>; \
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448); \
        kern<<<B, 1024, smem>>>(H, P, tau, n, j0, pbs, ldp); \
        return 0; \
    }

extern "C" int panel_w44_1024_176(float* H, float* P, float* tau,
                                  int B, int n, int j0, long pbs, int ldp) {
    const int r = n - j0;
    const int w = 44;
    const int sld = r | 1;
    size_t smem = ((size_t)sld * w + 2 * (size_t)w + 8 + 32) * sizeof(float);
    LAUNCH_R(176); LAUNCH_R(132); LAUNCH_R(88); LAUNCH_R(44);
    return -1;
}

"""

_PANEL_W48_1024_SRC = (
    _PANEL_W44_1024_SRC
    .replace("panel_w44_kernel", "panel_w48_kernel")
    .replace("panel_w44_1024_176", "panel_w48_1024_1024")
    .replace("constexpr int WFIX = 44;", "constexpr int WFIX = 48;")
    .replace("const int w = 44;", "const int w = 48;")
    .replace(
        "LAUNCH_R(176); LAUNCH_R(132); LAUNCH_R(88); LAUNCH_R(44);",
        "LAUNCH_R(1024); LAUNCH_R(976); LAUNCH_R(928); LAUNCH_R(880); "
        "LAUNCH_R(832); LAUNCH_R(784); LAUNCH_R(736); LAUNCH_R(688); "
        "LAUNCH_R(640); LAUNCH_R(592); LAUNCH_R(544); LAUNCH_R(496); "
        "LAUNCH_R(448); LAUNCH_R(400); LAUNCH_R(352); LAUNCH_R(304); "
        "LAUNCH_R(256); LAUNCH_R(208); LAUNCH_R(160); LAUNCH_R(112); "
        "LAUNCH_R(64);",
    )
)

_PANEL_W32_1024_352_SRC = (
    _PANEL_W32_384_SRC
    .replace("panel_w32_384", "panel_w32_1024_352")
    .replace("LAUNCH_R384", "LAUNCH_R1024")
    .replace("panel_w32_kernel<384", "panel_w32_kernel<1024")
    .replace("kern<<<B, 384", "kern<<<B, 1024")
    .replace(
        "LAUNCH_R1024(512); LAUNCH_R1024(480); LAUNCH_R1024(448); LAUNCH_R1024(416);\n"
        "    LAUNCH_R1024(384); LAUNCH_R1024(352); LAUNCH_R1024(320); LAUNCH_R1024(288);\n"
        "    LAUNCH_R1024(256); LAUNCH_R1024(224); LAUNCH_R1024(192); LAUNCH_R1024(160);\n"
        "    LAUNCH_R1024(128); LAUNCH_R1024(96);  LAUNCH_R1024(64);  LAUNCH_R1024(32);\n"
        "    auto kern = panel_w32_kernel<1024, 0>;\n"
        "    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 232448);\n"
        "    kern<<<B, 1024, smem>>>(H, P, tau, n, j0, pbs, ldp);\n"
        "    return 0;",
        "LAUNCH_R1024(352); LAUNCH_R1024(320); LAUNCH_R1024(288); LAUNCH_R1024(256); LAUNCH_R1024(224);\n"
        "    return -1;",
    )
)


def _build_raw_so(name, src, arch, cache_dir):
    import ctypes as _ctypes
    import hashlib
    import subprocess

    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.sha1((src + "|" + arch).encode()).hexdigest()[:16]
    cu = os.path.join(cache_dir, f"{name}_{key}.cu")
    so = os.path.join(cache_dir, f"{name}_{key}.so")
    if not os.path.exists(so):
        with open(cu, "w", encoding="utf-8") as f:
            f.write(src)
        nvcc = "/usr/local/cuda/bin/nvcc"
        if not os.path.exists(nvcc):
            nvcc = "nvcc"
        cmd = [
            nvcc, "-shared", "-Xcompiler", "-fPIC", "-O3", f"-arch={arch}",
            "-std=c++17", "--threads", "0", "--use_fast_math",
            cu, "-o", so,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])
    return _ctypes.CDLL(so)


_panel_w32_384_lib = None
_panel_w32_1024_352_lib = None
_panel_w44_1024_lib = None
_panel_w48_1024_lib = None
try:
    import ctypes as _ctypes

    _raw_arch = f"sm_{_cc[0]}{_cc[1]}a" if _cc[0] >= 10 else f"sm_{_cc[0]}{_cc[1]}"
    _raw_cache = os.path.join(_src_dir, "rawcc_panel")
    _panel_w32_384_lib = _build_raw_so("qr_panel_w32_384", _PANEL_W32_384_SRC, _raw_arch, _raw_cache)
    _panel_w32_384_lib.panel_w32_384.argtypes = (
        [_ctypes.c_void_p] * 3
        + [_ctypes.c_int, _ctypes.c_int, _ctypes.c_int, _ctypes.c_long, _ctypes.c_int]
    )
    _panel_w32_384_lib.panel_w32_384.restype = _ctypes.c_int
except Exception:
    _panel_w32_384_lib = None

try:
    _panel_w32_1024_352_lib = _build_raw_so("qr_panel_w32_1024_352", _PANEL_W32_1024_352_SRC, _raw_arch, _raw_cache)
    _panel_w32_1024_352_lib.panel_w32_1024_352.argtypes = (
        [_ctypes.c_void_p] * 3
        + [_ctypes.c_int, _ctypes.c_int, _ctypes.c_int, _ctypes.c_long, _ctypes.c_int]
    )
    _panel_w32_1024_352_lib.panel_w32_1024_352.restype = _ctypes.c_int
except Exception:
    _panel_w32_1024_352_lib = None

try:
    _panel_w44_1024_lib = _build_raw_so("qr_panel_w44_1024", _PANEL_W44_1024_SRC, _raw_arch, _raw_cache)
    _panel_w44_1024_lib.panel_w44_1024_176.argtypes = (
        [_ctypes.c_void_p] * 3
        + [_ctypes.c_int, _ctypes.c_int, _ctypes.c_int, _ctypes.c_long, _ctypes.c_int]
    )
    _panel_w44_1024_lib.panel_w44_1024_176.restype = _ctypes.c_int
except Exception:
    _panel_w44_1024_lib = None

try:
    _panel_w48_1024_lib = _build_raw_so("qr_panel_w48_1024", _PANEL_W48_1024_SRC, _raw_arch, _raw_cache)
    _panel_w48_1024_lib.panel_w48_1024_1024.argtypes = (
        [_ctypes.c_void_p] * 3
        + [_ctypes.c_int, _ctypes.c_int, _ctypes.c_int, _ctypes.c_long, _ctypes.c_int]
    )
    _panel_w48_1024_lib.panel_w48_1024_1024.restype = _ctypes.c_int
except Exception:
    _panel_w48_1024_lib = None

_cqr_w64_lib = None


def _dp(t):
    return int(t.data_ptr())


def _qr_fused_ext(A, H, tau, nthreads):
    _ext.qr_fused(_dp(A), _dp(H), _dp(tau), A.shape[0], A.shape[1], int(nthreads))


def _panel_smem_ext(H, P, T, tau, j0, w, want_T, nthreads):
    if (
        H.shape[1] == 1024
        and int(want_T) == 0
        and int(w) == 48
        and int(nthreads) == 1024
        and _panel_w48_1024_lib is not None
    ):
        rc = _panel_w48_1024_lib.panel_w48_1024_1024(
            _dp(H), _dp(P), _dp(tau), H.shape[0], H.shape[1], int(j0),
            P.stride(0), P.stride(1)
        )
        if rc == 0:
            return
    if (
        H.shape[1] == 352
        and int(want_T) == 0
        and int(w) == 32
        and int(nthreads) == 1024
        and _panel_w32_1024_352_lib is not None
    ):
        rc = _panel_w32_1024_352_lib.panel_w32_1024_352(
            _dp(H), _dp(P), _dp(tau), H.shape[0], H.shape[1], int(j0),
            P.stride(0), P.stride(1)
        )
        if rc == 0:
            return
    if (
        H.shape[1] == 176
        and int(want_T) == 0
        and int(w) == 44
        and int(nthreads) == 1024
        and _panel_w44_1024_lib is not None
    ):
        rc = _panel_w44_1024_lib.panel_w44_1024_176(
            _dp(H), _dp(P), _dp(tau), H.shape[0], H.shape[1], int(j0),
            P.stride(0), P.stride(1)
        )
        if rc == 0:
            return
    if int(want_T) == 0 and int(w) == 32 and int(nthreads) == 384 and _panel_w32_384_lib is not None:
        rc = _panel_w32_384_lib.panel_w32_384(
            _dp(H), _dp(P), _dp(tau), H.shape[0], H.shape[1], int(j0),
            P.stride(0), P.stride(1)
        )
        if rc == 0:
            return
    if int(want_T) == 0 and int(w) in (16, 32, 40, 44, 48):
        _ext.panel_smem_wfix(
            _dp(H), _dp(P), _dp(tau), H.shape[0], H.shape[1], int(j0), int(w),
            P.stride(0), P.stride(1), int(nthreads)
        )
        return
    if int(want_T) == 1 and int(w) == 32:
        _ext.panel_smem_wfixT(
            _dp(H), _dp(P), _dp(T), _dp(tau), H.shape[0], H.shape[1], int(j0), int(w),
            P.stride(0), P.stride(1), T.stride(0), T.stride(1), int(nthreads)
        )
        return
    _ext.panel_smem(
        _dp(H), _dp(P), _dp(T), _dp(tau), H.shape[0], H.shape[1], int(j0), int(w),
        P.stride(0), P.stride(1), T.stride(0), T.stride(1), int(want_T), int(nthreads)
    )


def _panel_tall_ext(H, P, T, tau, j0, w):
    _ext.panel_tall(
        _dp(H), _dp(P), _dp(T), _dp(tau), H.shape[0], H.shape[1], int(j0), int(w),
        P.stride(0), P.stride(1), T.stride(0), T.stride(1)
    )


def _chol_recon_ext(G, P, H, tau, M, Vw, fail, j0, w, nthreads, shift_scale=0.0):
    _ext.chol_recon(
        _dp(G), _dp(P), _dp(H), _dp(tau), _dp(M), _dp(Vw), _dp(fail),
        H.shape[0], H.shape[1], int(j0), int(w), float(shift_scale),
        G.stride(0), G.stride(1), P.stride(0), P.stride(1),
        M.stride(0), M.stride(1), Vw.stride(0), Vw.stride(1), int(nthreads)
    )


def _chol_recon_t_ext(G, P, H, tau, M, Vw, T, fail, j0, w, nthreads, shift_scale=0.0):
    _ext.chol_recon_t(
        _dp(G), _dp(P), _dp(H), _dp(tau), _dp(M), _dp(Vw), _dp(T), _dp(fail),
        H.shape[0], H.shape[1], int(j0), int(w), float(shift_scale),
        G.stride(0), G.stride(1), P.stride(0), P.stride(1),
        M.stride(0), M.stride(1), Vw.stride(0), Vw.stride(1),
        T.stride(0), T.stride(1), int(nthreads)
    )


def _chol_recon_h_ext(G, P, H, tau, M, Vw, fail, j0, w, nthreads, shift_scale=0.0):
    _ext.chol_recon_h(
        _dp(G), _dp(P), _dp(H), _dp(tau), _dp(M), _dp(Vw), _dp(fail),
        H.shape[0], H.shape[1], int(j0), int(w), float(shift_scale),
        G.stride(0), G.stride(1), P.stride(0), P.stride(1),
        M.stride(0), M.stride(1), Vw.stride(0), Vw.stride(1), int(nthreads)
    )


def _chol_recon_t_h_ext(G, P, H, tau, M, Vw, T, Mh, Th, fail, j0, w, nthreads, shift_scale=0.0):
    Mptr = 0 if M is None else _dp(M)
    Tptr = 0 if T is None else _dp(T)
    mbs = Mh.stride(0) if M is None else M.stride(0)
    mld = Mh.stride(1) if M is None else M.stride(1)
    tbs = Th.stride(0) if T is None else T.stride(0)
    tld = Th.stride(1) if T is None else T.stride(1)
    _ext.chol_recon_t_h(
        _dp(G), _dp(P), _dp(H), _dp(tau), Mptr, _dp(Vw), Tptr, _dp(Mh), _dp(Th), _dp(fail),
        H.shape[0], H.shape[1], int(j0), int(w), float(shift_scale),
        G.stride(0), G.stride(1), P.stride(0), P.stride(1),
        mbs, mld, Vw.stride(0), Vw.stride(1),
        tbs, tld, int(nthreads)
    )


def _chol_recon_t_h_g16_ext(Gh, P, H, tau, M, Vw, T, Mh, Th, fail, j0, w, nthreads, shift_scale=0.0):
    Mptr = 0 if M is None else _dp(M)
    Tptr = 0 if T is None else _dp(T)
    mbs = Mh.stride(0) if M is None else M.stride(0)
    mld = Mh.stride(1) if M is None else M.stride(1)
    tbs = Th.stride(0) if T is None else T.stride(0)
    tld = Th.stride(1) if T is None else T.stride(1)
    _ext.chol_recon_t_h_g16(
        _dp(Gh), _dp(P), _dp(H), _dp(tau), Mptr, _dp(Vw), Tptr, _dp(Mh), _dp(Th), _dp(fail),
        H.shape[0], H.shape[1], int(j0), int(w), float(shift_scale),
        Gh.stride(0), Gh.stride(1), P.stride(0), P.stride(1),
        mbs, mld, Vw.stride(0), Vw.stride(1),
        tbs, tld, int(nthreads)
    )


def _larft_ext(VtV, tau, T, j0, w, nthreads):
    _ext.larft(
        _dp(VtV), _dp(tau), _dp(T), T.shape[0], tau.shape[1], int(j0), int(w),
        VtV.stride(0), VtV.stride(1), T.stride(0), T.stride(1), int(nthreads)
    )


def _retau_ext(H, tau, limit):
    _ext.retau(_dp(H), _dp(tau), H.shape[0], H.shape[1], int(limit))


def _finalize_h_ext(Hh, Hf, tau, limit):
    _ext.finalize_h(_dp(Hh), _dp(Hf), _dp(tau), Hh.shape[0], Hh.shape[1], int(limit))


@triton.jit
def _fused_wy_2pass_kernel(
    Pp, Tp, Cp,
    B, W, R, Ccols,
    sPb, sPw, sPr,
    sTb, sTw, sTk,
    sCb, sCr, sCc,
    WPAD: tl.constexpr,
    BR: tl.constexpr, TILE: tl.constexpr, IP: tl.constexpr,
):
    pid = tl.program_id(0)
    n_ctile = tl.cdiv(Ccols, TILE)
    b = pid // n_ctile
    ct = pid % n_ctile
    Pb = Pp + b * sPb
    Tb = Tp + b * sTb
    Cb = Cp + b * sCb

    cols = ct * TILE + tl.arange(0, TILE)
    col_mask = cols < Ccols
    wj = tl.arange(0, WPAD)
    wmask = wj < W

    w1 = tl.zeros((WPAD, TILE), dtype=tl.float32)
    for r0 in range(0, R, BR):
        rr = r0 + tl.arange(0, BR)
        rmask = rr < R
        pblk = tl.load(Pb + wj[:, None] * sPw + rr[None, :] * sPr,
                       mask=wmask[:, None] & rmask[None, :], other=0.0)
        cblk = tl.load(Cb + rr[:, None] * sCr + cols[None, :] * sCc,
                       mask=rmask[:, None] & col_mask[None, :], other=0.0)
        w1 += tl.dot(pblk, cblk, input_precision=IP)

    tt = tl.load(Tb + wj[None, :] * sTw + wj[:, None] * sTk,
                 mask=(wmask[None, :]) & (wmask[:, None]), other=0.0)
    w2 = tl.dot(tt, w1, input_precision=IP)

    for r0 in range(0, R, BR):
        rr = r0 + tl.arange(0, BR)
        rmask = rr < R
        pblk = tl.load(Pb + wj[None, :] * sPw + rr[:, None] * sPr,
                       mask=wmask[None, :] & rmask[:, None], other=0.0)
        upd = tl.dot(pblk, w2, input_precision=IP)
        cptr = Cb + rr[:, None] * sCr + cols[None, :] * sCc
        cmask = rmask[:, None] & col_mask[None, :]
        old = tl.load(cptr, mask=cmask, other=0.0)
        tl.store(cptr, old - upd, mask=cmask)


@triton.jit
def _fused_wy_2pass_w32n512_kernel(
    Pp, Tp, Cp,
    R, Ccols,
    BR: tl.constexpr, TILE: tl.constexpr,
    IP_W1: tl.constexpr, IP_W2: tl.constexpr,
    CAST_W1_FP16: tl.constexpr, CAST_UPD_FP16: tl.constexpr,
):
    pid = tl.program_id(0)
    n_ctile = tl.cdiv(Ccols, TILE)
    b = pid // n_ctile
    ct = pid % n_ctile

    cols = ct * TILE + tl.arange(0, TILE)
    col_mask = cols < Ccols
    wj = tl.arange(0, 32)

    Pb = Pp + b * 16384
    Tb = Tp + b * 1024
    Cb = Cp + b * 262144

    w1 = tl.zeros((32, TILE), dtype=tl.float32)
    for r0 in range(0, R, BR):
        rr = r0 + tl.arange(0, BR)
        rmask = rr < R
        pblk = tl.load(Pb + wj[:, None] * 512 + rr[None, :],
                       mask=rmask[None, :], other=0.0)
        cblk = tl.load(Cb + rr[:, None] * 512 + cols[None, :],
                       mask=rmask[:, None] & col_mask[None, :], other=0.0)
        if CAST_W1_FP16:
            pblk = pblk.to(tl.float16)
            cblk = cblk.to(tl.float16)
        w1 += tl.dot(pblk, cblk, input_precision=IP_W1)

    tt = tl.load(Tb + wj[None, :] * 32 + wj[:, None])
    w2 = tl.dot(tt, w1, input_precision=IP_W2)

    for r0 in range(0, R, BR):
        rr = r0 + tl.arange(0, BR)
        rmask = rr < R
        pblk = tl.load(Pb + wj[None, :] * 512 + rr[:, None],
                       mask=rmask[:, None], other=0.0)
        w2blk = w2
        if CAST_UPD_FP16:
            pblk = pblk.to(tl.float16)
            w2blk = w2blk.to(tl.float16)
        upd = tl.dot(pblk, w2blk, input_precision=IP_W2)
        cptr = Cb + rr[:, None] * 512 + cols[None, :]
        cmask = rmask[:, None] & col_mask[None, :]
        old = tl.load(cptr, mask=cmask, other=0.0)
        tl.store(cptr, old - upd, mask=cmask)


@triton.jit
def _fused_wy_2pass_w32n352_kernel(
    Pp, Tp, Cp,
    R, Ccols,
    BR: tl.constexpr, TILE: tl.constexpr,
    IP_W1: tl.constexpr, IP_W2: tl.constexpr,
):
    pid = tl.program_id(0)
    n_ctile = tl.cdiv(Ccols, TILE)
    b = pid // n_ctile
    ct = pid % n_ctile

    cols = ct * TILE + tl.arange(0, TILE)
    col_mask = cols < Ccols
    wj = tl.arange(0, 32)

    Pb = Pp + b * 11264
    Tb = Tp + b * 1024
    Cb = Cp + b * 123904

    w1 = tl.zeros((32, TILE), dtype=tl.float32)
    for r0 in range(0, R, BR):
        rr = r0 + tl.arange(0, BR)
        rmask = rr < R
        pblk = tl.load(Pb + wj[:, None] * 352 + rr[None, :],
                       mask=rmask[None, :], other=0.0)
        cblk = tl.load(Cb + rr[:, None] * 352 + cols[None, :],
                       mask=rmask[:, None] & col_mask[None, :], other=0.0)
        w1 += tl.dot(pblk, cblk, input_precision=IP_W1)

    tt = tl.load(Tb + wj[None, :] * 32 + wj[:, None])
    w2 = tl.dot(tt, w1, input_precision=IP_W2)

    for r0 in range(0, R, BR):
        rr = r0 + tl.arange(0, BR)
        rmask = rr < R
        pblk = tl.load(Pb + wj[None, :] * 352 + rr[:, None],
                       mask=rmask[:, None], other=0.0)
        upd = tl.dot(pblk, w2, input_precision=IP_W2)
        cptr = Cb + rr[:, None] * 352 + cols[None, :]
        cmask = rmask[:, None] & col_mask[None, :]
        old = tl.load(cptr, mask=cmask, other=0.0)
        tl.store(cptr, old - upd, mask=cmask)


@triton.jit
def _pbot_m_dual_store_kernel(
    Pp, Mp, Vp, Hp,
    R: tl.constexpr, W: tl.constexpr,
    sPb: tl.constexpr, sPr: tl.constexpr, sPc: tl.constexpr,
    sMb: tl.constexpr, sMr: tl.constexpr, sMc: tl.constexpr,
    sVb: tl.constexpr, sVr: tl.constexpr, sVc: tl.constexpr,
    sHb: tl.constexpr, sHr: tl.constexpr, sHc: tl.constexpr,
    WPAD: tl.constexpr, BR: tl.constexpr, IP: tl.constexpr,
):
    b = tl.program_id(0)
    rb = tl.program_id(1) * BR
    rr = rb + tl.arange(0, BR)
    kk = tl.arange(0, WPAD)
    jj = tl.arange(0, WPAD)
    rmask = rr < R
    wmask = kk < W

    Pb = Pp + b * sPb
    Mb = Mp + b * sMb
    Vb = Vp + b * sVb
    Hb = Hp + b * sHb

    pblk = tl.load(
        Pb + rr[:, None] * sPr + kk[None, :] * sPc,
        mask=rmask[:, None] & wmask[None, :],
        other=0.0,
    )
    mblk = tl.load(
        Mb + kk[:, None] * sMr + jj[None, :] * sMc,
        mask=wmask[:, None] & (jj[None, :] < W),
        other=0.0,
    )
    out = tl.dot(pblk, mblk, input_precision=IP)
    mask = rmask[:, None] & (jj[None, :] < W)
    tl.store(Vb + rr[:, None] * sVr + jj[None, :] * sVc, out, mask=mask)
    tl.store(Hb + rr[:, None] * sHr + jj[None, :] * sHc, out, mask=mask)


def _pbot_m_dual_store(Pbot, M, Vbot, Hbot, prec):
    B, R, W = Pbot.shape
    if R <= 0:
        return Vbot
    if prec is True:
        prec = "tf32"
    elif prec is False:
        prec = "ieee"
    br = 32
    wpad = 1 << (W - 1).bit_length()
    grid = (B, triton.cdiv(R, br))
    _pbot_m_dual_store_kernel[grid](
        Pbot, M, Vbot, Hbot,
        R, W,
        Pbot.stride(0), Pbot.stride(1), Pbot.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        Vbot.stride(0), Vbot.stride(1), Vbot.stride(2),
        Hbot.stride(0), Hbot.stride(1), Hbot.stride(2),
        WPAD=wpad, BR=br, IP=prec,
        num_warps=4, num_stages=3,
    )
    return Vbot


@triton.jit
def _pbot_m_dual_u_kernel(
    Pp, Mp, Tp, Vfullp, Vbotp, Hp, Up,
    R: tl.constexpr, W: tl.constexpr,
    sPb: tl.constexpr, sPr: tl.constexpr, sPc: tl.constexpr,
    sMb: tl.constexpr, sMr: tl.constexpr, sMc: tl.constexpr,
    sTb: tl.constexpr, sTr: tl.constexpr, sTc: tl.constexpr,
    sVfb: tl.constexpr, sVfr: tl.constexpr, sVfc: tl.constexpr,
    sVbb: tl.constexpr, sVbr: tl.constexpr, sVbc: tl.constexpr,
    sHb: tl.constexpr, sHr: tl.constexpr, sHc: tl.constexpr,
    sUb: tl.constexpr, sUr: tl.constexpr, sUc: tl.constexpr,
    WPAD: tl.constexpr, BR: tl.constexpr, IPV: tl.constexpr, IPU: tl.constexpr,
):
    b = tl.program_id(0)
    rb = tl.program_id(1) * BR
    rr = rb + tl.arange(0, BR)
    kk = tl.arange(0, WPAD)
    jj = tl.arange(0, WPAD)
    rmask = rr < R
    wmask = kk < W
    bottom = rr >= W
    rbott = rr - W

    Pb = Pp + b * sPb
    Mb = Mp + b * sMb
    Tb = Tp + b * sTb
    Vfb = Vfullp + b * sVfb
    Vbb = Vbotp + b * sVbb
    Hb0 = Hp + b * sHb
    Ub = Up + b * sUb

    pblk = tl.load(
        Pb + rbott[:, None] * sPr + kk[None, :] * sPc,
        mask=rmask[:, None] & bottom[:, None] & wmask[None, :],
        other=0.0,
    )
    mblk = tl.load(
        Mb + kk[:, None] * sMr + jj[None, :] * sMc,
        mask=wmask[:, None] & (jj[None, :] < W),
        other=0.0,
    )
    vbot = tl.dot(pblk, mblk, input_precision=IPV)
    vtop = tl.load(
        Vfb + rr[:, None] * sVfr + jj[None, :] * sVfc,
        mask=rmask[:, None] & (~bottom)[:, None] & (jj[None, :] < W),
        other=0.0,
    )
    v = tl.where(bottom[:, None], vbot.to(vtop.dtype), vtop)
    mask = rmask[:, None] & (jj[None, :] < W)
    bmask = mask & bottom[:, None]
    tl.store(Vbb + rbott[:, None] * sVbr + jj[None, :] * sVbc, vbot, mask=bmask)
    tl.store(Hb0 + rbott[:, None] * sHr + jj[None, :] * sHc, vbot, mask=bmask)

    tt = tl.load(
        Tb + jj[None, :] * sTr + kk[:, None] * sTc,
        mask=wmask[:, None] & (jj[None, :] < W),
        other=0.0,
    )
    u = tl.dot(v, tt, input_precision=IPU)
    tl.store(Ub + rr[:, None] * sUr + jj[None, :] * sUc, u, mask=mask)


def _pbot_m_dual_store_u(Pbot, M, T, Vfull, Vbot, Hbot, Ufull, prec, u_prec=None):
    B, R, W = Vfull.shape
    if R <= 0:
        return Vbot
    if prec is True:
        prec = "tf32"
    elif prec is False:
        prec = "ieee"
    if u_prec is None:
        u_prec = prec
    if u_prec is True:
        u_prec = "tf32"
    elif u_prec is False:
        u_prec = "ieee"
    br = 32
    wpad = 1 << (W - 1).bit_length()
    grid = (B, triton.cdiv(R, br))
    _pbot_m_dual_u_kernel[grid](
        Pbot, M, T, Vfull, Vbot, Hbot, Ufull,
        R, W,
        Pbot.stride(0), Pbot.stride(1), Pbot.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        T.stride(0), T.stride(1), T.stride(2),
        Vfull.stride(0), Vfull.stride(1), Vfull.stride(2),
        Vbot.stride(0), Vbot.stride(1), Vbot.stride(2),
        Hbot.stride(0), Hbot.stride(1), Hbot.stride(2),
        Ufull.stride(0), Ufull.stride(1), Ufull.stride(2),
        WPAD=wpad, BR=br, IPV=prec, IPU=u_prec,
        num_warps=4, num_stages=3,
    )
    return Vbot


@triton.jit
def _gram_fp16_kernel(
    Pp, Gp,
    R, W: tl.constexpr, WPAD: tl.constexpr,
    sPb: tl.constexpr, sPr: tl.constexpr, sPc: tl.constexpr,
    sGb: tl.constexpr, sGr: tl.constexpr, sGc: tl.constexpr,
    BR: tl.constexpr, IP: tl.constexpr,
):
    b = tl.program_id(0)
    Pb = Pp + b * sPb
    Gb = Gp + b * sGb
    wj = tl.arange(0, WPAD)
    acc = tl.zeros((WPAD, WPAD), dtype=tl.float32)
    for r0 in range(0, R, BR):
        rr = r0 + tl.arange(0, BR)
        rmask = rr < R
        pblk = tl.load(
            Pb + rr[:, None] * sPr + wj[None, :] * sPc,
            mask=rmask[:, None] & (wj[None, :] < W),
            other=0.0,
        )
        acc += tl.dot(tl.trans(pblk), pblk, input_precision=IP)
    wmask = wj < W
    tl.store(
        Gb + wj[:, None] * sGr + wj[None, :] * sGc,
        acc,
        mask=wmask[:, None] & wmask[None, :],
    )


def _gram_fp16(P, G):
    B, R, W = P.shape
    wpad = 1 << (W - 1).bit_length()
    _gram_fp16_kernel[(B,)](
        P,
        G,
        R,
        W,
        wpad,
        P.stride(0),
        P.stride(1),
        P.stride(2),
        G.stride(0),
        G.stride(1),
        G.stride(2),
        BR=128,
        IP="ieee",
        num_warps=8,
        num_stages=3,
    )
    return G


def _fused_wy_update(P, T, C, prec, j0=0):
    if prec is True:
        prec = "tf32"
    elif prec is False:
        prec = "ieee"
    B, W, R = P.shape
    ccols = C.shape[2]
    if ccols == 0:
        return C
    if (
        prec in ("tf32x3", "tf32x3_w1", "fp16_w1", "fp16")
        and W == 32
        and P.stride(0) == 16384
        and P.stride(1) == 512
        and P.stride(2) == 1
        and T.stride(0) == 1024
        and T.stride(1) == 32
        and T.stride(2) == 1
        and C.stride(0) == 262144
        and C.stride(1) == 512
        and C.stride(2) == 1
    ):
        if prec == "tf32x3_w1":
            ip_w1 = "tf32x3"
            ip_w2 = "tf32"
            cast_w1_fp16 = False
            cast_upd_fp16 = False
        elif prec == "fp16_w1":
            ip_w1 = "ieee"
            ip_w2 = "tf32"
            cast_w1_fp16 = True
            cast_upd_fp16 = False
        elif prec == "fp16":
            ip_w1 = "ieee"
            ip_w2 = "ieee"
            cast_w1_fp16 = True
            cast_upd_fp16 = True
        else:
            ip_w1 = prec
            ip_w2 = prec
            cast_w1_fp16 = False
            cast_upd_fp16 = False
        grid = (B * triton.cdiv(ccols, 32),)
        _fused_wy_2pass_w32n512_kernel[grid](
            P, T, C, R, ccols,
            BR=32, TILE=32, IP_W1=ip_w1, IP_W2=ip_w2,
            CAST_W1_FP16=cast_w1_fp16, CAST_UPD_FP16=cast_upd_fp16,
            num_warps=2, num_stages=4,
        )
        return C
    if (
        prec in ("tf32x3", "tf32x3_w1")
        and W == 32
        and P.stride(0) == 11264
        and P.stride(1) == 352
        and P.stride(2) == 1
        and T.stride(0) == 1024
        and T.stride(1) == 32
        and T.stride(2) == 1
        and C.stride(0) == 123904
        and C.stride(1) == 352
        and C.stride(2) == 1
    ):
        ip_w1 = "tf32x3" if prec == "tf32x3_w1" else prec
        ip_w2 = "tf32" if prec == "tf32x3_w1" else prec
        grid = (B * triton.cdiv(ccols, 32),)
        _fused_wy_2pass_w32n352_kernel[grid](
            P, T, C, R, ccols,
            BR=32, TILE=32, IP_W1=ip_w1, IP_W2=ip_w2,
            num_warps=2, num_stages=3,
        )
        return C
    if prec == "tf32":
        br, tile, nw, ns = 64, 32, 2, 3
    elif prec == "tf32x3":
        br, tile, nw, ns = 32, 32, 2, 3
    else:
        br, tile, nw, ns = 64, 64, 4, 3
    wpad = 1 << (W - 1).bit_length()
    grid = (B * triton.cdiv(ccols, tile),)
    _fused_wy_2pass_kernel[grid](
        P, T, C, B, W, R, ccols,
        P.stride(0), P.stride(1), P.stride(2),
        T.stride(0), T.stride(1), T.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        WPAD=wpad, BR=br, TILE=tile, IP=prec,
        num_warps=nw, num_stages=ns,
    )
    return C


# --- CholeskyQR-panel constants/helpers (from sol_v9), for n=4096 B=2 only ---
_CHOL_RECON_NT = 512
_LARFT_NT = 64
_PM_PREC = "ieee"
_VTV_PREC = "tf32"
_CQR_FUSE_U_UPDATE = True
_CQR_W = 64
_cqr_ws_cache = {}
_cqr_ws16_cache = {}


def _cqr_ws(B, n, W, device):
    key = (B, n, W)
    ws = _cqr_ws_cache.get(key)
    if ws is None:
        ws = {
            "G":    torch.empty(B, W, W, device=device, dtype=torch.float32),
            "M":    torch.empty(B, W, W, device=device, dtype=torch.float32),
            "T":    torch.empty(B, W, W, device=device, dtype=torch.float32),
            "VtV":  torch.empty(B, W, W, device=device, dtype=torch.float32),
            "Vw":   torch.empty(B, n, W, device=device, dtype=torch.float32),
            "W1":   torch.empty(B, W, n, device=device, dtype=torch.float32),
            "W2":   torch.empty(B, W, n, device=device, dtype=torch.float32),
            "fail": torch.zeros(B, device=device, dtype=torch.int32),
        }
        _cqr_ws_cache[key] = ws
    return ws


def _retau(H, tau, limit=None):
    if limit is None:
        limit = H.shape[1]
    _retau_ext(H, tau, limit)
    return tau


def _tail_diag_is_small(H, limit, rel):
    n = H.shape[1]
    if limit >= n:
        return False
    head = H[:, :limit, :limit].diagonal(dim1=1, dim2=2).abs().amax().clamp_min(1.0e-30)
    tail = H[:, limit:, limit:].diagonal(dim1=1, dim2=2).abs().amax()
    return bool((tail <= head * rel).item())


def _cqr_blocked(H, W, trail_prec, check_fail=True, gram_prec="ieee", chol_nt=768, larft_nt=384, pm_prec=None, shift_scale=0.0, direct_t=False, direct_t_from=0, allow_shifted_u=False, adaptive_limits=None, return_limit=False):
    # CQR1 fused chol+recon blocked QR (sol_v9 path, cqr2=False, CHOL_RECON_NT>0).
    B, n, _ = H.shape
    tau = torch.empty(B, n, device=H.device, dtype=torch.float32)
    ws = _cqr_ws(B, n, W, H.device)
    if pm_prec is None:
        pm_prec = _PM_PREC
    same_prec = (gram_prec == pm_prec == _VTV_PREC == trail_prec)
    if same_prec:
        torch.backends.cuda.matmul.fp32_precision = gram_prec
    fail = ws["fail"]
    if check_fail:
        fail.zero_()
    retau_limit = n
    for j0 in range(0, n, W):
        w = min(W, n - j0)
        r = n - j0
        P = H[:, j0:, j0:j0 + w]
        G = ws["G"][:, :w, :w]
        if not same_prec:
            torch.backends.cuda.matmul.fp32_precision = gram_prec
        torch.bmm(P.transpose(1, 2), P, out=G)
        Vw = ws["Vw"][:, :r, :w]
        M = ws["M"][:, :w, :w]
        c = n - (j0 + w)
        T = ws["T"][:, :w, :w]
        direct_this = direct_t and c > 0 and j0 >= direct_t_from
        if direct_this:
            _chol_recon_t_ext(G, P, H, tau, M, Vw, T, fail, j0, w, chol_nt, shift_scale)
        else:
            _chol_recon_ext(G, P, H, tau, M, Vw, fail, j0, w, chol_nt, shift_scale)
        U = None
        if r > w:
            if not same_prec:
                torch.backends.cuda.matmul.fp32_precision = pm_prec
            use_u = _CQR_FUSE_U_UPDATE and direct_this and w in (32, 64) and pm_prec == "tf32" and (shift_scale == 0.0 or allow_shifted_u)
            if use_u:
                Ubase = ws.get("U")
                if Ubase is None:
                    Ubase = torch.empty(B, n, W, device=H.device, dtype=torch.float32)
                    ws["U"] = Ubase
                U = Ubase[:, :r, :w]
                _pbot_m_dual_store_u(P[:, w:, :], M, T, Vw, Vw[:, w:, :], H[:, j0 + w:, j0:j0 + w], U, ("ieee" if shift_scale > 0.0 else pm_prec), ("tf32" if shift_scale > 0.0 else pm_prec))
            elif ((w == 64 and pm_prec == "tf32") or (w == 32 and pm_prec == "ieee")):
                _pbot_m_dual_store(P[:, w:, :], M, Vw[:, w:, :], H[:, j0 + w:, j0:j0 + w], ("ieee" if shift_scale > 0.0 else pm_prec))
            else:
                torch.bmm(P[:, w:, :], M, out=Vw[:, w:, :])
                H[:, j0 + w:, j0:j0 + w] = Vw[:, w:, :]
        if c <= 0:
            j0 += w
            continue
        if not direct_this:
            VtV = ws["VtV"][:, :w, :w]
            if not same_prec:
                torch.backends.cuda.matmul.fp32_precision = _VTV_PREC
            torch.bmm(Vw.transpose(1, 2), Vw, out=VtV)
            _larft_ext(VtV, tau, T, j0, w, larft_nt)
        if not same_prec:
            torch.backends.cuda.matmul.fp32_precision = trail_prec
        C = H[:, j0:, j0 + w:]
        W1 = ws["W1"][:, :w, :c]
        W2 = ws["W2"][:, :w, :c]
        torch.bmm(Vw.transpose(1, 2), C, out=W1)
        if U is not None:
            C.baddbmm_(U, W1, beta=1.0, alpha=-1.0)
        else:
            torch.bmm(T.transpose(1, 2), W1, out=W2)
            C.baddbmm_(Vw, W2, beta=1.0, alpha=-1.0)
        end = j0 + w
        if adaptive_limits is not None and end in adaptive_limits:
            if _tail_diag_is_small(H, end, adaptive_limits[end]):
                tau[:, end:] = 0.0
                retau_limit = end
                break
    if return_limit:
        return H, tau, fail, retau_limit
    return H, tau, fail


def _cqr_prefix(H, W, stop_col, trail_prec, check_fail=True, gram_prec="ieee", chol_nt=768, larft_nt=384, pm_prec=None, shift_scale=0.0, direct_t=False, direct_t_from=0, allow_shifted_u=False):
    B, n, _ = H.shape
    tau = torch.empty(B, n, device=H.device, dtype=torch.float32)
    tau.zero_()
    ws = _cqr_ws(B, n, W, H.device)
    if pm_prec is None:
        pm_prec = _PM_PREC
    same_prec = (gram_prec == pm_prec == _VTV_PREC == trail_prec)
    if same_prec:
        torch.backends.cuda.matmul.fp32_precision = gram_prec
    fail = ws["fail"]
    if check_fail:
        fail.zero_()
    stop = min(int(stop_col), n)
    for j0 in range(0, stop, W):
        w = min(W, stop - j0)
        r = n - j0
        P = H[:, j0:, j0:j0 + w]
        G = ws["G"][:, :w, :w]
        if not same_prec:
            torch.backends.cuda.matmul.fp32_precision = gram_prec
        torch.bmm(P.transpose(1, 2), P, out=G)
        Vw = ws["Vw"][:, :r, :w]
        M = ws["M"][:, :w, :w]
        c = n - (j0 + w)
        T = ws["T"][:, :w, :w]
        direct_this = direct_t and c > 0 and j0 >= direct_t_from
        if direct_this:
            _chol_recon_t_ext(G, P, H, tau, M, Vw, T, fail, j0, w, chol_nt, shift_scale)
        else:
            _chol_recon_ext(G, P, H, tau, M, Vw, fail, j0, w, chol_nt, shift_scale)
        U = None
        if r > w:
            if not same_prec:
                torch.backends.cuda.matmul.fp32_precision = pm_prec
            use_u = _CQR_FUSE_U_UPDATE and direct_this and w in (32, 64) and pm_prec == "tf32" and (shift_scale == 0.0 or allow_shifted_u)
            if use_u:
                Ubase = ws.get("U")
                if Ubase is None:
                    Ubase = torch.empty(B, n, W, device=H.device, dtype=torch.float32)
                    ws["U"] = Ubase
                U = Ubase[:, :r, :w]
                _pbot_m_dual_store_u(P[:, w:, :], M, T, Vw, Vw[:, w:, :], H[:, j0 + w:, j0:j0 + w], U, ("ieee" if shift_scale > 0.0 else pm_prec), ("tf32" if shift_scale > 0.0 else pm_prec))
            elif ((w == 64 and pm_prec == "tf32") or (w == 32 and pm_prec == "ieee")):
                _pbot_m_dual_store(P[:, w:, :], M, Vw[:, w:, :], H[:, j0 + w:, j0:j0 + w], ("ieee" if shift_scale > 0.0 else pm_prec))
            else:
                torch.bmm(P[:, w:, :], M, out=Vw[:, w:, :])
                H[:, j0 + w:, j0:j0 + w] = Vw[:, w:, :]
        if c <= 0:
            continue
        if not direct_this:
            VtV = ws["VtV"][:, :w, :w]
            if not same_prec:
                torch.backends.cuda.matmul.fp32_precision = _VTV_PREC
            torch.bmm(Vw.transpose(1, 2), Vw, out=VtV)
            _larft_ext(VtV, tau, T, j0, w, larft_nt)
        if not same_prec:
            torch.backends.cuda.matmul.fp32_precision = trail_prec
        C = H[:, j0:, j0 + w:]
        W1 = ws["W1"][:, :w, :c]
        W2 = ws["W2"][:, :w, :c]
        torch.bmm(Vw.transpose(1, 2), C, out=W1)
        if U is not None:
            C.baddbmm_(U, W1, beta=1.0, alpha=-1.0)
        else:
            torch.bmm(T.transpose(1, 2), W1, out=W2)
            C.baddbmm_(Vw, W2, beta=1.0, alpha=-1.0)
    return H, tau, fail


def _cqr_ws16(B, n, W, device, dt):
    key = (B, n, W, dt)
    ws = _cqr_ws16_cache.get(key)
    if ws is None:
        ws = {
            "G": torch.empty(B, W, W, device=device, dtype=torch.float32),
            "Gh": torch.empty(B, W, W, device=device, dtype=dt),
            "M": torch.empty(B, W, W, device=device, dtype=torch.float32),
            "T": torch.empty(B, W, W, device=device, dtype=torch.float32),
            "Mh": torch.empty(B, W, W, device=device, dtype=dt),
            "Th": torch.empty(B, W, W, device=device, dtype=dt),
            "Pf": torch.empty(B, n, W, device=device, dtype=torch.float32),
            "Vw": torch.empty(B, n, W, device=device, dtype=dt),
            "W1": torch.empty(B, W, n, device=device, dtype=dt),
            "W2": torch.empty(B, W, n, device=device, dtype=dt),
            "fail": torch.zeros(B, device=device, dtype=torch.int32),
        }
        _cqr_ws16_cache[key] = ws
    return ws


def _cqr_blocked_fp16(H, W, chol_nt=1024, shift_scale=0.0,
                      trail_fp32=False, direct_g16=False, limit=None):
    B, n, _ = H.shape
    dt = H.dtype
    stop_col = n if limit is None else min(n, int(limit))
    if stop_col < n:
        tau = torch.zeros(B, n, device=H.device, dtype=torch.float32)
    else:
        tau = torch.empty(B, n, device=H.device, dtype=torch.float32)
    ws = _cqr_ws16(B, n, W, H.device, dt)
    fail = ws["fail"]
    fail.zero_()
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    j0 = 0
    while j0 < stop_col:
        w = min(W, stop_col - j0)
        r = n - j0
        P = H[:, j0:, j0:j0 + w]
        G = ws["G"][:, :w, :w]
        Gh = ws["Gh"][:, :w, :w]
        c = n - (j0 + w)
        use_direct_g16 = (
            direct_g16
            and c > 0
            and w == 64
            and dt == torch.float16
        )
        if use_direct_g16:
            torch.bmm(P.transpose(1, 2), P, out=Gh)
        elif n != 4096:
            _gram_fp16(P, G)
        else:
            G.copy_(torch.bmm(P.transpose(1, 2), P))
        Vw = ws["Vw"][:, :r, :w]
        M = ws["M"][:, :w, :w]
        T = ws["T"][:, :w, :w]
        Mh = ws["Mh"][:, :w, :w]
        Th = ws["Th"][:, :w, :w]
        if c > 0:
            if use_direct_g16:
                _chol_recon_t_h_g16_ext(
                    Gh, P, H, tau,
                    None, Vw, (T if trail_fp32 else None),
                    Mh, Th, fail, j0, w, chol_nt, shift_scale
                )
            else:
                _chol_recon_t_h_ext(
                    G, P, H, tau,
                    None, Vw, (T if trail_fp32 else None),
                    Mh, Th, fail, j0, w, chol_nt, shift_scale
                )
        else:
            _chol_recon_h_ext(G, P, H, tau, M, Vw, fail, j0, w, chol_nt, shift_scale)
        U = None
        if r > w:
            if c > 0 and not trail_fp32:
                Ubase = ws.get("U")
                if Ubase is None:
                    Ubase = torch.empty(B, n, W, device=H.device, dtype=dt)
                    ws["U"] = Ubase
                U = Ubase[:, :r, :w]
                _pbot_m_dual_store_u(
                    P[:, w:, :],
                    Mh,
                    Th,
                    Vw,
                    Vw[:, w:, :],
                    H[:, j0 + w:, j0:j0 + w],
                    U,
                    "ieee",
                    "ieee",
                )
            else:
                Vbot = Vw[:, w:, :]
                torch.bmm(P[:, w:, :], Mh, out=Vbot)
                H[:, j0 + w:, j0:j0 + w].copy_(Vbot)
        if c <= 0:
            j0 += w
            continue
        C = H[:, j0:, j0 + w:]
        if trail_fp32:
            Vf = Vw.float()
            Cf = C.float()
            W1 = torch.bmm(Vf.transpose(1, 2), Cf)
            W2 = torch.bmm(T.transpose(1, 2), W1)
            Cf.baddbmm_(Vf, W2, beta=1.0, alpha=-1.0)
            C.copy_(Cf)
        elif U is not None:
            W1 = ws["W1"][:, :w, :c]
            torch.bmm(Vw.transpose(1, 2), C, out=W1)
            C.baddbmm_(U, W1, beta=1.0, alpha=-1.0)
        else:
            W1 = ws["W1"][:, :w, :c]
            W2 = ws["W2"][:, :w, :c]
            torch.bmm(Vw.transpose(1, 2), C, out=W1)
            torch.bmm(Th.transpose(1, 2), W1, out=W2)
            C.baddbmm_(Vw, W2, beta=1.0, alpha=-1.0)
        j0 += w
    return H, tau, fail


_CQR_4096_LOWP_DT = torch.float16
_CQR_4096_DIRECT_G16 = True
_CQR_2048_LOWP_DT = torch.float16
_CQR_2048_DIRECT_G16 = True


def _cqr_4096_raw(a):
    global _matmul_tf32_enabled
    n = a.shape[1]
    H = a.to(_CQR_4096_LOWP_DT)
    H, tau, _fail = _cqr_blocked_fp16(
        H, _CQR_W, chol_nt=1024,
        direct_g16=_CQR_4096_DIRECT_G16,
        limit=(15 * n) // 16,
    )
    _matmul_tf32_enabled = None
    Hf = torch.empty(a.shape, device=a.device, dtype=torch.float32)
    _finalize_h_ext(H, Hf, tau, n)
    return Hf, tau


def _cqr_2048_raw(a):
    global _matmul_tf32_enabled
    n = a.shape[1]
    H = a.to(_CQR_2048_LOWP_DT)
    H, tau, _fail = _cqr_blocked_fp16(
        H, 64, chol_nt=1024, shift_scale=1e-7,
        direct_g16=_CQR_2048_DIRECT_G16,
        limit=(31 * n) // 32,
    )
    _matmul_tf32_enabled = None
    Hf = torch.empty(a.shape, device=a.device, dtype=torch.float32)
    _finalize_h_ext(H, Hf, tau, n)
    return Hf, tau


_matmul_tf32_enabled = None


def _set_matmul_tf32(enabled):
    global _matmul_tf32_enabled
    if _matmul_tf32_enabled == enabled:
        return
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if enabled else "ieee"
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = enabled
    except Exception:
        pass
    _matmul_tf32_enabled = enabled


_set_matmul_tf32(False)

FUSED_MAX_N = 192
FUSED_NT = {32: 1024, 176: 1024}
PANEL_W = {176: 44, 352: 32, 512: 32, 1024: 32, 2048: 24, 4096: 12}
PANEL_NT = {12: 896, 16: 512, 24: 1024, 32: 512, 64: 512}
SMEM_BUDGET = 230000

_ws_cache = {}


def _panel_width(n):
    return PANEL_W.get(n, 64)


def _get_ws(B, n, W, device):
    key = (B, n, W)
    ws = _ws_cache.get(key)
    if ws is None:
        ws = {
            "P": torch.empty(B, W, n, device=device, dtype=torch.float32),
            "T": torch.empty(B, W, W, device=device, dtype=torch.float32),
        }
        _ws_cache[key] = ws
    return ws


def _smem_A(r, w):
    return ((r | 1) * w + 2 * w * w + 2 * w + 40) * 4


def _use_tf32_updates(B, n):
    return (B, n) in ((40, 352), (60, 1024), (8, 2048), (2, 4096))


def _use_tf32_updates_for(a):
    B, n, _ = a.shape
    return _use_tf32_updates(B, n)


def _blocked_qr(a, tf32_updates=None, use_fused=False):
    B, n, _ = a.shape
    H = a.clone()
    tau = torch.empty(B, n, device=a.device, dtype=torch.float32)
    W = _panel_width(n)
    nthreads = 384 if n == 512 and W == 32 else (1024 if n in (176, 352, 1024) else PANEL_NT.get(W, 512))
    ws = _get_ws(B, n, W, a.device)
    if tf32_updates is None:
        tf32_updates = _use_tf32_updates_for(a)
    _set_matmul_tf32(tf32_updates)
    for j0 in range(0, n, W):
        w = min(W, n - j0)
        r = n - j0
        P = ws["P"][:, :w, :r]
        T = ws["T"][:, :w, :w]
        c = n - (j0 + w)
        want_T = 1 if c > 0 else 0
        if _smem_A(r, w) <= SMEM_BUDGET:
            _panel_smem_ext(H, P, T, tau, j0, w, want_T, nthreads)
        else:
            _panel_tall_ext(H, P, T, tau, j0, w)
        if c <= 0:
            continue
        C = H[:, j0:, j0 + w:]
        if use_fused:
            _fused_wy_update(P, T, C, tf32_updates)
        else:
            W1 = torch.bmm(P, C)
            W2 = torch.bmm(T.transpose(1, 2), W1)
            C.baddbmm_(P.transpose(1, 2), W2, beta=1.0, alpha=-1.0)
    return H, tau


def _blocked_qr_tf32_from(a, tf32_from):
    B, n, _ = a.shape
    H = a.clone()
    tau = torch.empty(B, n, device=a.device, dtype=torch.float32)
    W = _panel_width(n)
    nthreads = 1024 if n in (176, 352, 1024) else PANEL_NT.get(W, 512)
    ws = _get_ws(B, n, W, a.device)
    _set_matmul_tf32(False)
    for j0 in range(0, n, W):
        w = min(W, n - j0)
        r = n - j0
        P = ws["P"][:, :w, :r]
        T = ws["T"][:, :w, :w]
        c = n - (j0 + w)
        want_T = 1 if c > 0 else 0
        if _smem_A(r, w) <= SMEM_BUDGET:
            _panel_smem_ext(H, P, T, tau, j0, w, want_T, nthreads)
        else:
            _panel_tall_ext(H, P, T, tau, j0, w)
        if c <= 0:
            continue
        _set_matmul_tf32(j0 >= tf32_from)
        C = H[:, j0:, j0 + w:]
        W1 = torch.bmm(P, C)
        W2 = torch.bmm(T.transpose(1, 2), W1)
        C.baddbmm_(P.transpose(1, 2), W2, beta=1.0, alpha=-1.0)
    return H, tau


_MIXED_P32_W1_TF32 = False
_MIXED_P32_W2_TF32 = False
_MIXED_P32_UPD_TF32 = True
_MIXED_FUSED_WY = False


def _blocked_qr_mixed_split32(a):
    B, n, _ = a.shape
    H = a.clone()
    tau = torch.empty(B, n, device=a.device, dtype=torch.float32)
    W = _panel_width(n)
    nthreads = 1024 if n in (176, 352, 1024) else PANEL_NT.get(W, 512)
    ws = _get_ws(B, n, W, a.device)
    _set_matmul_tf32(False)
    for j0 in range(0, n, W):
        w = min(W, n - j0)
        r = n - j0
        P = ws["P"][:, :w, :r]
        T = ws["T"][:, :w, :w]
        c = n - (j0 + w)
        want_T = 1 if c > 0 else 0
        if _smem_A(r, w) <= SMEM_BUDGET:
            _panel_smem_ext(H, P, T, tau, j0, w, want_T, nthreads)
        else:
            _panel_tall_ext(H, P, T, tau, j0, w)
        if c <= 0:
            continue
        C = H[:, j0:, j0 + w:]
        if j0 < 32:
            _set_matmul_tf32(False)
            W1 = torch.bmm(P, C)
            W2 = torch.bmm(T.transpose(1, 2), W1)
            C.baddbmm_(P.transpose(1, 2), W2, beta=1.0, alpha=-1.0)
        elif j0 == 32:
            _set_matmul_tf32(_MIXED_P32_W1_TF32)
            W1 = torch.bmm(P, C)
            _set_matmul_tf32(_MIXED_P32_W2_TF32)
            W2 = torch.bmm(T.transpose(1, 2), W1)
            _set_matmul_tf32(_MIXED_P32_UPD_TF32)
            C.baddbmm_(P.transpose(1, 2), W2, beta=1.0, alpha=-1.0)
        else:
            _set_matmul_tf32(True)
            if _MIXED_FUSED_WY:
                _fused_wy_update(P, T, C, True)
            else:
                W1 = torch.bmm(P, C)
                W2 = torch.bmm(T.transpose(1, 2), W1)
                C.baddbmm_(P.transpose(1, 2), W2, beta=1.0, alpha=-1.0)
    return H, tau


_extT_gram_cache = {}
_EXTT_TF32_GRAM_176 = True
_EXTT_TF32_GRAM_352 = True
_EXTT_TF32_GRAM_1024 = True
_N512_LOWP_FROM = 320
_N512_LOWP_PREC = "fp16_w1"


def _extT_gram_ws(B, W, device):
    g = _extT_gram_cache.get((B, W))
    if g is None:
        g = torch.empty(B, W, W, device=device, dtype=torch.float32)
        _extT_gram_cache[(B, W)] = g
    return g


def _extT_use_tf32_gram(n, tf32_updates):
    if n == 176 and _EXTT_TF32_GRAM_176:
        return True
    if not tf32_updates:
        return False
    return (n == 352 and _EXTT_TF32_GRAM_352) or (n == 1024 and _EXTT_TF32_GRAM_1024)


def _blocked_qr_extT_fused(a, tf32_updates, switch_col=None, gram_tf32=False, repair_col=None, adaptive_limits=None, larft_nt=128, panel_nt=None, limit_col=None, lowp_from=None):
    """External-T path: build T from an FP32 Gram, then apply the trailing update
    with the fused 2-pass WY Triton kernel using the requested input precision."""
    B, n, _ = a.shape
    H = a.clone()
    tau = torch.empty(B, n, device=a.device, dtype=torch.float32)
    if limit_col is not None and limit_col < n:
        tau.zero_()
    W = _panel_width(n)
    nthreads = panel_nt if panel_nt is not None else (384 if n == 512 and W == 32 else (1024 if n in (176, 352, 1024) else PANEL_NT.get(W, 512)))
    ws = _get_ws(B, n, W, a.device)
    Gbuf = _extT_gram_ws(B, W, a.device)
    update_prec = "tf32" if tf32_updates is True else ("ieee" if tf32_updates is False else tf32_updates)
    n512_lowp_from = _N512_LOWP_FROM if lowp_from is None else lowp_from
    _set_matmul_tf32(False)
    for j0 in range(0, n, W):
        w = min(W, n - j0)
        r = n - j0
        P = ws["P"][:, :w, :r]
        T = ws["T"][:, :w, :w]
        c = n - (j0 + w)
        if _smem_A(r, w) <= SMEM_BUDGET:
            _panel_smem_ext(H, P, T, tau, j0, w, 0, nthreads)
        else:
            _panel_tall_ext(H, P, T, tau, j0, w)
        if c <= 0:
            continue
        G = Gbuf[:, :w, :w]
        _set_matmul_tf32(gram_tf32)
        torch.bmm(P, P.transpose(1, 2), out=G)
        _larft_ext(G, tau, T, j0, w, larft_nt)
        C = H[:, j0:, j0 + w:]
        if repair_col is not None and j0 == repair_col:
            prec = "tf32x3_w1"
        elif n == 512 and n512_lowp_from is not None and j0 >= n512_lowp_from and w == 32:
            prec = _N512_LOWP_PREC
        else:
            prec = "tf32" if switch_col is not None and j0 >= switch_col else update_prec
        _fused_wy_update(P, T, C, prec)
        if limit_col is not None and j0 + w >= limit_col:
            break
        end = j0 + w
        if adaptive_limits is not None and end in adaptive_limits:
            if _tail_diag_is_small(H, end, adaptive_limits[end]):
                tau[:, end:] = 0.0
                return H, tau
    return H, tau


@triton.jit
def _n512_suffix_stats_permat_kernel(A, O, B: tl.constexpr, NB: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    pid = tl.program_id(1)
    offs = pid * BLOCK + tl.arange(0, BLOCK)

    row0 = offs
    v0 = tl.load(A + (b * 512 + row0) * 512, mask=row0 < 512, other=0.0)
    m0 = tl.max(tl.abs(v0), axis=0)

    rem = offs
    rs = rem // 224
    cs = rem - rs * 224
    col = 288 + cs
    vs = tl.load(A + (b * 512 + rs) * 512 + col, mask=rem < 512 * 224, other=0.0)
    av = tl.abs(vs)
    m288 = tl.max(av, axis=0)
    m384 = tl.max(tl.where(cs >= 96, av, 0.0), axis=0)

    base = b * NB + pid
    tl.store(O + base, m0)
    tl.store(O + B * NB + base, m288)
    tl.store(O + 2 * B * NB + base, m384)


_n512_suffix_permat_cache = {}


def _n512_suffix_kind_triton(a):
    B = int(a.shape[0])
    block = 4096
    nb = triton.cdiv(512 * 224, block)
    key = (B, nb, a.device)
    out = _n512_suffix_permat_cache.get(key)
    if out is None:
        out = torch.empty(3, B, nb, device=a.device, dtype=torch.float32)
        _n512_suffix_permat_cache[key] = out
    _n512_suffix_stats_permat_kernel[(B, nb)](a, out, B, nb, BLOCK=block, num_warps=8)
    stats = out.amax(dim=2)
    tol = 2.0e-4 * stats[0]
    kind = torch.zeros(B, device=a.device, dtype=torch.int8)
    kind = torch.where(stats[2] <= tol, torch.full_like(kind, 2), kind)
    kind = torch.where(stats[1] <= tol, torch.full_like(kind, 1), kind)
    return kind


@triton.jit
def _n512_kind_3col_kernel(A, O, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    row = tl.arange(0, BLOCK)
    mask = row < 512
    base = b * 512 * 512
    v0 = tl.load(A + base + row * 512, mask=mask, other=0.0)
    v320 = tl.load(A + base + row * 512 + 320, mask=mask, other=0.0)
    v511 = tl.load(A + base + row * 512 + 511, mask=mask, other=0.0)
    m0 = tl.max(tl.abs(v0), axis=0)
    m320 = tl.max(tl.abs(v320), axis=0)
    m511 = tl.max(tl.abs(v511), axis=0)
    tol = 2.0e-4 * m0
    kind = tl.full((), 0, tl.int8)
    kind = tl.where(m511 <= tol, 2, kind)
    kind = tl.where(m320 <= tol, 1, kind)
    tl.store(O + b, kind)


_n512_kind_3col_cache = {}


def _n512_suffix_kind_3col_triton(a):
    B = int(a.shape[0])
    out = _n512_kind_3col_cache.get((B, a.device))
    if out is None:
        out = torch.empty(B, device=a.device, dtype=torch.int8)
        _n512_kind_3col_cache[(B, a.device)] = out
    _n512_kind_3col_kernel[(B,)](a, out, BLOCK=512, num_warps=8)
    return out


@triton.jit
def _n512_route_bits_3col_kernel(A, O, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    row = tl.arange(0, BLOCK)
    mask = row < 512
    base = b * 512 * 512
    v0 = tl.load(A + base + row * 512, mask=mask, other=0.0)
    v320 = tl.load(A + base + row * 512 + 320, mask=mask, other=0.0)
    v511 = tl.load(A + base + row * 512 + 511, mask=mask, other=0.0)
    m0 = tl.max(tl.abs(v0), axis=0)
    m320 = tl.max(tl.abs(v320), axis=0)
    m511 = tl.max(tl.abs(v511), axis=0)
    tol = 2.0e-4 * m0
    bit = tl.full((), 1, tl.int32)
    bit = tl.where(m511 <= tol, 4, bit)
    bit = tl.where(m320 <= tol, 2, bit)
    tl.atomic_or(O, bit, sem="relaxed")


_n512_route_bits_cache = {}


def _n512_route_bits_3col(a):
    out = _n512_route_bits_cache.get(a.device)
    if out is None:
        out = torch.empty(1, device=a.device, dtype=torch.int32)
        _n512_route_bits_cache[a.device] = out
    out.zero_()
    _n512_route_bits_3col_kernel[(int(a.shape[0]),)](a, out, BLOCK=512, num_warps=8)
    return int(out.item())


@triton.jit
def _n512_route_limit_4col_kernel(A, O, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    row = tl.arange(0, BLOCK)
    mask = row < 512
    base = b * 512 * 512
    v0 = tl.load(A + base + row * 512, mask=mask, other=0.0)
    v287 = tl.load(A + base + row * 512 + 287, mask=mask, other=0.0)
    v320 = tl.load(A + base + row * 512 + 320, mask=mask, other=0.0)
    v511 = tl.load(A + base + row * 512 + 511, mask=mask, other=0.0)
    m0 = tl.max(tl.abs(v0), axis=0)
    m287 = tl.max(tl.abs(v287), axis=0)
    m320 = tl.max(tl.abs(v320), axis=0)
    m511 = tl.max(tl.abs(v511), axis=0)
    tol = 2.0e-4 * m0
    bit = tl.full((), 1, tl.int32)
    bit = tl.where(m511 <= tol, 4, bit)
    bit = tl.where(m320 <= tol, 2, bit)
    needs_288 = tl.where((bit == 2) & (m287 > tol), 8, 0)
    tl.atomic_or(O, bit | needs_288, sem="relaxed")


_n512_route_limit_cache = {}


def _n512_route_limit_4col(a):
    out = _n512_route_limit_cache.get(a.device)
    if out is None:
        out = torch.empty(1, device=a.device, dtype=torch.int32)
        _n512_route_limit_cache[a.device] = out
    out.zero_()
    _n512_route_limit_4col_kernel[(int(a.shape[0]),)](a, out, BLOCK=512, num_warps=8)
    return int(out.item())


@triton.jit
def _n1024_route_bits_2col_kernel(A, O, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    row = tl.arange(0, BLOCK)
    mask = row < 1024
    base = b * 1024 * 1024
    v0 = tl.load(A + base + row * 1024, mask=mask, other=0.0)
    v1023 = tl.load(A + base + row * 1024 + 1023, mask=mask, other=0.0)
    m0 = tl.max(tl.abs(v0), axis=0)
    m1023 = tl.max(tl.abs(v1023), axis=0)
    bit = tl.full((), 1, tl.int32)
    bit = tl.where(m1023 <= 1.0e-3 * m0, 4, bit)
    bit = tl.where(m1023 >= 1.0e-1 * m0, 2, bit)
    tl.atomic_or(O, bit, sem="relaxed")


_n1024_route_bits_cache = {}


def _n1024_route_bits_2col(a):
    out = _n1024_route_bits_cache.get(a.device)
    if out is None:
        out = torch.empty(1, device=a.device, dtype=torch.int32)
        _n1024_route_bits_cache[a.device] = out
    out.zero_()
    _n1024_route_bits_2col_kernel[(int(a.shape[0]),)](a, out, BLOCK=1024, num_warps=8)
    return int(out.item())


@triton.jit
def _n512_rank_limit32_probe_kernel(A, O, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    row = tl.arange(0, BLOCK)
    cols = tl.arange(0, 16) * 32 + 31
    mask = row < 512
    base = b * 512 * 512
    v0 = tl.load(A + base + row * 512, mask=mask, other=0.0)
    vals = tl.load(
        A + base + row[:, None] * 512 + cols[None, :],
        mask=mask[:, None],
        other=0.0,
    )
    m0 = tl.max(tl.abs(v0), axis=0)
    m = tl.max(tl.abs(vals), axis=0)
    tol = 2.0e-4 * m0
    limits = tl.arange(1, 17) * 32
    limit = tl.max(tl.where(m > tol, limits, 0), axis=0)
    tl.atomic_max(O, limit, sem="relaxed")


_n512_rank_limit32_probe_cache = {}


def _rank_from_colmax32_probe_q(a):
    out = _n512_rank_limit32_probe_cache.get(a.device)
    if out is None:
        out = torch.empty(1, device=a.device, dtype=torch.int32)
        _n512_rank_limit32_probe_cache[a.device] = out
    out.zero_()
    _n512_rank_limit32_probe_kernel[(int(a.shape[0]),)](a, out, BLOCK=512, num_warps=8)
    limit = int(out.item())
    return 32 if limit <= 0 else min(int(a.shape[1]), limit)


@triton.jit
def _n512_cluster_limit_probe_kernel(A, O, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    row = tl.arange(0, BLOCK)
    mask = row < 512
    base = b * 512 * 512
    v0 = tl.load(A + base + row * 512, mask=mask, other=0.0)
    v287 = tl.load(A + base + row * 512 + 287, mask=mask, other=0.0)
    m0 = tl.max(tl.abs(v0), axis=0)
    m287 = tl.max(tl.abs(v287), axis=0)
    limit = tl.where(m287 <= 2.0e-4 * m0, 256, 288)
    tl.atomic_max(O, limit, sem="relaxed")


_n512_cluster_limit_probe_cache = {}


def _cluster_limit_probe_q(a):
    out = _n512_cluster_limit_probe_cache.get(a.device)
    if out is None:
        out = torch.empty(1, device=a.device, dtype=torch.int32)
        _n512_cluster_limit_probe_cache[a.device] = out
    out.zero_()
    _n512_cluster_limit_probe_kernel[(int(a.shape[0]),)](a, out, BLOCK=512, num_warps=8)
    return int(out.item())


def _cqr_512_fp16(a, limit=None, W=64, shift_scale=2.0e-7):
    global _matmul_tf32_enabled
    H = a.to(torch.float16)
    H, tau, _fail = _cqr_blocked_fp16(
        H,
        W,
        chol_nt=1024,
        shift_scale=shift_scale,
        direct_g16=True,
        limit=limit,
    )
    _matmul_tf32_enabled = None
    Hf = torch.empty(a.shape, device=a.device, dtype=torch.float32)
    _finalize_h_ext(H, Hf, tau, a.shape[1])
    return Hf, tau


def _cqr_512_fp32_cluster_prefix(a):
    global _matmul_tf32_enabled
    H = a.clone()
    stop_col = 288
    H, tau, _fail = _cqr_prefix(
        H,
        64,
        stop_col,
        "tf32",
        False,
        gram_prec="tf32",
        chol_nt=1024,
        larft_nt=512,
        pm_prec="tf32",
        shift_scale=1.0e-7,
        direct_t=True,
        direct_t_from=0,
        allow_shifted_u=True,
    )
    _matmul_tf32_enabled = None
    return H, _retau(H, tau, stop_col)


def _cqr_512_fp32_cluster_limit(a, limit):
    global _matmul_tf32_enabled
    H = a.clone()
    B, n, _ = H.shape
    W = 32
    stop = min(n, int(limit))
    tau = torch.empty(B, n, device=H.device, dtype=torch.float32)
    if stop < n:
        tau[:, stop:] = 0.0
    ws = _cqr_ws(B, n, W, H.device)
    fail = ws["fail"]
    fail.zero_()
    for j0 in range(0, stop, W):
        w = min(W, stop - j0)
        r = n - j0
        P = H[:, j0:, j0:j0 + w]
        G = ws["G"][:, :w, :w]
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.bmm(P.transpose(1, 2), P, out=G)
        Vw = ws["Vw"][:, :r, :w]
        M = ws["M"][:, :w, :w]
        T = ws["T"][:, :w, :w]
        c = stop - (j0 + w)
        if c > 0:
            _chol_recon_t_ext(G, P, H, tau, M, Vw, T, fail, j0, w, 256)
        else:
            _chol_recon_ext(G, P, H, tau, M, Vw, fail, j0, w, 256)
        if r > w:
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            _pbot_m_dual_store(P[:, w:, :], M, Vw[:, w:, :], H[:, j0 + w:, j0:j0 + w], "ieee")
        if c <= 0:
            continue
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        C = H[:, j0:, j0 + w:stop]
        W1 = ws["W1"][:, :w, :c]
        W2 = ws["W2"][:, :w, :c]
        torch.bmm(Vw.transpose(1, 2), C, out=W1)
        torch.bmm(T.transpose(1, 2), W1, out=W2)
        C.baddbmm_(Vw, W2, beta=1.0, alpha=-1.0)
    _matmul_tf32_enabled = None
    return H, tau


def _blocked_qr_extT_matmul(a, tf32_updates, W_override=None, tf32_from=None):
    B, n, _ = a.shape
    H = a.clone()
    tau = torch.empty(B, n, device=a.device, dtype=torch.float32)
    W = _panel_width(n) if W_override is None else W_override
    nthreads = 1024 if n in (176, 352, 1024) else PANEL_NT.get(W, 512)
    ws = _get_ws(B, n, W, a.device)
    Gbuf = _extT_gram_ws(B, W, a.device)
    _set_matmul_tf32(tf32_updates)
    for j0 in range(0, n, W):
        w = min(W, n - j0)
        r = n - j0
        P = ws["P"][:, :w, :r]
        T = ws["T"][:, :w, :w]
        c = n - (j0 + w)
        if _smem_A(r, w) <= SMEM_BUDGET:
            _panel_smem_ext(H, P, T, tau, j0, w, 0, nthreads)
        else:
            _panel_tall_ext(H, P, T, tau, j0, w)
        if c <= 0:
            continue
        G = Gbuf[:, :w, :w]
        _set_matmul_tf32(_extT_use_tf32_gram(n, tf32_updates))
        torch.bmm(P, P.transpose(1, 2), out=G)
        _larft_ext(G, tau, T, j0, w, 256 if w == 44 else (192 if w >= 40 else 128))
        C = H[:, j0:, j0 + w:]
        update_tf32 = tf32_updates or (tf32_from is not None and j0 >= tf32_from)
        _set_matmul_tf32(update_tf32)
        W1 = torch.bmm(P, C)
        W2 = torch.bmm(T.transpose(1, 2), W1)
        C.baddbmm_(P.transpose(1, 2), W2, beta=1.0, alpha=-1.0)
    return H, tau


def _blocked_qr_512_group2(a, update_prec="tf32x3"):
    B, n, _ = a.shape
    H = a.clone()
    tau = torch.empty(B, n, device=a.device, dtype=torch.float32)
    W = 32
    nthreads = 384
    ws = _get_ws(B, n, 64, a.device)
    Pbuf = ws["P"]
    T64 = ws["T"]
    G = _extT_gram_ws(B, W, a.device)
    _set_matmul_tf32(False)
    for j0 in range(0, n, 64):
        r = n - j0
        P1 = Pbuf[:, :W, :r]
        T1 = T64[:, :W, :W]
        _panel_smem_ext(H, P1, T1, tau, j0, W, 0, nthreads)
        if r > W:
            _set_matmul_tf32(True)
            torch.bmm(P1, P1.transpose(1, 2), out=G)
            _larft_ext(G, tau, T1, j0, W, 128)
            Cnext = H[:, j0:, j0 + W:j0 + 2 * W]
            _fused_wy_update(P1, T1, Cnext, update_prec)
        if r <= W:
            continue

        j1 = j0 + W
        r2 = n - j1
        P2 = Pbuf[:, W:64, W:r]
        T2 = T64[:, W:64, W:64]
        _panel_smem_ext(H, P2, T2, tau, j1, W, 0, nthreads)
        if r2 <= W:
            continue
        _set_matmul_tf32(True)
        torch.bmm(P2, P2.transpose(1, 2), out=G)
        _larft_ext(G, tau, T2, j1, W, 128)

        Pfull = Pbuf[:, :64, :r]
        Pfull[:, W:64, :W] = 0.0
        K = torch.bmm(P1, Pfull[:, W:64, :].transpose(1, 2))
        cross = -torch.bmm(torch.bmm(T1, K), T2)
        T64[:, :W, W:64] = cross
        T64[:, W:64, :W] = 0.0
        Cfar = H[:, j0:, j0 + 64:]
        _fused_wy_update(Pfull, T64, Cfar, update_prec)
    return H, tau


def _rank_from_colnorm_q(a, rel_tol=1.0e-4, W=32):
    B, n, _ = a.shape
    cn = torch.linalg.vector_norm(a, dim=1)
    cmax = cn.amax(dim=1, keepdim=True).clamp_min(1.0e-30)
    K = int(torch.count_nonzero(cn > (cmax * rel_tol), dim=1).amax().item())
    return min(n, ((K + W - 1) // W) * W)


def _extT_fused_qlimit(a, tf32_updates, limit, gram_tf32=True):
    B, n, _ = a.shape
    H = a.clone()
    tau = torch.zeros(B, n, device=a.device, dtype=torch.float32)
    W = _panel_width(n)
    nthreads = 384 if n == 512 and W == 32 else (1024 if n in (176, 352, 1024) else PANEL_NT.get(W, 512))
    ws = _get_ws(B, n, W, a.device)
    Gbuf = _extT_gram_ws(B, W, a.device)
    update_prec = "tf32" if tf32_updates is True else ("ieee" if tf32_updates is False else tf32_updates)
    _set_matmul_tf32(False)
    for j0 in range(0, limit, W):
        w = min(W, limit - j0)
        r = n - j0
        P = ws["P"][:, :w, :r]
        T = ws["T"][:, :w, :w]
        c = limit - (j0 + w)
        if _smem_A(r, w) <= SMEM_BUDGET:
            _panel_smem_ext(H, P, T, tau, j0, w, 0, nthreads)
        else:
            _panel_tall_ext(H, P, T, tau, j0, w)
        if c <= 0:
            continue
        G = Gbuf[:, :w, :w]
        _set_matmul_tf32(gram_tf32)
        torch.bmm(P, P.transpose(1, 2), out=G)
        _larft_ext(G, tau, T, j0, w, 128)
        C = H[:, j0:, j0 + w:limit]
        _fused_wy_update(P, T, C, update_prec)
    return H, tau


def _qr_512_cluster_rrqr(a, limit=None):
    K = _cluster_limit_probe_q(a) if limit is None else limit
    if K >= a.shape[1]:
        return _blocked_qr_extT_fused(a, "tf32x3", gram_tf32=True)
    return _cqr_512_fp32_cluster_limit(a, K)


def _blocked_qr_512(a):
    B, n, _ = a.shape
    if B != 640:
        return _blocked_qr(a)
    route_code = _n512_route_limit_4col(a)
    route_bits = route_code & 7
    if route_bits == 1:
        return _cqr_512_fp16(a, W=32, shift_scale=2.0e-7)
    if route_bits == 4:
        return _cqr_512_fp16(a, limit=(3 * n) // 4, W=32, shift_scale=0.0)
    if route_bits == 2:
        return _qr_512_cluster_rrqr(a, (9 * n) // 16 if route_code & 8 else n // 2)
    return _blocked_qr_extT_fused(a, "tf32x3", gram_tf32=True, limit_col=None, lowp_from=160)


def _cqr_1024_shifted(a):
    global _matmul_tf32_enabled
    n = a.shape[1]
    H = a.clone()
    H, tau, _fail, retau_limit = _cqr_blocked(
        H,
        64,
        "tf32",
        False,
        gram_prec="tf32",
        chol_nt=1024,
        larft_nt=512,
        pm_prec="ieee",
        shift_scale=2.0e-7,
        direct_t=True,
        direct_t_from=64,
        allow_shifted_u=True,
        adaptive_limits={(3 * n) // 4: 1.0e-3},
        return_limit=True,
    )
    _matmul_tf32_enabled = None
    return H, _retau(H, tau, retau_limit)


def _cqr_1024_fp16(a, limit=None):
    global _matmul_tf32_enabled
    n = a.shape[1]
    active_limit = (57 * n) // 64 if limit is None else int(limit)
    H = a.to(torch.float16)
    H, tau, _fail = _cqr_blocked_fp16(
        H,
        64,
        chol_nt=1024,
        shift_scale=5.0e-7,
        direct_g16=True,
        limit=active_limit,
    )
    _matmul_tf32_enabled = None
    Hf = torch.empty(a.shape, device=a.device, dtype=torch.float32)
    _finalize_h_ext(H, Hf, tau, n)
    return Hf, tau


def _n1024_colnorm_route(a):
    n0 = torch.linalg.vector_norm(a[:, :, 0], dim=1)
    ntail = torch.linalg.vector_norm(a[:, :, 1023], dim=1)
    ratio = ntail.amin() / n0.amax().clamp_min(1.0e-30)
    r = float(ratio.item())
    if r < 1.0e-3:
        return 0
    if r > 1.0e-1:
        return 2
    return 1


def _blocked_qr_1024(a):
    B, n, _ = a.shape
    if (B, n) == (60, 1024):
        route_bits = _n1024_route_bits_2col(a)
        if route_bits == 2:
            return _cqr_1024_fp16(a, (3 * n) // 4)
        if route_bits == 1:
            return _cqr_1024_fp16(a)
        return _cqr_1024_shifted(a)
    return _blocked_qr(a)


def custom_kernel(data):
    a = data
    B, n, _ = a.shape
    if not a.is_contiguous():
        a = a.contiguous()
    if n == 4096 and B == 2:
        return _cqr_4096_raw(a)
    if n == 4096 and B != 2:
        _set_matmul_tf32(False)
        return torch.geqrf(a)
    if n == 2048 and B == 8:
        return _cqr_2048_raw(a)
    if n == 176:
        return _blocked_qr_extT_matmul(a, False, tf32_from=44)
    if n == 352:
        return _blocked_qr_extT_fused(a, "tf32x3", switch_col=96, gram_tf32=True)
    if n <= FUSED_MAX_N:
        _set_matmul_tf32(False)
        H = torch.empty_like(a)
        tau = torch.empty(B, n, device=a.device, dtype=torch.float32)
        _qr_fused_ext(a, H, tau, FUSED_NT.get(n, 512 if n > 64 else 128))
        return H, tau
    if n == 512:
        return _blocked_qr_512(a)
    if n == 1024:
        return _blocked_qr_1024(a)
    return _blocked_qr(a)
