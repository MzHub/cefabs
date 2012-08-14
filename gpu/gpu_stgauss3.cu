//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2012 Computer Graphics Systems Group at the
// Hasso-Plattner-Institut, Potsdam, Germany <www.hpi3d.de>
//
// Permission to use, copy, modify, and/or distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
//
#include "gpu_stgauss3.h"
#include "gpu_sampler.h"
#include "gpu_st.h"


static texture<float4, 2, cudaReadModeElementType> s_texSRC;
inline __host__ __device__ texture<float4,2>& texSRC() { return s_texSRC; }

static texture<float4, 2, cudaReadModeElementType> s_texST;
inline __host__ __device__ texture<float4,2>& texST() { return s_texST; }

#if 0
inline __device__
float3 st_integrate_rk2g( float2 p0, float sigma,
                          unsigned w, unsigned h,
                          float step_size, bool adaptive )
{
    float radius = 2 * sigma;
    float twoSigma2 = 2 * sigma * sigma;

    float3 c = make_float3(tex2D(s_texSRC, p0.x, p0.y));
    float sum = 1;

    float2 v0 = st_minor_ev(tex2D(s_texST, p0.x, p0.y));
    float sign = -1;
    float dr = (radius + 0.5f * step_size) / CUDART_PI_F;
    do {
        float2 v = v0 * sign;
        float2 p = p0;
        float u = 0;

        for (;;) {
            float2 t = st_minor_ev(tex2D(s_texST, p.x, p.y));
            if (dot(v, t) < 0) t = -t;

            float2 ph = p + 0.5f * step_size * t;
            t = st_minor_ev(tex2D(s_texST, ph.x, ph.y));
            float vt = dot(v, t);
            if (vt < 0) {
                t = -t;
                vt = -vt;
            }

            v = t;
            p += step_size * t;

            if (adaptive) {
                float rstep = dr * acosf(fminf(vt,1));
                u += fmaxf(rstep, step_size);
            } else {
                u += step_size;
            }

            if ((u >= radius) || (p.x < 0) || (p.x >= w) ||
                (p.y < 0) || (p.y >= h)) break;

            float k = __expf(-u * u / twoSigma2);
            c += k * make_float3(tex2D(s_texSRC, p.x, p.y));
            sum += k;
        }

        sign *= -1;
    } while (sign > 0);
    return c / sum;
}
#endif


struct stgauss3_filter {
     __device__ stgauss3_filter( float sigma ) {
        radius_ = 2 * sigma;
        twoSigma2_ = 2 * sigma * sigma;
        c_ = make_float3(0);
        w_ = 0;
    }

    __device__ float radius() const {
        return radius_;
    }

    __device__ void operator()(float u, float2 p) {
        float k = __expf(-u * u / twoSigma2_);
        c_ += k * make_float3(tex2D(s_texSRC, p.x, p.y));
        w_ += k;
    }

    float radius_;
    float twoSigma2_;
    float3 c_;
    float w_;
};


template<int order, typename SRC, typename ST>
__global__ void imp_stgauss3_filter( gpu_plm2<float4> dst, SRC src, ST st, float sigma,
                                     float step_size, bool adaptive )
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float2 p0 = make_float2(ix + 0.5f, iy + 0.5f);
    stgauss3_filter f(sigma);
    if (order == 2) {
        st_integrate_rk2(p0, st, f, dst.w, dst.h, step_size, adaptive);
    } else {
        st_integrate_euler(p0, st, f, dst.w, dst.h, step_size, adaptive);
        //dst(ix,iy) = make_float4(st_integrate_rk2g(p0, sigma, dst.w, dst.h, step_size, adaptive), 1);
        //return;
    }
    dst(ix, iy) = make_float4(f.c_ / f.w_, 1);
}


gpu_image<float4> gpu_stgauss3_filter( const gpu_image<float4>& src, const gpu_image<float4>& st,
                                       float sigma, bool src_linear, bool st_linear,
                                       int order, float step_size, bool adaptive )
{
    assert(src.size() == st.size());
    if (sigma <= 0) return src;
    gpu_image<float4> dst(src.size());

    gpu_sampler<float4, texSRC> src_sampler(src, src_linear? cudaFilterModeLinear : cudaFilterModePoint);
    gpu_sampler<float4, texST> st_sampler(st, st_linear? cudaFilterModeLinear : cudaFilterModePoint);

    if (order == 2)
        imp_stgauss3_filter<2><<<dst.blocks(), dst.threads()>>>(dst, src_sampler, st_sampler, sigma, step_size, adaptive);
    else
        imp_stgauss3_filter<1><<<dst.blocks(), dst.threads()>>>(dst, src_sampler, st_sampler, sigma, step_size, adaptive);

    GPU_CHECK_ERROR();
    return dst;
}
