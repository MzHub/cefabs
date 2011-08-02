//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2011 Computer Graphics Systems Group at the
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
#include "gpu_ivacef.h"
#include "gpu_color.h"
#include "gpu_st.h"
#include "gpu_gauss.h"
#include "gpu_stgauss2.h"


static texture<float, 2, cudaReadModeElementType> texSRC1;
static texture<float4, 2, cudaReadModeElementType> texSRC4;


__global__ void imp_st_sobel( const gpu_plm2<float4> prev, gpu_plm2<float4> dst, float threshold2 ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float3 u = (
          -0.183f * make_float3(tex2D(texSRC4, ix-1, iy-1)) +
          -0.634f * make_float3(tex2D(texSRC4, ix-1, iy)) + 
          -0.183f * make_float3(tex2D(texSRC4, ix-1, iy+1)) +
          +0.183f * make_float3(tex2D(texSRC4, ix+1, iy-1)) +
          +0.634f * make_float3(tex2D(texSRC4, ix+1, iy)) + 
          +0.183f * make_float3(tex2D(texSRC4, ix+1, iy+1))
          ) * 0.5f;

    float3 v = (
          -0.183f * make_float3(tex2D(texSRC4, ix-1, iy-1)) + 
          -0.634f * make_float3(tex2D(texSRC4, ix,   iy-1)) + 
          -0.183f * make_float3(tex2D(texSRC4, ix+1, iy-1)) +
          +0.183f * make_float3(tex2D(texSRC4, ix-1, iy+1)) +
          +0.634f * make_float3(tex2D(texSRC4, ix,   iy+1)) + 
          +0.183f * make_float3(tex2D(texSRC4, ix+1, iy+1))
          ) * 0.5f;
    
    float3 g = make_float3(dot(u, u), dot(v, v), dot(u, v));

    float mag = g.x * g.x + g.y * g.y + 2 * g.z * g.z;
    if (mag < threshold2) {
        if (prev.ptr) {
            dst(ix, iy) = prev(ix, iy);
        } else {
            dst(ix, iy) = make_float4( g, 0 );
        }
    } else {
        dst(ix, iy) = make_float4( g, mag );
    }
}


__global__ void imp_jacobi_step( gpu_plm2<float4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h) 
        return;

    float4 c = tex2D(texSRC4, ix, iy);
    float3 o;
    if (c.w > 0) {
        o = make_float3(c);
    } else {
        o = 0.25f * (
           make_float3(tex2D(texSRC4, ix+1, iy)) +
           make_float3(tex2D(texSRC4, ix-1, iy)) + 
           make_float3(tex2D(texSRC4, ix, iy+1)) +
           make_float3(tex2D(texSRC4, ix, iy-1))
        );
    }

    dst(ix, iy) = make_float4( o, c.w );
}                          


__global__ void imp_relax_down( const gpu_plm2<float4> src, gpu_plm2<float4> dst) {
    int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float4 sum = make_float4(0);
    {
        float4 c = src(2*ix, 2*iy);
        if (c.w > 0) sum += make_float4(make_float3(c), 1);
    }

    if (2*ix+1 < src.w) {
        float4 c = src(2*ix+1, 2*iy);
        if (c.w > 0) sum += make_float4(make_float3(c), 1);
    }

    if (2*iy+1 < src.h) {
        float4 c = src(2*ix, 2*iy+1);
        if (c.w > 0) sum += make_float4(make_float3(c), 1);

        if (2*ix+1 < src.w) {
            float4 c = src(2*ix+1, 2*iy+1);
            if (c.w > 0) sum += make_float4(make_float3(c), 1);
        }
    }
    
    if (sum.w > 0) {
        dst(ix, iy) = make_float4(make_float3(sum) / sum.w, 1);
    } else {
        dst(ix, iy) = make_float4(0);
    }
}


__global__ void imp_relax_up( const gpu_plm2<float4> src0, const gpu_plm2<float4> src1, gpu_plm2<float4> dst) {
    int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float4 c = src0(ix, iy);
    if (c.w == 0) {
        c = make_float4(make_float3(src1(ix/2, iy/2)), 0);
    }
    dst(ix, iy) = c;
}


static gpu_image<float4> jacobi_step(const gpu_image<float4>& src) {
    gpu_image<float4> dst(src.size());
    bind(&texSRC4, src);
    imp_jacobi_step<<<dst.blocks(), dst.threads()>>>(dst);
    GPU_CHECK_ERROR();
    return dst;
}


static gpu_image<float4> relax_down( const gpu_image<float4>& src ) {
    gpu_image<float4> dst((src.w()+1)/2, (src.h()+1)/2);
    imp_relax_down<<<dst.blocks(), dst.threads()>>>(src, dst);
    GPU_CHECK_ERROR();
    return dst;
}


static gpu_image<float4> relax_up( const gpu_image<float4>& src0, const gpu_image<float4>& src1 ) {
    gpu_image<float4> dst(src0.size());
    imp_relax_up<<<dst.blocks(), dst.threads()>>>(src0, src1, dst);
    GPU_CHECK_ERROR();
    return dst;
}


static gpu_image<float4> relax( const gpu_image<float4>& st ) {
    if ((st.w() <= 1) || (st.h() <= 1))
        return st;
    gpu_image<float4> tmp;
    tmp = relax_down(st);
    tmp = relax(tmp);
    tmp = relax_up(st, tmp);
    tmp = jacobi_step(tmp);
    return tmp;
}


static gpu_image<float4> st_sobel( const gpu_image<float4>& src, const gpu_image<float4>& prev, float threshold ) {
    gpu_image<float4> dst(src.size());
    bind(&texSRC4, src);
    imp_st_sobel<<<src.blocks(), src.threads()>>>(prev, dst, threshold*threshold);
    GPU_CHECK_ERROR();
    if (!prev.is_valid()) {
        dst = relax(dst);
    }
    return dst;
}


struct minmax_gray_t {
    float max_sum;
    float min_sum;
    float3 max_val;
    float3 min_val;
    float gray_sum;
    float gray_N;

    __device__ void init(float3 c) {
        float sum = 0.299f * abs(c.z) + 0.587f * abs(c.y) + 0.114f * abs(c.x);
        max_val = min_val = c;
        max_sum = min_sum = gray_sum = sum;
        gray_N = 1;
    }

    __device__ void add(float3 c) {
        float sum = 0.299f * abs(c.z) + 0.587f * abs(c.y) + 0.114f * abs(c.x);
        gray_sum += sum;
        gray_N += 1;
        if (sum > max_sum) { 
            max_sum = sum;
            max_val = c;
        }
        if (sum < min_sum) { 
            min_sum = sum;
            min_val = c;
        }
    }

    __device__ float gray_mean() { return gray_sum / gray_N; } 
};


__global__ void imp_gradient_shock( const gpu_plm2<float4> tfm, gpu_plm2<float4> dst, float sigma, float tau, float radius ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;
    const float PI = 3.14159265358979323846f;

    float4 t = tfm(ix, iy);
    float2 n = make_float2(t.y, -t.x);
    float2 nabs = fabs(n);

    float sign;
    {
        float sigma2 = sigma * sigma;
        float twoSigma2 = 2 * sigma2;

        float ds = 1.0f / ((nabs.x > nabs.y)? nabs.x : nabs.y);

        float sum = -sigma2 * tex2D(texSRC1, ix + 0.5f, iy + 0.5f);

        float halfWidth = 5 * sigma;
        for( float d = ds; d <= halfWidth; d += ds ) {
            float k = (d*d - sigma2) * __expf( -d*d / twoSigma2 ); 
            
            float2 o = d*n;
            float c = tex2D( texSRC1, 0.5f + ix - o.x, 0.5f + iy - o.y) + tex2D( texSRC1, 0.5f + ix + o.x, 0.5f + iy + o.y);
            sum += k * c;
        }

        sum = -sum / (sqrtf(2*PI) /** sigma2*/ * sigma2 * sigma);
        if (fabs(sum) < tau) sum = 0;
        sign = sum;
    }

    minmax_gray_t mm;
    float3 c0 = make_float3(tex2D( texSRC4, ix + 0.5f, iy + 0.5f));
    mm.init(c0);
    if (dot(n,n) > 0) {
        float ds;
        float2 dp;
        if (nabs.x > nabs.y) {
            ds = 1.0f / nabs.x;
            dp = make_float2(0,0.5f);
        } else {
            ds = 1.0f / nabs.y;
            dp = make_float2(0.5f,0);
        }

        float2 uv = make_float2(ix + 0.5f, iy + 0.5f);
        for( float d = ds; d <= radius; d += ds ) {
            float2 o = d*n;
            float2 q;

            q = make_float2(uv.x + o.x + dp.x, uv.y + o.y + dp.y);
            {
                float3 c = make_float3(tex2D( texSRC4, q.x, q.y));
                mm.add(c);
            }

            q = make_float2(uv.x + o.x - dp.x, uv.y + o.y - dp.y);
            {
                float3 c = make_float3(tex2D( texSRC4, q.x, q.y));
                mm.add(c);
            }

            q = make_float2(uv.x - o.x + dp.x, uv.y - o.y + dp.x);
            {
                float3 c = make_float3(tex2D( texSRC4, q.x, q.y));
                mm.add(c);
            }

            q = make_float2(uv.x - o.x - dp.x, uv.y - o.y - dp.x);
            {
                float3 c = make_float3(tex2D( texSRC4, q.x, q.y));
                mm.add(c);
            }
        }
    }

    dst(ix, iy) = make_float4((sign > 0)? mm.max_val : ((sign < 0)? mm.min_val : c0), 1);
}


gpu_image<float4> gpu_ivacef_shock( const gpu_image<float>& L, const gpu_image<float4>& src, 
                                    const gpu_image<float4>& tfm, float sigma, float tau,
                                    float radius ) {
    gpu_image<float4> dst(src.size());
    bind(&texSRC4, src);
    bind(&texSRC1, L);
    texSRC1.filterMode = cudaFilterModeLinear;
    imp_gradient_shock<<<dst.blocks(), dst.threads()>>>( tfm, dst, sigma, tau, radius );
    texSRC1.filterMode = cudaFilterModePoint;
    GPU_CHECK_ERROR();
    return dst;
}


gpu_image<float4> gpu_ivacef_sobel( const gpu_image<float4>& src, const gpu_image<float4>& st, 
                                    float sigma_d, float tau_r )
{
    gpu_image<float4> st2 = st_sobel(src, st, tau_r);
    st2 = gpu_gauss_filter_xy(st2, sigma_d);
    return st2;
}


gpu_image<float4> gpu_ivacef( const gpu_image<float4>& src, int N, float sigma_d, float tau_r, 
                              float sigma_t, float max_angle, float sigma_i, float sigma_g, 
                              float r, float tau_s, float sigma_a )
{
    gpu_image<float4> img = src;
    gpu_image<float4> st;

    for (int k = 0; k < N; ++k) {
        st = gpu_ivacef_sobel(img, st, sigma_d, tau_r);
        img = gpu_stgauss2_filter(img, st, sigma_t, max_angle, true, true, true, 2, 1);

        st = gpu_ivacef_sobel(img, st, sigma_d, tau_r);
        gpu_image<float4> tfm = gpu_st_tfm(st);   

        gpu_image<float> L = gpu_rgb2gray(img);
        L =  gpu_gauss_filter_xy(L, sigma_i);
        img = gpu_ivacef_shock(L, img, tfm, sigma_g, tau_s, r);
    }

    img = gpu_stgauss2_filter(img, st, sigma_a, 90, false, true, true, 2, 1);
    return img;
}
