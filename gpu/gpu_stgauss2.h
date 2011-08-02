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
#pragma once

#include "gpu_image.h"
#include <vector>

gpu_image<float> gpu_stgauss2_filter( const gpu_image<float>& src, const gpu_image<float4>& st, 
                                      float sigma, float max_angle, bool adaptive,
                                      bool src_linear, bool st_linear, int order, float step_size );

gpu_image<float4> gpu_stgauss2_filter( const gpu_image<float4>& src, const gpu_image<float4>& st, 
                                       float sigma, float max_angle, bool adaptive,
                                       bool src_linear, bool st_linear, int order, float step_size );

std::vector<float3> gpu_stgauss2_path( int ix, int iy, const cpu_image<float4>& st, 
                                       float sigma, float max_angle, bool adaptive, 
                                       bool st_linear, int order, float step_size );


template <typename ST, typename F>
inline __host__ __device__ void st_integrate_euler( float2 p0, const ST& st, F& f, float cos_max, 
                                                    unsigned w, unsigned h, float step_size ) 
{
    f(0, 0, p0);
    float2 v0 = st2tangent(st(p0.x, p0.y));
    float sign = -1;
    do {
        float2 v = v0 * sign;
        float2 p = p0 + step_size * v;
        float u = step_size;
        while ((u < f.radius()) && 
               (p.x >= 0) && (p.x < w) && (p.y >= 0) && (p.y < h))  
        {
            f(sign, u, p);

            float2 t = st2tangent(st(p.x, p.y));
            float vt = dot(v, t);
            if (fabs(vt) <= cos_max) break;
            if (vt < 0) t = -t;

            v = t;
            p += step_size * t;
            u += step_size;
        }

        sign *= -1;
    } while (sign > 0);
}


template <typename ST, typename F>
inline __host__ __device__ void st_integrate_rk2( float2 p0, const ST& st, F& f, float cos_max, 
                                                  unsigned w, unsigned h, float step_size ) 
{
    f(0, 0, p0);
    float2 v0 = st2tangent(st(p0.x, p0.y));
    float sign = -1;
    do {
        float2 v = v0 * sign;
        float2 p = p0 + step_size * v;
        float u = step_size;
        while ((u < f.radius()) && 
               (p.x >= 0) && (p.x < w) && (p.y >= 0) && (p.y < h))  
        {
            f(sign, u, p);

            float2 t = st2tangent(st(p.x, p.y));
            float vt = dot(v, t);
            //if (fabs(vt) <= cos_max) break;
            if (vt < 0) t = -t;

            t = st2tangent(st(p.x + 0.5f * step_size * t.x, p.y + 0.5f * step_size * t.y));
            vt = dot(v, t);
            if (fabs(vt) <= cos_max) break;
            if (vt < 0) t = -t;

            v = t;
            p += step_size * t;
            u += step_size;
        }

        sign *= -1;
    } while (sign > 0);
}


template <typename ST, typename F>
inline __host__ __device__ void st_integrate_rk4( float2 p0, const ST& st, F& f, float cos_max, 
                                                  unsigned w, unsigned h, float step_size ) {
    f(0, 0, p0);
    float2 v0 = st2tangent(st(p0.x, p0.y));
    float sign = -1;
    do {
        float2 v = v0 * sign;
        float2 p = p0 + step_size * v;
        float u = step_size;
        while ((u < f.radius()) && 
               (p.x >= 0) && (p.x < w) && (p.y >= 0) && (p.y < h))  
        {
            f(sign, u, p);

            float2 k1 = st2tangent(st(p.x, p.y));
            float vt = dot(v, k1);
            if (vt < 0) k1 = -k1;

            float2 k2 = st2tangent(st(p.x + 0.5f * step_size * k1.x, p.y + 0.5f * step_size * k1.y));
            vt = dot(v, k2);
            if (vt < 0) k2 = -k2;

            float2 k3 = st2tangent(st(p.x + 0.5f * step_size * k2.x, p.y + 0.5f * step_size * k2.y));
            vt = dot(v, k3);
            if (vt < 0) k3 = -k3;

            float2 k4 = st2tangent(st(p.x + step_size * k3.x, p.y + step_size * k3.y));
            vt = dot(v, k4);
            if (vt < 0) k4 = -k4;

            float2 t = (k1 + 2*k2 + 2*k3 + k4) / 6.0f;
            vt = dot(v, t);
            if (fabs(vt) <= cos_max) break;
            v = t;
            p += step_size * t;
            u += step_size;
        }

        sign *= -1;
    } while (sign > 0);
}
