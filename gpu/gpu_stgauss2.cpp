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
#include "gpu_stgauss2.h"
#include "gpu_st.h"
#include "cpu_sampler.h"
#include <deque>


struct stgauss2_path {
    __host__ stgauss2_path(std::deque<float3>& dst, float sigma) 
         : dst_(dst)
    { 
        radius_ = 2 * sigma;
    }

    __host__ float radius() const {
        return radius_;
    }

    __host__ void operator()(float sign, float u, float2 p) {
        if (sign < 0) 
            dst_.push_front(make_float3(p.x, p.y, sign*u));
        else 
            dst_.push_back(make_float3(p.x, p.y, sign*u));
    }

    std::deque<float3>& dst_;
    float radius_;
};


std::vector<float3> gpu_stgauss2_path( int ix, int iy, const cpu_image<float4>& st, float sigma, float max_angle, 
                                       bool adaptive, bool st_linear, int order, float step_size )
{
    cpu_sampler<float4> st_sampler(st, st_linear? cudaFilterModeLinear : cudaFilterModePoint);
    float2 p0 = make_float2(ix + 0.5f, iy + 0.5f);
    if (adaptive) {
        float A = st2A(st(p0.x, p0.y));
        sigma *= 0.25f * (1 + A)*(1 + A);
    }
    std::deque<float3> C;
    float cos_max = cosf(radians(max_angle));
    stgauss2_path f(C, sigma);
    if (order == 1) st_integrate_euler(p0, st_sampler, f, cos_max, st.w(), st.h(), step_size);
    if (order == 2) st_integrate_rk2(p0, st_sampler, f, cos_max, st.w(), st.h(), step_size);
    if (order == 4) st_integrate_rk4(p0, st_sampler, f, cos_max, st.w(), st.h(), step_size);
    
    return std::vector<float3>(C.begin(), C.end());
}
