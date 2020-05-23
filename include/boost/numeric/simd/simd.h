/*
 *   Copyright (C) 2019-2020 Cem Bassoy (cem.bassoy@gmail.com)
 *
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU Lesser General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef BOOST_NUMERIC_SIMD_H
#define BOOST_NUMERIC_SIMD_H


#if defined(__AVX512F__)

#include "simd_avx512.h"
#include "simd_avx.h"
#include "simd_sse.h"
#include "simd_no_intrinsics.h"
#include "csimd.h"

namespace boost::numeric::simd
{

using double2  = simd_t<double, sse_intrinsics_tag>;
using double4  = simd_t<double, avx_intrinsics_tag>;
using double8  = simd_t<double, avx512_intrinsics_tag>;

using float4  = simd_t<float,  sse_intrinsics_tag>;
using float8  = simd_t<float,  avx_intrinsics_tag>;
using float16 = simd_t<float,  avx512_intrinsics_tag>;

using cdouble2 = csimd_t<double2>;
using cdouble4 = csimd_t<double4>;
using cdouble8 = csimd_t<double8>;

using cfloat4  = csimd_t<float4 >;
using cfloat8  = csimd_t<float8 >;
using cfloat16 = csimd_t<float16>;

using max_intrinsics_tag = avx512_intrinsics_tag;

}

#elif defined(__AVX__)

#include "simd_avx.h"
#include "simd_sse.h"
#include "simd_no_intrinsics.h"
#include "csimd.h"

namespace boost::numeric::simd 
{

using double2  = simd_t<double, sse_intrinsics_tag>;
using double4  = simd_t<double, avx_intrinsics_tag>;
using double8  = simdn_t<double,8>;

using float4  = simd_t<float,  sse_intrinsics_tag>;
using float8  = simd_t<float,  avx_intrinsics_tag>;
using float16 = simdn_t<float,16>;

using cdouble2  = csimd_t<double2 >;
using cdouble4  = csimd_t<double4 >;
using cdouble8  = csimd_t<double8 >;

using cfloat4  = csimd_t<float4>;
using cfloat8  = csimd_t<float8>;
using cfloat16 = csimd_t<float16>;

using max_intrinsics_tag = avx_intrinsics_tag;


}

#elif defined(__SSE__)

#include "simd_sse.h"
#include "simd_no_intrinsics.h"
#include "csimd.h"

namespace boost::numeric::simd  
{

using double2  = simd_t <double, sse_intrinsics_tag>;
using double4  = simdn_t<double, 4>;
using double8  = simdn_t<double, 8>;

using float4   = simd_t <float,  sse_intrinsics_tag>;
using float8   = simdn_t<float,  8>;
using float16  = simdn_t<float, 16>;

using cdouble2  = csimd_t<double2 >;
using cdouble4  = csimd_t<double4 >;
using cdouble8  = csimd_t<double8 >;

using cfloat4   = csimd_t<float4 >;
using cfloat8   = csimd_t<float8 >;
using cfloat16  = csimd_t<float16>;

using max_intrinsics_tag = sse_intrinsics_tag;


}

#else

#include "simd_no_intrinsics.h"
#include "csimd.h"

namespace boost::numeric::simd
{

using double2  = simdn_t<double, 2>;
using double4  = simdn_t<double, 4>;
using double8  = simdn_t<double, 8>;

using float4  = simdn_t<float,  4>;
using float8  = simdn_t<float,  8>;
using float16 = simdn_t<float, 16>;

using cdouble2  = csimd_t<double2 >;
using cdouble4  = csimd_t<double4 >;
using cdouble8  = csimd_t<double8 >;

using cfloat4   = csimd_t<float4 >;
using cfloat8   = csimd_t<float8 >;
using cfloat16  = csimd_t<float16>;

using max_intrinsics_tag = no_intrinsics_tag;


}

#endif


namespace std 
{
template<> struct alignment_of<boost::numeric::simd::double2> : public integral_constant<size_t, boost::numeric::simd::double2::size()> {};
template<> struct alignment_of<boost::numeric::simd::double4> : public integral_constant<size_t, boost::numeric::simd::double4::size()> {};
template<> struct alignment_of<boost::numeric::simd::double8> : public integral_constant<size_t, boost::numeric::simd::double8::size()> {};
template<> struct alignment_of<boost::numeric::simd::float4>  : public integral_constant<size_t, boost::numeric::simd::float4 ::size()> {};
template<> struct alignment_of<boost::numeric::simd::float8>  : public integral_constant<size_t, boost::numeric::simd::float8 ::size()> {};
template<> struct alignment_of<boost::numeric::simd::float16> : public integral_constant<size_t, boost::numeric::simd::float16::size()> {};
}







#endif


