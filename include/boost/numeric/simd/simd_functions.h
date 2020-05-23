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


#ifndef BOOST_NUMERIC_SIMD_FUNCTIONS_H
#define BOOST_NUMERIC_SIMD_FUNCTIONS_H

#include "simd.h"


//////////////////////////
/// Functions Float<8>
//////////////////////////

#if defined (__AVX512F__)

namespace boost::numeric::simd 
{

inline float16 rsqrt(const float16& v)
{
	return float16(_mm512_rsqrt14_ps(v.data()));
}


inline float16 sqrt(const float16 &v)
{
	return float16(_mm512_sqrt_ps(v.data()));
}


inline typename float16::value_type sum(const float16 & v)
{
	return _mm512_reduce_add_ps(v.data());
}


inline typename float16::value_type dot(const float16 & v, const float16 & w)
{
	return sum ( v * w );
}



inline typename float16::value_type norm2(const float16 & v)
{
	return std::sqrt( boost::numeric::simd::dot ( v , v ) );
}


inline float16 min(const float16 & v, const float16 & w)
{
	return float16(_mm512_min_ps(v.data(),w.data()));
}

inline float16 max(const float16 & v, const float16 & w)
{
	return float16(_mm512_max_ps(v.data(),w.data()));
}

inline float16 ceil(const float16 & v)
{
	return float16(_mm512_ceil_ps(v.data()));
}

inline float16 floor(const float16 & v)
{
	return float16(_mm512_floor_ps(v.data()));
}

inline float16 round(const float16 & v)
{
	return float16( _mm512_roundscale_round_ps(v.data(), 0, _MM_FROUND_TO_NEAREST_INT ) ); // sse4
}


inline float16 pow(const float16 & x, const float16& y)
{
	float16 buf;
	for(auto i = 0u; i < float16::size(); ++i)
		buf[i] = std::pow(x[i],y[i]);

	return buf;
}


inline float16 pow(const float16 & x, const typename float16::value_type& y)
{
	float16 buf;
	for(auto i = 0u; i < float16::size(); ++i)
		buf[i] = std::pow(x[i],y);
	return buf;
}



inline float16 sin(const float16 & v)
{
	float16 buf;
	for(auto i = 0u; i < float16::size(); ++i)
		buf[i] = std::sin(v[i]);
	return buf;
}

inline float16 cos(const float16 & v)
{
	float16 buf;
	for(auto i = 0u; i < float16::size(); ++i)
		buf[i] = std::cos(v[i]);
	return buf;
}


// return sin, argument cos
inline float16 sincos(float16 & c, const float16 & x)
{
	c = cos(x);
	return sin(x);
}


float16
inline tan(const float16 & v)
{
	float16 buf;
	for(auto i = 0u; i < float16::size(); ++i)
		buf[i] = std::tan(v[i]);
	return buf;
}


float16
inline exp(const float16 & v)
{
	float16 buf;
	for(auto i = 0u; i < float16::size(); ++i)
		buf[i] = std::exp(v[i]);
	return buf;
}

float16
inline log(const float16 & v)
{
	float16 buf;
	for(auto i = 0u; i < float16::size(); ++i)
		buf[i] = std::log(v[i]);
	return buf;
}

float16
inline log10(const float16 & v)
{
	float16 buf;
	for(auto i = 0u; i < float16::size(); ++i)
		buf[i] = std::log10(v[i]);
	return buf;
}

}



//////////////////////////
/// Functions Double<8>
//////////////////////////




namespace boost::numeric::simd
{



inline double8 rsqrt(const double8& v)
{
	return double8(_mm512_rsqrt14_pd(v.data()));
}


inline double8 sqrt(const double8 &v)
{
	return double8(_mm512_sqrt_pd(v.data()));
}


inline typename double8::value_type sum(const double8 & v)
{
	return _mm512_reduce_add_pd(v.data());
}


inline typename double8::value_type dot(const double8 & v, const double8 & w)
{
	return sum ( v * w );
}



inline typename double8::value_type norm2(const double8 & v)
{
	return std::sqrt( boost::numeric::simd::dot ( v , v ) );
}


inline double8 min(const double8 & v, const double8 & w)
{
	return double8(_mm512_min_pd(v.data(),w.data()));
}

inline double8 max(const double8 & v, const double8 & w)
{
	return double8(_mm512_max_pd(v.data(),w.data()));
}

inline double8 ceil(const double8 & v)
{
	return double8(_mm512_ceil_pd(v.data()));
}

inline double8 floor(const double8 & v)
{
	return double8(_mm512_floor_pd(v.data()));
}

inline double8 round(const double8 & v)
{
	return double8( _mm512_roundscale_round_pd(v.data(), 0, _MM_FROUND_TO_NEAREST_INT ) ); // sse4
}


inline double8 pow(const double8 & x, const double8& y)
{
	double8 buf;
	for(auto i = 0u; i < double8::size(); ++i)
		buf[i] = std::pow(x[i],y[i]);

	return buf;
}


inline double8 pow(const double8 & x, const typename double8::value_type& y)
{
	double8 buf;
	for(auto i = 0u; i < double8::size(); ++i)
		buf[i] = std::pow(x[i],y);
	return buf;
}



inline double8 sin(const double8 & v)
{
	double8 buf;
	for(auto i = 0u; i < double8::size(); ++i)
		buf[i] = std::sin(v[i]);
	return buf;
}

inline double8 cos(const double8 & v)
{
	double8 buf;
	for(auto i = 0u; i < double8::size(); ++i)
		buf[i] = std::cos(v[i]);
	return buf;
}


// return sin, argument cos
inline double8 sincos(double8 & c, const double8 & x)
{
	c = cos(x);
	return sin(x);
}


double8
inline tan(const double8 & v)
{
	double8 buf;
	for(auto i = 0u; i < double8::size(); ++i)
		buf[i] = std::tan(v[i]);
	return buf;
}


double8
inline exp(const double8 & v)
{
	double8 buf;
	for(auto i = 0u; i < double8::size(); ++i)
		buf[i] = std::exp(v[i]);
	return buf;
}

double8
inline log(const double8 & v)
{
	double8 buf;
	for(auto i = 0u; i < double8::size(); ++i)
		buf[i] = std::log(v[i]);
	return buf;
}

double8
inline log10(const double8 & v)
{
	double8 buf;
	for(auto i = 0u; i < double8::size(); ++i)
		buf[i] = std::log10(v[i]);
	return buf;
}

}

#endif

//////////////////////////
/// Functions Float<8>
//////////////////////////


#if defined (__AVX__)

namespace boost::numeric::simd
{

float8
inline rsqrt(const float8& v)
{
	return float8(_mm256_rsqrt_ps(v.data()));
}



float8
inline sqrt(const float8 &v)
{
	return float8(_mm256_sqrt_ps(v.data()));
}


typename float8::value_type
inline sum(const float8 & v)
{
	auto temp = _mm256_permute2f128_ps(v.data(),v.data(),1);
	temp = _mm256_add_ps(temp,v.data());
	temp = _mm256_hadd_ps(temp,temp);
	temp = _mm256_hadd_ps(temp,temp);
	return ((float8::value_type*)&temp)[0];
	//return temp[0];
}


typename float8::value_type
inline dot(const float8 & v, const float8 & w)
{
	return sum ( v * w );
}



typename float8::value_type
inline norm2(const float8 & v)
{
	return std::sqrt( boost::numeric::simd::dot ( v , v ) );
}


float8
inline min(const float8 & v, const float8 & w)
{
	return float8(_mm256_min_ps(v.data(),w.data()));
}

float8
inline max(const float8 & v, const float8 & w)
{
	return float8(_mm256_max_ps(v.data(),w.data()));
}

float8
inline ceil(const float8 & v)
{
	return float8(_mm256_ceil_ps(v.data()));
}

float8
inline floor(const float8 & v)
{
	return float8(_mm256_floor_ps(v.data()));
}

float8
inline round(const float8 & v)
{
	return float8( _mm256_round_ps(v.data(), _MM_FROUND_TO_NEAREST_INT ) ); // sse4
}


float8
inline pow(const float8 & x, const float8& y)
{

#ifdef _INTEL_SVML_
	return float8(_mm256_pow_ps(x.data(), y.data()));
#else
	float8 buf;
	buf[0] = std::pow(x[0],y[0]); buf[1] = std::pow(x[1],y[1]); buf[2] = std::pow(x[2],y[2]); buf[3] = std::pow(x[3],y[3]);
	buf[4] = std::pow(x[0],y[0]); buf[5] = std::pow(x[5],y[5]); buf[6] = std::pow(x[6],y[6]); buf[7] = std::pow(x[7],y[7]);
	return buf;
#endif
}


float8
inline pow(const float8 & x, const typename float8::value_type& y)
{
#ifdef _INTEL_SVML_
	return float8(_mm256_pow_ps(x.data(), _mm256_set1_ps(y)  ));
#else
	float8 buf;
	buf[0] = std::pow(x[0],y); buf[1] = std::pow(x[1],y); buf[2] = std::pow(x[2],y); buf[3] = std::pow(x[3],y);
	buf[4] = std::pow(x[0],y); buf[5] = std::pow(x[5],y); buf[6] = std::pow(x[6],y); buf[7] = std::pow(x[7],y);
	return buf;
#endif
}



float8
inline sin(const float8 & v)
{
#ifdef _INTEL_SVML_
	return float8(_mm256_sin_ps(v.data()));
#else
	float8 buf;
	buf[0] = std::sin(v[0]); buf[1] = std::sin(v[1]); buf[2] = std::sin(v[2]); buf[3] = std::sin(v[3]);
	buf[4] = std::sin(v[4]); buf[5] = std::sin(v[5]); buf[6] = std::sin(v[6]); buf[7] = std::sin(v[7]);
	return buf;
#endif
}

float8
inline cos(const float8 & v)
{
#ifdef _INTEL_SVML_
	return float8(_mm256_cos_ps(v.data()));
#else
	float8 buf;
	buf[0] = std::cos(v[0]); buf[1] = std::cos(v[1]); buf[2] = std::cos(v[2]); buf[3] = std::cos(v[3]);
	buf[4] = std::cos(v[4]); buf[5] = std::cos(v[5]); buf[6] = std::cos(v[6]); buf[7] = std::cos(v[7]);
	return buf;
#endif
}


// return sin, argument cos
float8
inline sincos(float8 & c, const float8 & x)
{
	c = cos(x);
    return sin(x);
}


float8
inline tan(const float8 & v)
{
#ifdef _INTEL_SVML_
	return float8(_mm256_tan_ps(v.data()));
#else
	float8 buf;
	buf[0] = std::tan(v[0]); buf[1] = std::tan(v[1]); buf[2] = std::tan(v[2]); buf[3] = std::tan(v[3]);
	buf[4] = std::tan(v[4]); buf[5] = std::tan(v[5]); buf[6] = std::tan(v[6]); buf[7] = std::tan(v[7]);
	return buf;
#endif
}

float8
inline exp(const float8 & v)
{
#ifdef _INTEL_SVML_
	return float8(_mm256_exp_ps(v.data()));
#else
	float8 buf;
	buf[0] = std::exp(v[0]); buf[1] = std::exp(v[1]); buf[2] = std::exp(v[2]); buf[3] = std::exp(v[3]);
	buf[4] = std::exp(v[4]); buf[5] = std::exp(v[5]); buf[6] = std::exp(v[6]); buf[7] = std::exp(v[7]);
	return buf;
#endif
}

float8
inline log(const float8 & v)
{
#ifdef _INTEL_SVML_
	return float8(_mm256_log_ps(v.data()));
#else
	float8 buf;
	buf[0] = std::log(v[0]); buf[1] = std::log(v[1]); buf[2] = std::log(v[2]); buf[3] = std::log(v[3]);
	buf[4] = std::log(v[4]); buf[5] = std::log(v[5]); buf[6] = std::log(v[6]); buf[7] = std::log(v[7]);
	return buf;
#endif
}

float8
inline log10(const float8 & v)
{
#ifdef _INTEL_SVML_
	return float8(_mm256_log10_ps(v.data()));
#else
	float8 buf;
	buf[0] = std::log10(v[0]); buf[1] = std::log10(v[1]); buf[2] = std::log10(v[2]); buf[3] = std::log10(v[3]);
	buf[4] = std::log10(v[4]); buf[5] = std::log10(v[5]); buf[6] = std::log10(v[6]); buf[7] = std::log10(v[7]);
	return buf;
#endif
}


}










////////////////////////////
/// Functions Double4
////////////////////////////
namespace boost::numeric::simd{

inline double4 rsqrt(const double4& double_vector)
{
	double4 one(1.0);
	return double4(_mm256_div_pd(one.data(),_mm256_sqrt_pd(double_vector.data())));
}



inline double4 sqrt(const double4 &v)
{
	return double4(_mm256_sqrt_pd(v.data()));
}


inline typename double4::value_type sum(const double4 & v)
{
	auto temp = _mm256_permute2f128_pd(v.data(),v.data(),1);
	temp = _mm256_add_pd(temp,v.data());
	temp = _mm256_hadd_pd(temp,temp);
	return ((double4::value_type*)&temp)[0];
}


inline typename double4::value_type dot(const double4 & v, const double4 & w)
{
	return sum ( v * w );
}



inline typename double4::value_type norm2(const double4 & v)
{
	return std::sqrt( boost::numeric::simd::dot ( v , v ) );
}


inline double4 min(const double4 & v, const double4 & w)
{
	return double4(_mm256_min_pd(v.data(),w.data()));
}

inline double4 max(const double4 & v, const double4 & w)
{
	return double4(_mm256_max_pd(v.data(),w.data()));
}

inline double4 ceil(const double4 & v)
{
	return double4(_mm256_ceil_pd(v.data()));
}

inline double4 floor(const double4 & v)
{
	return double4(_mm256_floor_pd(v.data()));
}

inline double4 round(const double4 & v)
{
	return double4( _mm256_round_pd(v.data(), _MM_FROUND_TO_NEAREST_INT ) ); // sse4
}


inline double4 pow(const double4 & x, const double4& y)
{
#ifdef _INTEL_SVML_
	return double4(_mm256_pow_pd(x.data(), y.data()));
#else
	double4 buf;
	buf[0] = std::pow(x[0],y[0]); buf[1] = std::pow(x[1],y[1]); buf[2] = std::pow(x[2],y[2]); buf[3] = std::pow(x[3],y[3]);
	return buf;
#endif
}


inline double4 pow(const double4 & x, const typename double4::value_type& y)
{
#ifdef _INTEL_SVML_
	return double4(_mm256_pow_pd(x.data(), _mm256_set1_pd(y)  ));
#else
	double4 buf;
	buf[0] = std::pow(x[0],y); buf[1] = std::pow(x[1],y); buf[2] = std::pow(x[2],y); buf[3] = std::pow(x[3],y);
	return buf;
#endif
}

inline double4 sin(const double4 & v)
{
#ifdef _INTEL_SVML_
	return double4(_mm256_sin_pd(v.data()));
#else
	double4 buf;
	buf[0] = std::sin(v[0]); buf[1] = std::sin(v[1]); buf[2] = std::sin(v[2]); buf[3] = std::sin(v[3]);
	return buf;
#endif
}

inline double4 cos(const double4 & v)
{
#ifdef _INTEL_SVML_
	return double4(_mm256_cos_pd(v.data()));
#else
	double4 buf;
	buf[0] = std::cos(v[0]); buf[1] = std::cos(v[1]); buf[2] = std::cos(v[2]); buf[3] = std::cos(v[3]);
	return buf;
#endif
}

// return sin, argument cos
inline double4 sincos(double4 & c, const double4 & x)
{
    c = cos(x);
    return sin(x);
}

inline double4 tan(const double4 & v)
{
#ifdef _INTEL_SVML_
	return double4(_mm256_tan_pd(v.data()));
#else
	double4 buf;
	buf[0] = std::tan(v[0]); buf[1] = std::tan(v[1]); buf[2] = std::tan(v[2]); buf[3] = std::tan(v[3]);
	return buf;
#endif
}

inline double4 exp(const double4 & v)
{
#ifdef _INTEL_SVML_
	return double4(_mm256_exp_pd(v.data()));
#else
	double4 buf;
	buf[0] = std::exp(v[0]); buf[1] = std::exp(v[1]); buf[2] = std::exp(v[2]); buf[3] = std::exp(v[3]);
	return buf;
#endif
}

inline double4 log(const double4 & v)
{
#ifdef _INTEL_SVML_
	return double4(_mm256_log_pd(v.data()));
#else
	double4 buf;
	buf[0] = std::log(v[0]); buf[1] = std::log(v[1]); buf[2] = std::log(v[2]); buf[3] = std::log(v[3]);
	return buf;
#endif
}

inline double4 log10(const double4 & v)
{
#ifdef _INTEL_SVML_
	return double4(_mm256_log10_pd(v.data()));
#else
	double4 buf;
	buf[0] = std::log10(v[0]); buf[1] = std::log10(v[1]); buf[2] = std::log10(v[2]); buf[3] = std::log10(v[3]);
	return buf;
#endif
}

}
#endif


#if defined (__SSE__)

////////////////////////////
///// Functions Float<4>
////////////////////////////

namespace boost::numeric::simd{

inline bool is_all_zero(const float4 & v)
{
	return (v[0] == 0.0f) && (v[1] == 0.0f) && (v[2] == 0.0f) && (v[3] == 0.0f);
}

inline bool is_all_one(const float4 & v)
{
	return (v[0] == 1.0f) && (v[1] == 1.0f) && (v[2] == 1.0f) && (v[3] == 1.0f);
}


inline float4 rsqrt(const float4& v)
{
#if NEWTON                                                      // Switch on Newton-Raphson correction
	float4 temp = float4(_mm_rsqrt_ps(v.data()));
    temp *= (temp * temp * v - 3.0f) * (-0.5f);
    return temp;
#else
	return float4(_mm_rsqrt_ps(v.data()));
#endif

}

inline float4 sqrt(const float4 &v)
{
	return float4(_mm_sqrt_ps(v.data()));
}


typename float4::value_type sum(const float4 & v)
{
//	auto temp = _mm_hadd_ps(_mm_hadd_ps(v.data(),v.data()),v.data());
//	return ((float4::value_type*)&temp)[0];
    return v[0]+v[1]+v[2]+v[3];
}

inline typename float4::value_type dot(const float4 & v, const float4 & w)
{
	return sum ( v * w );
}

inline typename float4::value_type norm2(const float4 & v)
{
	return std::sqrt( boost::numeric::simd::dot( v , v ));
}


inline float4 min(const float4 & v, const float4 & w)
{
	return float4(_mm_min_ps(v.data(),w.data()));
}

inline float4 max(const float4 & v, const float4 & w)
{
	return float4(_mm_max_ps(v.data(),w.data()));
}

inline float4 ceil(const float4 & v)
{
	return float4(_mm_ceil_ps(v.data()));
}

inline float4 floor(const float4 & v)
{
	return float4(_mm_floor_ps(v.data()));
}

inline float4 round(const float4 & v)
{
	return float4( _mm_round_ps(v.data(), _MM_FROUND_TO_NEAREST_INT ) ); // sse4
}


inline float4 pow(const float4 & x, const float4& y)
{
#ifdef _INTEL_SVML_
	return float4(_mm_pow_ps(x.data(), y.data()));
#else
	float4 buf;
	buf[0] = std::pow(x[0],y[0]); buf[1] = std::pow(x[1],y[1]); buf[2] = std::pow(x[2],y[2]); buf[3] = std::pow(x[3],y[3]);
	return buf;
#endif
}


inline float4 pow(const float4 & x, const typename float4::value_type& y)
{
#ifdef _INTEL_SVML_
	return float4(_mm_pow_ps(x.data(), _mm_set1_ps(y)  ));
#else
	float4 buf;
	buf[0] = std::pow(x[0],y); buf[1] = std::pow(x[1],y); buf[2] = std::pow(x[2],y); buf[3] = std::pow(x[3],y);
	return buf;
#endif
}


inline float4 sin(const float4 & v)
{
#ifdef _INTEL_SVML_
	return float4(_mm_sin_ps(v.data()));
#else
	float4 buf;
	buf[0] = std::sin(v[0]); buf[1] = std::sin(v[1]); buf[2] = std::sin(v[2]); buf[3] = std::sin(v[3]);
	return buf;
#endif
}

inline float4 cos(const float4 & v)
{
#ifdef _INTEL_SVML_
	return float4(_mm_cos_ps(v.data()));
#else
	float4 buf;
	buf[0] = std::cos(v[0]); buf[1] = std::cos(v[1]); buf[2] = std::cos(v[2]); buf[3] = std::cos(v[3]);
	return buf;
#endif
}

inline float4 sincos(float4 & c, const float4 & x)
{
    c = cos(x);
    return sin(x);
}


inline float4 tan(const float4 & v)
{
#ifdef _INTEL_SVML_
	return float4(_mm_tan_ps(v.data()));
#else
	float4 buf;
	buf[0] = std::tan(v[0]); buf[1] = std::tan(v[1]); buf[2] = std::tan(v[2]); buf[3] = std::tan(v[3]);
	return buf;
#endif
}

inline float4 exp(const float4 & v)
{
#ifdef _INTEL_SVML_
	return float4(_mm_exp_ps(v.data()));
#else
	float4 buf;
	buf[0] = std::exp(v[0]); buf[1] = std::exp(v[1]); buf[2] = std::exp(v[2]); buf[3] = std::exp(v[3]);
	return buf;
#endif
}

inline float4 log(const float4 & v)
{
#ifdef _INTEL_SVML_
	return float4(_mm_log_ps(v.data()));
#else
	float4 buf;
	buf[0] = std::log(v[0]); buf[1] = std::log(v[1]); buf[2] = std::log(v[2]); buf[3] = std::log(v[3]);
	return buf;
#endif
}

inline float4 log10(const float4 & v)
{
#ifdef _INTEL_SVML_
	return float4(_mm_log10_ps(v.data()));
#else
	float4 buf;
	buf[0] = std::log10(v[0]); buf[1] = std::log10(v[1]); buf[2] = std::log10(v[2]); buf[3] = std::log10(v[3]);
	return buf;
#endif
}

} // namespace boost::numeric::simd



namespace boost::numeric::simd{

inline
bool is_all_zero(const double2 & v)
{
	return (v[0] == 0.0) && (v[1] == 0.0);
}

inline
bool is_all_one(const double2 & v)
{
	return (v[0] == 1.0) && (v[1] == 1.0);
}




inline
double2
sqrt(const double2 &v)
{
#ifdef _INTEL_SVML_ // less efficient
	return double2(_mm_svml_sqrt_pd(v.data()));
#else
	return double2(_mm_sqrt_pd(v.data()));
#endif
}

inline
double2
rsqrt(const double2& v)
{
#ifdef _INTEL_SVML_
	return double2(_mm_invsqrt_pd(v.data()));
#else
	#if NEWTON                                                      // Switch on Newton-Raphson correction
			float4 in(v[0],v[1],0,0);
			float4 temp = rsqrt(in);
    		temp *= (temp * temp * in - 3.0f) * (-0.5f);
			double2 out(temp[0],temp[1]);
    		return out;
	#endif
	return 1.0/sqrt(v);
#endif
}


inline
typename double2::value_type
sum(const double2 & v)
{
    return v[0]+v[1];
//	auto temp = _mm_hadd_pd(v.data(),v.data());
//	return ((double2::value_type*)&temp)[0];
}


typename double2::value_type
inline dot(const double2 & v, const double2 & w)
{
	return sum ( v * w );
}



typename double2::value_type
inline norm2(const double2 & v)
{
	return std::sqrt( boost::numeric::simd::dot ( v , v ) );
}

inline
double2
min(const double2 & v, const double2 & w)
{
	return double2(_mm_min_pd(v.data(),w.data()));
}

inline
double2
max(const double2 & v, const double2 & w)
{
	return double2(_mm_max_pd(v.data(),w.data()));
}

inline
double2
ceil(const double2 & v)
{
	return double2(_mm_ceil_pd(v.data()));
}

inline
double2
floor(const double2 & v)
{
	return double2(_mm_floor_pd(v.data()));
}

inline
double2
round(const double2 & v)
{
	return double2( _mm_round_pd(v.data(), _MM_FROUND_TO_NEAREST_INT ) ); // sse4
}



double2
inline pow(const double2 & x, const double2& y)
{
#ifdef _INTEL_SVML_
	return double2(_mm_pow_pd(x.data(), y.data()));
#else
	double2 buf;
	buf[0] = std::pow(x[0],y[0]); buf[1] = std::pow(x[1],y[1]);
	return buf;
#endif
}


double2
inline pow(const double2 & x, const typename double2::value_type& y)
{
#ifdef _INTEL_SVML_
	return double2(_mm_pow_pd(x.data(), _mm_set1_pd(y)  ));
#else
	double2 buf;
	buf[0] = std::pow(x[0],y); buf[1] = std::pow(x[1],y);
	return buf;
#endif
}





double2
inline sin(const double2 & v)
{
#ifdef _INTEL_SVML_
	return double2(_mm_sin_pd(v.data()));
#else
	double2 buf;
	buf[0] = std::sin(v[0]); buf[1] = std::sin(v[1]);
	return buf;
#endif
}

double2
inline cos(const double2 & v)
{
#ifdef _INTEL_SVML_
	return double2(_mm_cos_pd(v.data()));
#else
	double2 buf;
	buf[0] = std::cos(v[0]); buf[1] = std::cos(v[1]);
	return buf;
#endif
}


// return sin, argument cos
double2
inline sincos(double2 & c, const double2 & x)
{
    c = cos(x);
    return sin(x);
}

double2
inline tan(const double2 & v)
{
#ifdef _INTEL_SVML_
	return double2(_mm_tan_pd(v.data()));
#else
	double2 buf;
	buf[0] = std::tan(v[0]); buf[1] = std::tan(v[1]);
	return buf;
#endif
}

double2
inline exp(const double2 & v)
{
#ifdef _INTEL_SVML_
	return double2(_mm_exp_pd(v.data()));
#else
	double2 buf;
	buf[0] = std::exp(v[0]); buf[1] = std::exp(v[1]);
	return buf;
#endif
}

double2
inline log(const double2 & v)
{
#ifdef _INTEL_SVML_
	return double2(_mm_log_pd(v.data()));
#else
	double2 buf;
	buf[0] = std::log(v[0]); buf[1] = std::log(v[1]);
	return buf;
#endif
}

double2
inline log10(const double2 & v)
{
#ifdef _INTEL_SVML_
	return double2(_mm_log10_pd(v.data()));
#else
	double2 buf;
	buf[0] = std::log10(v[0]); buf[1] = std::log10(v[1]);
	return buf;
#endif
}
} // namespace boost::numeric::simd


#endif

//////////////////////////
/// Functions simd no intrinsics
//////////////////////////

namespace boost::numeric::simd{

template<class T, std::size_t N>
inline bool is_all_zero(const simdn_t<T,N> & v)
{
	return std::all_of(v.begin(),v.end(), [](const T& a) { return a == 0; } );
}

template<class T, std::size_t N>
inline bool is_all_one(const simdn_t<T,N> & v)
{
	return std::all_of(v.begin(),v.end(), [](const T& a) { return a == 1; } );
}


template<class T, std::size_t N>
inline simdn_t<T,N> sqrt(const simdn_t<T,N> &v)
{
	simdn_t<T,N> res;
	std::transform(v.begin(),v.end(), res.begin(), [](const T& a) { return std::sqrt(a); } );
	return res;
}

template<class T, std::size_t N>
inline simdn_t<T,N> rsqrt(const simdn_t<T,N>& v)
{
	return 1.0/sqrt(v);
}


template<class T, std::size_t N>
inline T sum(const simdn_t<T,N> & v)
{
	return std::accumulate(v.begin(),v.end(), T(0));
}

template<class T, std::size_t N>
inline T dot(const simdn_t<T,N> & v, const simdn_t<T,N> & w)
{
	return std::inner_product(v.begin(),v.end(), w.begin(), T(0));
}

template<class T, std::size_t N>
inline T norm2(const simdn_t<T,N> & v)
{
	return std::sqrt( dot ( v , v ) );
}

template<class T, std::size_t N>
inline simdn_t<T,N> min(const simdn_t<T,N> & v, const simdn_t<T,N> & w)
{
	simdn_t<T,N> r;
	std::transform(v.begin(),v.end(), w.begin(), r.begin(), [](const T& a, const T& b) {return (a < b) ? a : b;}  );
	return r;
}


template<class T, std::size_t N>
inline simdn_t<T,N> max(const simdn_t<T,N> & v, const simdn_t<T,N> & w)
{
	simdn_t<T,N> r;
	std::transform(v.begin(),v.end(), w.begin(), r.begin(), [](const T& a, const T& b) {return (a > b) ? a : b;}  );
	return r;
}

template<class T, std::size_t N>
inline simdn_t<T,N> ceil(const simdn_t<T,N> & v)
{
	simdn_t<T,N> res;
	return std::transform(v.begin(),v.end(), res.begin(), [](const T& a) { return std::ceil(a); } );
}

template<class T, std::size_t N>
inline simdn_t<T,N> floor(const simdn_t<T,N> & v)
{
	simdn_t<T,N> res;
	return std::transform(v.begin(),v.end(), res.begin(), [](const T& a) { return std::floor(a); } );
}

template<class T, std::size_t N>
inline simdn_t<T,N> round(const simdn_t<T,N> & v)
{
	simdn_t<T,N> res;
	return std::transform(v.begin(),v.end(), res.begin(), [](const T& a) { return std::round(a); } );
}



template<class T, std::size_t N>
inline simdn_t<T,N> pow(const simdn_t<T,N> & x, const simdn_t<T,N>& y)
{
	simdn_t<T,N> r;
	std::transform(x.begin(), x.end(), y.begin(), r.begin(), [](const T& a, const T& b) { return std::pow(a,b); } );
	return r;
}


template<class T, std::size_t N>
inline simdn_t<T,N> pow(const simdn_t<T,N> & x, const T& y)
{
	simdn_t<T,N> r;
	std::transform(x.begin(), x.end(), r.begin(), [y](const T& a) { return std::pow(a,y); } );
	return r;
}




template<class T, std::size_t N>
inline simdn_t<T,N> sin(const simdn_t<T,N>& v)
{
	simdn_t<T,N> r;
	std::transform(v.begin(), v.end(), r.begin(), [](const T& a) { return std::sin(a); } );
	return r;
}

template<class T, std::size_t N>
inline simdn_t<T,N> cos(const simdn_t<T,N>& v)
{
	simdn_t<T,N> r;
	std::transform(v.begin(), v.end(), r.begin(), [](const T& a) { return std::cos(a); } );
	return r;
}


template<class T, std::size_t N>
inline
simdn_t<T,N>
sincos(simdn_t<T,N>& c_, simdn_t<T,N>& x)
{
	simdn_t<T,N> s;
    std::transform(x.begin(), x.end(), s.begin(), [](const T& a) { return std::sin(a); } );
    std::transform(x.begin(), x.end(), c_.begin(), [](const T& a) { return std::cos(a); } );
    return s;
}

template<class T, std::size_t N>
inline simdn_t<T,N> tan(const simdn_t<T,N>& v)
{
	simdn_t<T,N> r;
	std::transform(v.begin(), v.end(), r.begin(), [](const T& a) { return std::tan(a); } );
	return r;
}

template<class T, std::size_t N>
inline simdn_t<T,N> exp(const simdn_t<T,N>& v)
{
	simdn_t<T,N> r;
	std::transform(v.begin(), v.end(), r.begin(), [](const T& a) { return std::exp(a); } );
	return r;
}

template<class T, std::size_t N>
inline simdn_t<T,N> log(const simdn_t<T,N>& v)
{
	simdn_t<T,N> r;
	std::transform(v.begin(), v.end(), r.begin(), [](const T& a) { return std::log(a); } );
	return r;
}

template<class T, std::size_t N>
inline simdn_t<T,N> log10(const simdn_t<T,N> & v)
{
	simdn_t<T,N> r;
	std::transform(v.begin(), v.end(), r.begin(), [](const T& a) { return std::log10(a); } );
	return r;
}


} //namespace boost::numeric::simd


#if 0
namespace boost::numeric::simd{
namespace simd_function{

template<class T, size_t N>
inline
bool is_all_zero(simdn_t<T,N> && v)
{
	return std::all_of(v.begin(), v.end(), std::bind(std::equal_to<T>(),std::placeholders::_1,T(0)));
}

template<class T, size_t N>
inline
bool is_all_zero(const simdn_t<T,N> & v)
{
	return std::all_of(v.begin(), v.end(), std::bind(std::equal_to<T>(),std::placeholders::_1,T(0)));
}

template<class T, size_t N>
inline
bool is_all_one(simdn_t<T,N> && v)
{
	return std::all_of(v.begin(), v.end(), std::bind(std::equal_to<T>(),std::placeholders::_1,T(1)));
}

template<class T, size_t N>
inline
bool is_all_one(const simdn_t<T,N> & v)
{
	return std::all_of(v.begin(), v.end(), std::bind(std::equal_to<T>(),std::placeholders::_1,T(1)));
}



template<class T, size_t N>
inline
simdn_t<T,N>
sqrt(const simdn_t<T,N>&v)
{
	simdn_t<T,N> res;

	using simd_type = typename simdn_t<T,N>::simd_type;
	std::transform(v.begin(),v.end(), res.begin(), [](const simd_type& a) { return boost::numeric::simd::sqrt(a); } );
	return res;
}


template<class T, size_t N>
inline
simdn_t<T,N>
rsqrt(const simdn_t<T,N>&v)
{
	simdn_t<T,N> res;

	using simd_type = typename simdn_t<T,N>::simd_type;
	std::transform(v.begin(),v.end(), res.begin(), [](const simd_type& a) { return boost::numeric::simd::rsqrt(a); } );
	return res;
}


template<class T, size_t N>
inline
typename simdn_t<T,N>::value_type
sum(const simdn_t<T,N>& v)
{
	using simd_type = typename simdn_t<T,N>::simd_type;
	using value_type = typename simdn_t<T,N>::value_type;
	value_type t = value_type(0);
	for(const simd_type& a : v) t += boost::numeric::simd::sum(a);
	return t;
}

template<class T, size_t N>
inline
typename simdn_t<T,N>::value_type
dot(const simdn_t<T,N>& v, const simdn_t<T,N>& w)
{
	return sum ( v * w );
}

template<class T, size_t N>
inline
typename simdn_t<T,N>::value_type
norm2(const simdn_t<T,N> & v)
{
	return std::sqrt( dot ( v , v ) );
}

template<class T, size_t N>
inline
simdn_t<T,N>
min(const simdn_t<T,N> & v, const simdn_t<T,N> & w)
{
	simdn_t<T,N> r;
	for(auto i = 0; i < N; ++i) r[i] = v[i] < w[i];
	return r;
}

template<class T, size_t N>
inline
simdn_t<T,N>
max(const simdn_t<T,N> & v, const simdn_t<T,N> & w)
{
	simdn_t<T,N> r;
	for(auto i = 0; i < N; ++i) r[i] = v[i] > w[i];
	return r;
}

template<class T, size_t N>
inline
simdn_t<T,N>
ceil(const simdn_t<T,N> & v)
{
	simdn_t<T,N> res;
	return std::transform(v.begin(),v.end(), res.begin(), [](const T& a) { return boost::numeric::simd::ceil(a); } );
}


template<class T, size_t N>
inline
simdn_t<T,N>
floor(const simdn_t<T,N> & v)
{
	simdn_t<T,N> res;
	return std::transform(v.begin(),v.end(), res.begin(), [](const T& a) { return boost::numeric::simd::floor(a); } );
}

template<class T, size_t N>
inline
simdn_t<T,N>
round(const simdn_t<T,N> & v)
{
	simdn_t<T,N> res;
	return std::transform(v.begin(),v.end(), res.begin(), [](const T& a) { return boost::numeric::simd::round(a); } );
}


template<class T, size_t N>
inline
simdn_t<T,N>
pow(const simdn_t<T,N> & x, const simdn_t<T,N>& y)
{
	simdn_t<T,N> r;
	std::transform(x.begin(), x.end(), y.begin(), r.begin(), [](const T& a, const T& b) { return boost::numeric::simd::pow(a,b); } );
	return r;
}

template<class T, size_t N>
inline
simdn_t<T,N>
pow(const simdn_t<T,N> & x, const typename simdn_t<T,N>::value_type& y)
{
	simdn_t<T,N> r;
	std::transform(x.begin(), x.end(), r.begin(), [y](const T& a) { return boost::numeric::simd::pow(a,y); } );
	return r;
}


template<class T, size_t N>
inline
simdn_t<T,N>
sin(const simdn_t<T,N> & v)
{
	simdn_t<T,N> res;
	std::transform(v.begin(),v.end(), res.begin(), [](const T& a) { return boost::numeric::simd::sin(a); } );
	return res;
}


template<class T, size_t N>
inline
simdn_t<T,N>
cos(const simdn_t<T,N> & v)
{
	simdn_t<T,N> res;
	std::transform(v.begin(),v.end(), res.begin(), [](const T& a) { return boost::numeric::simd::cos(a); } );
	return res;
}



// return sin, argument cos
template<class T, size_t N>
simdn_t<T,N>
inline sincos(simdn_t<T,N> & c, const simdn_t<T,N> & x)
{
    c = cos(x);
    return sin(x);
}



template<class T, size_t N>
inline
simdn_t<T,N>
tan(const simdn_t<T,N> & v)
{
	simdn_t<T,N> res;
	std::transform(v.begin(),v.end(), res.begin(), [](const T& a) { return boost::numeric::simd::tan(a); } );
	return res;
}

template<class T, size_t N>
inline
simdn_t<T,N>
exp(const simdn_t<T,N> & v)
{
	simdn_t<T,N> res;
	std::transform(v.begin(),v.end(), res.begin(), [](const T& a) { return boost::numeric::simd::exp(a); } );
	return res;
}

template<class T, size_t N>
inline
simdn_t<T,N>
log(const simdn_t<T,N> & v)
{
	simdn_t<T,N> res;
	std::transform(v.begin(),v.end(), res.begin(), [](const T& a) { return boost::numeric::simd::log(a); } );
	return res;
}

template<class T, size_t N>
inline
simdn_t<T,N>
log10(const simdn_t<T,N> & v)
{
	simdn_t<T,N> res;
	std::transform(v.begin(),v.end(), res.begin(), [](const T& a) { return boost::numeric::simd::log10(a); } );
	return res;
}


} // namespace simd_function
} // namespace boost::numeric::simd
#endif

#endif // BOOST_NUMERIC_SIMD
