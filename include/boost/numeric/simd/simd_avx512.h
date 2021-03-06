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

#ifndef TLIB_SIMD_AVX512_H
#define TLIB_SIMD_AVX512_H

#include "simd_traits.h"

#include <ostream>

#include <immintrin.h> // avx, mic
#include <pmmintrin.h> // see3
#include <smmintrin.h> // sse4



/*! \brief simd_t class for 8-packed single precision floating point operations with avx intrinsics
 *
 * 512-bit - 64-byte - avx512 - 16 x float
 *
 *
 * --------- --------- --------- --------- --------- --------- --------- ---------
 *   v[15]      v[14]      v[13]      v[12]      v[11]      v[10]      v[9]      v[8]
 *
 *  255:224   223:192   191:160   159:128   127:96     95:64     63:32     31:0
 * --------- --------- --------- --------- --------- --------- --------- ---------
 *   v[7]      v[6]      v[5]      v[4]      v[3]      v[2]      v[1]      v[0]
 *
 */
namespace boost::numeric::simd 
{

template<class V, class I>
class simd_t;

template<>
class simd_t< float , avx512_intrinsics_tag >
{
public:
    //using vector_type = __m512; // typedef struct __declspec(align(64)) { float f[16]; }

    using value_type = float;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference  = value_type&;
    using const_reference  = const value_type&;
    using intrinsics_tag = avx512_intrinsics_tag;
    using traits = intrinsics_traits<intrinsics_tag>;
    using vector_type = __m512;
    using half_vector_type = __m256;
    using quarter_vector_type = __m128;

    // Constructors
	constexpr explicit simd_t()
        : _array()
    {
    }
	explicit simd_t(const_reference value)
        : _array{_mm512_set1_ps(value)}
    {
         /*__function(traits::bits, set1, s,   value );*/
    }

	simd_t(const simd_t& value)  {
        _array[0] = value._array[0]; _array[1] = value._array[1]; _array[2] = value._array[2]; _array[3] = value._array[3];
        _array[4] = value._array[4]; _array[5] = value._array[5]; _array[6] = value._array[6]; _array[7] = value._array[7];

        _array[8] = value._array[8]; _array[9] = value._array[9]; _array[10] = value._array[10]; _array[11] = value._array[11];
        _array[12] = value._array[12]; _array[13] = value._array[13]; _array[14] = value._array[14]; _array[15] = value._array[15];
    }

	simd_t(const vector_type& value) { _array = value;}
	simd_t(const value_type a, const value_type b, const value_type c, const value_type d,
         const value_type e, const value_type f, const value_type g, const value_type h,
         const value_type i, const value_type j, const value_type k, const value_type l,
         const value_type m, const value_type n, const value_type o, const value_type p)
    {
        _array = _mm512_setr_ps(a,b,c,d,e,f,g,h,  i,j,k,l,m,n,o,p );

    }

	explicit simd_t(const_pointer value) { _array = _mm512_load_ps(value); }

    // Destructor
	~simd_t() = default;

    inline void setLow(const half_vector_type& low) {  this->_array = _mm512_insertf32x8(this->_array, low, 0); }
    inline void setHigh(const half_vector_type& high) { this->_array = _mm512_insertf32x8(this->_array, high, 1); }

//    inline void setLow(const half_vector_type& low) {  this->_array = _mm256_insertf128_ps(this->_array, low, 0); }
//    inline void setHigh(const half_vector_type& high) { this->_array = _mm256_insertf128_ps(this->_array, high, 1); }


	inline void load  (const_pointer p)  { _array = _mm512_load_ps(p);  }
	inline void loadu (const_pointer p) { _array = _mm512_loadu_ps(p); }
	inline void store (pointer p)       { _mm512_store_ps(p,_array);   }
    inline void storeu(pointer p)      { _mm512_storeu_ps(p,_array);  }
    inline void stream(pointer p)      { _mm512_stream_ps(p,_array);  }
    static inline void fence(void)            { _mm_sfence(); }

    static inline void prefetch_nt(const_pointer p)    { _mm_prefetch (reinterpret_cast<char const*>(p), _MM_HINT_NTA); }
    static inline void prefetch_t0(const_pointer p)    { _mm_prefetch (reinterpret_cast<char const*>(p), _MM_HINT_T0); }
    static inline void prefetch_t1(const_pointer p)    { _mm_prefetch (reinterpret_cast<char const*>(p), _MM_HINT_T1); }
    static inline void prefetch_t2(const_pointer p)    { _mm_prefetch (reinterpret_cast<char const*>(p), _MM_HINT_T2); }

    inline half_vector_type low() const { return _mm512_extractf32x8_ps(this->_array,0); }
    inline half_vector_type high() const { return _mm512_extractf32x8_ps(this->_array,1); }

    // Assign
	inline simd_t& operator =(const vector_type& v) { _array = v; return *this; }
	inline simd_t& operator =(const_reference v) { _array = _mm512_set1_ps(v); return *this; }
	inline simd_t& operator =(const simd_t & v)  { _array = v._array; return *this; }

    // Arithmetic
	inline simd_t operator-(void) const { return (*this) * (-1); }
	inline simd_t operator+(void) const { return *this; }

	inline void operator+=(const simd_t &  v) { _array = _mm512_add_ps(_array,v._array); }
	inline void operator-=(const simd_t &  v) { _array = _mm512_sub_ps(_array,v._array); }
	inline void operator*=(const simd_t &  v) { _array = _mm512_mul_ps(_array,v._array); }
	inline void operator/=(const simd_t &  v) { _array = _mm512_div_ps(_array,v._array); }

	inline simd_t& operator++() { _array = _mm512_add_ps(_array,_mm512_set1_ps(1)); return *this;}
	inline simd_t& operator--() { _array = _mm512_sub_ps(_array,_mm512_set1_ps(1)); return *this;}


	inline simd_t operator+(const simd_t &  v) const { return simd_t(_mm512_add_ps(_array,v._array)); }
	inline simd_t operator-(const simd_t &  v) const { return simd_t(_mm512_sub_ps(_array,v._array)); }
	inline simd_t operator*(const simd_t &  v) const { return simd_t(_mm512_mul_ps(_array,v._array)); }
	inline simd_t operator/(const simd_t &  v) const { return simd_t(_mm512_div_ps(_array,v._array)); }

	inline void operator+=(const_reference v) { this->operator +=(simd_t(v)); }
	inline void operator-=(const_reference v) { this->operator -=(simd_t(v)); }
	inline void operator*=(const_reference v) { this->operator *=(simd_t(v)); }
	inline void operator/=(const_reference v) { this->operator /=(simd_t(v)); }


	inline simd_t operator+(const_reference v) const { return this->operator +(simd_t(v)); }
	inline simd_t operator-(const_reference v) const { return this->operator -(simd_t(v)); }
	inline simd_t operator*(const_reference v) const { return this->operator *(simd_t(v)); }
	inline simd_t operator/(const_reference v) const { return this->operator /(simd_t(v)); }

    // Compare
	inline bool operator==(const simd_t & v)  const { return (*this)[0] == v[0] && (*this)[1] == v[1] && (*this)[2] == v[2] && (*this)[3] == v[3] && (*this)[4] == v[4] && (*this)[5] == v[5] && (*this)[6] == v[6] && (*this)[7] == v[7];}
	inline bool operator!=(const simd_t & v)  const { return (*this)[0] != v[0] && (*this)[1] != v[1] && (*this)[2] != v[2] && (*this)[3] != v[3] && (*this)[4] != v[4] && (*this)[5] != v[5] && (*this)[6] != v[6] && (*this)[7] != v[7];}
	inline bool operator> (const simd_t & v)  const { return (*this)[0] >  v[0] && (*this)[1] >  v[1] && (*this)[2] >  v[2] && (*this)[3] >  v[3] && (*this)[4] >  v[4] && (*this)[5] >  v[5] && (*this)[6] >  v[6] && (*this)[7] >  v[7];}
	inline bool operator>=(const simd_t & v)  const { return (*this)[0] >= v[0] && (*this)[1] >= v[1] && (*this)[2] >= v[2] && (*this)[3] >= v[3] && (*this)[4] >= v[4] && (*this)[5] >= v[5] && (*this)[6] >= v[6] && (*this)[7] >= v[7];}
	inline bool operator< (const simd_t & v)  const { return (*this)[0] <  v[0] && (*this)[1] <  v[1] && (*this)[2] <  v[2] && (*this)[3] <  v[3] && (*this)[4] <  v[4] && (*this)[5] <  v[5] && (*this)[6] <  v[6] && (*this)[7] <  v[7];}
	inline bool operator<=(const simd_t & v)  const { return (*this)[0] <= v[0] && (*this)[1] <= v[1] && (*this)[2] <= v[2] && (*this)[3] <= v[3] && (*this)[4] <= v[4] && (*this)[5] <= v[5] && (*this)[6] <= v[6] && (*this)[7] <= v[7];}

	inline bool operator==(const_reference v)  const { return this->operator ==(simd_t(v)); }
	inline bool operator!=(const_reference v)  const { return this->operator !=(simd_t(v)); }
	inline bool operator> (const_reference v)  const { return this->operator > (simd_t(v)); }
	inline bool operator>=(const_reference v)  const { return this->operator >=(simd_t(v)); }
	inline bool operator< (const_reference v)  const { return this->operator < (simd_t(v)); }
	inline bool operator<=(const_reference v)  const { return this->operator <=(simd_t(v)); }


    // Access
//	inline reference operator[](int i) { return _array[i]; }
//	inline const_reference operator[](int i) const { return _array[i]; }
    // icc
    inline reference operator[](int i) { return ((value_type*)&_array)[i]; }
    inline const_reference operator[](int i) const { return ((value_type*)&_array)[i]; }

    friend
	std::ostream &operator<<(std::ostream & s, const simd_t & v) {// Component-wise output stream
        for (size_t i=0; i<_size; i++) s << v[i] << ' ';
        return s;
    }

    inline const vector_type& data() const { return this->_array; }

    inline static constexpr size_t size() { return _size; }

private:
    vector_type _array;
    static constexpr size_t _size = traits::bytes / sizeof(value_type) ;
};
}

//////////////////////////
/// Functions Float<16>
//////////////////////////

namespace boost::numeric::simd 
{



inline
simd_t< float , avx512_intrinsics_tag >
operator+(simd_t< float , avx512_intrinsics_tag >::const_reference r, const simd_t< float , avx512_intrinsics_tag >& vec)
{
    return vec.operator +(r);
}

inline
simd_t< float , avx512_intrinsics_tag >
operator-(simd_t< float , avx512_intrinsics_tag >::const_reference r, const simd_t< float , avx512_intrinsics_tag >& vec)
{
	return simd_t< float , avx512_intrinsics_tag >(r) - vec;
}


inline
simd_t< float , avx512_intrinsics_tag >
operator*(simd_t< float , avx512_intrinsics_tag >::const_reference r, const simd_t< float , avx512_intrinsics_tag >& vec)
{
    return vec.operator *(r);
}

inline simd_t< float , avx512_intrinsics_tag >
operator/(simd_t< float , avx512_intrinsics_tag >::const_reference r, const simd_t< float , avx512_intrinsics_tag >& vec)
{
	return simd_t< float , avx512_intrinsics_tag >(r)/vec;
}

}





namespace boost::numeric::simd 
{


//#define __function(bits,operation,precision,...)   _mm##bits_##operation_p##precision(__VA_ARGS__)
//#define __array_type( bits ) __m##bits

/*! \brief simd_t class for 8-packed single precision floating point operations with avx intrinsics
 *
 * 256-bit - 32-byte - avx - 8 x float
 *
 *
 *  255:224   223:192   191:160   159:128   127:96     95:64     63:32     31:0
 * --------- --------- --------- --------- --------- --------- --------- ---------
 *   v[7]      v[6]      v[5]      v[4]      v[3]      v[2]      v[1]      v[0]
 *
 */
template<>
class simd_t< double, avx512_intrinsics_tag >
{
public:
    //using vector_type = __m512d; // typedef struct __declspec(align(64)) { double f[8]; }

    using value_type = double;
    using pointer = value_type*;
    using const_pointer = const value_type*;
    using reference  = value_type&;
    using const_reference  = const value_type&;
    using intrinsics_tag = avx512_intrinsics_tag;
    using traits = intrinsics_traits<intrinsics_tag>;
    using vector_type = __m512d;
    using half_vector_type = __m256d;
    using quarter_vector_type = __m128d;

    // Constructors
	constexpr explicit simd_t() : _array() { }
	explicit simd_t(const_reference value)    { _array = _mm512_set1_pd(value); /*__function(traits::bits, set1, s,   value );*/ }  //

	simd_t(const simd_t& value)  {
        _array[0] = value._array[0]; _array[1] = value._array[1]; _array[2] = value._array[2]; _array[3] = value._array[3];
        _array[4] = value._array[4]; _array[5] = value._array[5]; _array[6] = value._array[6]; _array[7] = value._array[7];
    }
	simd_t(const vector_type& value) { _array = value;}
	simd_t(const value_type a, const value_type b, const value_type c, const value_type d,
            const value_type e, const value_type f, const value_type g, const value_type h) { _array = _mm512_setr_pd(a,b,c,d,e,f,g,h); }

	explicit simd_t(const_pointer value) { _array = _mm512_load_pd(value); }

    // Destructor
	~simd_t() = default;

    inline void setLow(const half_vector_type& low) {  this->_array = _mm512_insertf64x4(this->_array, low, 0); }
    inline void setHigh(const half_vector_type& high) { this->_array = _mm512_insertf64x4(this->_array, high, 1); }


    inline void load(const_pointer p)  { _array = _mm512_load_pd(p);  }
    inline void loadu(const_pointer p) { _array = _mm512_loadu_pd(p); }
    inline void store(pointer p)       { _mm512_store_pd(p,_array);   }
    inline void storeu(pointer p)      { _mm512_storeu_pd(p,_array);  }
    inline void stream(pointer p)      { _mm512_stream_pd(p,_array);  }
    static inline void fence(void)            { _mm_sfence(); }

    static inline void prefetch_nt(const_pointer p)    { _mm_prefetch (reinterpret_cast<char const*>(p), _MM_HINT_NTA); }
    static inline void prefetch_t0(const_pointer p)    { _mm_prefetch (reinterpret_cast<char const*>(p), _MM_HINT_T0); }
    static inline void prefetch_t1(const_pointer p)    { _mm_prefetch (reinterpret_cast<char const*>(p), _MM_HINT_T1); }
    static inline void prefetch_t2(const_pointer p)    { _mm_prefetch (reinterpret_cast<char const*>(p), _MM_HINT_T2); }

    // {v[3:0]}
    inline half_vector_type low() const { return _mm512_extractf64x4_pd(this->_array,0); }
    // {v[7:4]}
    inline half_vector_type high() const { return _mm512_extractf64x4_pd(this->_array,1); }

    // Assign
	inline simd_t& operator =(const vector_type& v) { _array = v; return *this; }
	inline simd_t& operator =(const_reference v) { _array = _mm512_set1_pd(v); return *this; }
	inline simd_t& operator =(const simd_t & v)  { _array = v._array; return *this; }

    // Arithmetic
	inline simd_t operator-(void) const { return (*this) * (-1); }
	inline simd_t operator+(void) const { return *this; }

	inline void operator+=(const simd_t &  v) { _array = _mm512_add_pd(_array,v._array); }
	inline void operator-=(const simd_t &  v) { _array = _mm512_sub_pd(_array,v._array); }
	inline void operator*=(const simd_t &  v) { _array = _mm512_mul_pd(_array,v._array); }
	inline void operator/=(const simd_t &  v) { _array = _mm512_div_pd(_array,v._array); }

	inline simd_t& operator++() { _array = _mm512_add_pd(_array,_mm512_set1_pd(1)); return *this;}
	inline simd_t& operator--() { _array = _mm512_sub_pd(_array,_mm512_set1_pd(1)); return *this;}


	inline simd_t operator+(const simd_t &  v) const { return simd_t(_mm512_add_pd(_array,v._array)); }
	inline simd_t operator-(const simd_t &  v) const { return simd_t(_mm512_sub_pd(_array,v._array)); }
	inline simd_t operator*(const simd_t &  v) const { return simd_t(_mm512_mul_pd(_array,v._array)); }
	inline simd_t operator/(const simd_t &  v) const { return simd_t(_mm512_div_pd(_array,v._array)); }

	inline void operator+=(const_reference v) { this->operator +=(simd_t(v)); }
	inline void operator-=(const_reference v) { this->operator -=(simd_t(v)); }
	inline void operator*=(const_reference v) { this->operator *=(simd_t(v)); }
	inline void operator/=(const_reference v) { this->operator /=(simd_t(v)); }


	inline simd_t operator+(const_reference v) const { return this->operator +(simd_t(v)); }
	inline simd_t operator-(const_reference v) const { return this->operator -(simd_t(v)); }
	inline simd_t operator*(const_reference v) const { return this->operator *(simd_t(v)); }
	inline simd_t operator/(const_reference v) const { return this->operator /(simd_t(v)); }

    // Compare
	inline bool operator==(const simd_t & v)  const { return (*this)[0] == v[0] && (*this)[1] == v[1] && (*this)[2] == v[2] && (*this)[3] == v[3] && (*this)[4] == v[4] && (*this)[5] == v[5] && (*this)[6] == v[6] && (*this)[7] == v[7];}
	inline bool operator!=(const simd_t & v)  const { return (*this)[0] != v[0] && (*this)[1] != v[1] && (*this)[2] != v[2] && (*this)[3] != v[3] && (*this)[4] != v[4] && (*this)[5] != v[5] && (*this)[6] != v[6] && (*this)[7] != v[7];}
	inline bool operator> (const simd_t & v)  const { return (*this)[0] >  v[0] && (*this)[1] >  v[1] && (*this)[2] >  v[2] && (*this)[3] >  v[3] && (*this)[4] >  v[4] && (*this)[5] >  v[5] && (*this)[6] >  v[6] && (*this)[7] >  v[7];}
	inline bool operator>=(const simd_t & v)  const { return (*this)[0] >= v[0] && (*this)[1] >= v[1] && (*this)[2] >= v[2] && (*this)[3] >= v[3] && (*this)[4] >= v[4] && (*this)[5] >= v[5] && (*this)[6] >= v[6] && (*this)[7] >= v[7];}
	inline bool operator< (const simd_t & v)  const { return (*this)[0] <  v[0] && (*this)[1] <  v[1] && (*this)[2] <  v[2] && (*this)[3] <  v[3] && (*this)[4] <  v[4] && (*this)[5] <  v[5] && (*this)[6] <  v[6] && (*this)[7] <  v[7];}
	inline bool operator<=(const simd_t & v)  const { return (*this)[0] <= v[0] && (*this)[1] <= v[1] && (*this)[2] <= v[2] && (*this)[3] <= v[3] && (*this)[4] <= v[4] && (*this)[5] <= v[5] && (*this)[6] <= v[6] && (*this)[7] <= v[7];}

	inline bool operator==(const_reference v)  const { return this->operator ==(simd_t(v)); }
	inline bool operator!=(const_reference v)  const { return this->operator !=(simd_t(v)); }
	inline bool operator> (const_reference v)  const { return this->operator > (simd_t(v)); }
	inline bool operator>=(const_reference v)  const { return this->operator >=(simd_t(v)); }
	inline bool operator< (const_reference v)  const { return this->operator < (simd_t(v)); }
	inline bool operator<=(const_reference v)  const { return this->operator <=(simd_t(v)); }


    // Access
//	inline reference operator[](int i) { return _array[i]; }
//	inline const_reference operator[](int i) const { return _array[i]; }
    // icc
    inline reference operator[](int i) { return ((value_type*)&_array)[i]; }
    inline const_reference operator[](int i) const { return ((value_type*)&_array)[i]; }

    friend
	std::ostream &operator<<(std::ostream & s, const simd_t & v) {// Component-wise output stream
        for (size_t i=0; i<_size; i++) s << v[i] << ' ';
        return s;
    }

    inline const vector_type& data() const { return this->_array; }

    inline static constexpr size_t size() { return _size; }

private:
    vector_type _array;
    static constexpr size_t _size = traits::bytes / sizeof(value_type) ;
};

}


//////////////////////////
/// Functions Double<8>
//////////////////////////

namespace boost::numeric::simd 
{



inline
simd_t< double , avx512_intrinsics_tag >
operator+(simd_t< double , avx512_intrinsics_tag >::const_reference r, const simd_t< double , avx512_intrinsics_tag >& vec)
{
    return vec.operator +(r);
}

inline
simd_t< double , avx512_intrinsics_tag >
operator-(simd_t< double , avx512_intrinsics_tag >::const_reference r, const simd_t< double , avx512_intrinsics_tag >& vec)
{
	return simd_t< double , avx512_intrinsics_tag >(r) - vec;
}


inline
simd_t< double , avx512_intrinsics_tag >
operator*(simd_t< double , avx512_intrinsics_tag >::const_reference r, const simd_t< double , avx512_intrinsics_tag >& vec)
{
    return vec.operator *(r);
}

inline simd_t< double , avx512_intrinsics_tag >
operator/(simd_t< double , avx512_intrinsics_tag >::const_reference r, const simd_t< double , avx512_intrinsics_tag >& vec)
{
	return simd_t< double , avx512_intrinsics_tag >(r)/vec;
}

}

#endif


