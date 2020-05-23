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

#ifndef BOOST_NUMERIC_SIMD_SSE_H
#define BOOST_NUMERIC_SIMD_SSE_H

#include <ostream>
#include <pmmintrin.h> // see3
#include <smmintrin.h> // sse4

#include "simd_traits.h"


/*! \brief simd_t class for 4-packed single precision floating point operations with sse intrinsics
 *
 * 128-bit - 16-byte - sse - 4 x float
 *
 *
 *  127:96     95:64    63:32      31:0
 * --------- --------- --------- ---------
 *   v[3]      v[2]      v[1]      v[0]
 *
 */

namespace boost::numeric::simd
{

template <class T, class I>
class simd_t;


template<>
class simd_t< float , sse_intrinsics_tag >
{

public:
	using value_type = float;
	using pointer = value_type*;
	using const_pointer = const value_type*;
	using reference  = value_type&;
	using const_reference  = const value_type&;
	using intrinsics_tag = sse_intrinsics_tag;
	using traits = intrinsics_traits<intrinsics_tag>;
	using vector_type = __m128; // __array_type(traits::bits);

	// Constructors
	constexpr explicit simd_t() : _array() {}
	explicit simd_t(const_reference value)    { _array = _mm_set1_ps(value); /*__function(traits::bits, set1, s,   value );*/ }  //
	simd_t(const simd_t& value)        { _array = value._array;}
	simd_t(simd_t&& value)             { _array = value._array; }
	simd_t(const vector_type& value) { _array = value; }
	simd_t(vector_type&& value)      { _array = value; }

	explicit simd_t(const_pointer value) { _array = _mm_load_ps(value); }

	inline void load(const_pointer p) { _array = _mm_load_ps(p); }
	inline void loadu(const_pointer p) { _array = _mm_loadu_ps(p); }
	inline void store(pointer p) { _mm_store_ps(p,_array); }
	inline void storeu(pointer p) { _mm_storeu_ps(p,_array); }
	inline void stream(pointer p) { _mm_stream_ps(p,_array); }

	static inline void fence(void)            { _mm_sfence(); }
	static inline void prefetch_nt(const_pointer p)    { _mm_prefetch (reinterpret_cast<char const*>(p), _MM_HINT_NTA); }
	static inline void prefetch_t0(const_pointer p)    { _mm_prefetch (reinterpret_cast<char const*>(p), _MM_HINT_T0); }
	static inline void prefetch_t1(const_pointer p)    { _mm_prefetch (reinterpret_cast<char const*>(p), _MM_HINT_T1); }
	static inline void prefetch_t2(const_pointer p)    { _mm_prefetch (reinterpret_cast<char const*>(p), _MM_HINT_T2); }


	simd_t(const value_type a, const value_type b, const value_type c, const value_type d) { _array = _mm_setr_ps(a,b,c,d); }

	// Destructor
	~simd_t() = default;

	// Assign
	inline simd_t& operator =(const vector_type& v) { _array = v; return *this; }
	inline simd_t& operator =(const_reference v) { _array = _mm_set1_ps(v); return *this; }
	inline simd_t& operator =(const simd_t & v)  { _array = v._array; return *this; }

	// Arithmetic
	inline simd_t operator-(void) const { return (*this) * (-1); }
	inline simd_t operator+(void) const { return *this; }

	inline void operator+=(const simd_t &  v) { _array = _mm_add_ps(_array,v._array); }
	inline void operator-=(const simd_t &  v) { _array = _mm_sub_ps(_array,v._array); }
	inline void operator*=(const simd_t &  v) { _array = _mm_mul_ps(_array,v._array); }
	inline void operator/=(const simd_t &  v) { _array = _mm_div_ps(_array,v._array); }

	inline simd_t operator+(const simd_t &  v) const { return simd_t(_mm_add_ps(_array,v._array)); }
	inline simd_t operator-(const simd_t &  v) const { return simd_t(_mm_sub_ps(_array,v._array)); }
	inline simd_t operator*(const simd_t &  v) const { return simd_t(_mm_mul_ps(_array,v._array)); }
	inline simd_t operator/(const simd_t &  v) const { return simd_t(_mm_div_ps(_array,v._array)); }

	inline simd_t& operator++() { _array = _mm_add_ps(_array,_mm_set1_ps(1)); return *this;}
	inline simd_t& operator--() { _array = _mm_sub_ps(_array,_mm_set1_ps(1)); return *this;}


	inline void operator+=(const_reference v) { this->operator +=(simd_t(v)); }
	inline void operator-=(const_reference v) { this->operator -=(simd_t(v)); }
	inline void operator*=(const_reference v) { this->operator *=(simd_t(v)); }
	inline void operator/=(const_reference v) { this->operator /=(simd_t(v)); }


	inline simd_t operator+(const_reference v) const { return this->operator +(simd_t(v)); }
	inline simd_t operator-(const_reference v) const { return this->operator -(simd_t(v)); }
	inline simd_t operator*(const_reference v) const { return this->operator *(simd_t(v)); }
	inline simd_t operator/(const_reference v) const { return this->operator /(simd_t(v)); }

	inline bool operator==(const simd_t & v)  const { return (*this)[0] == v[0] && (*this)[1] == v[1] && (*this)[2] == v[2] && (*this)[3] == v[3];}
	inline bool operator!=(const simd_t & v)  const { return (*this)[0] != v[0] && (*this)[1] != v[1] && (*this)[2] != v[2] && (*this)[3] != v[3];}
	inline bool operator> (const simd_t & v)  const { return (*this)[0] >  v[0] && (*this)[1] >  v[1] && (*this)[2] >  v[2] && (*this)[3] >  v[3];}
	inline bool operator>=(const simd_t & v)  const { return (*this)[0] >= v[0] && (*this)[1] >= v[1] && (*this)[2] >= v[2] && (*this)[3] >= v[3];}
	inline bool operator< (const simd_t & v)  const { return (*this)[0] <  v[0] && (*this)[1] <  v[1] && (*this)[2] <  v[2] && (*this)[3] <  v[3];}
	inline bool operator<=(const simd_t & v)  const { return (*this)[0] <= v[0] && (*this)[1] <= v[1] && (*this)[2] <= v[2] && (*this)[3] <= v[3];}

	inline bool operator==(const_reference v)  const { return this->operator ==(simd_t(v)); }
	inline bool operator!=(const_reference v)  const { return this->operator !=(simd_t(v)); }
	inline bool operator> (const_reference v)  const { return this->operator > (simd_t(v)); }
	inline bool operator>=(const_reference v)  const { return this->operator >=(simd_t(v)); }
	inline bool operator< (const_reference v)  const { return this->operator < (simd_t(v)); }
	inline bool operator<=(const_reference v)  const { return this->operator <=(simd_t(v)); }


	// Access, gcc
	//inline reference operator[](int i) { return _array[i]; }
	//inline const_reference operator[](int i) const { return _array[i]; }
	// icc
	inline reference operator[](int i) { return ((value_type*)&_array)[i]; }
	inline const_reference operator[](int i) const { return ((value_type*)&_array)[i]; }


	friend
	std::ostream &operator<<(std::ostream & s, const simd_t & v) {// Component-wise output stream
		for (size_t i=0; i<_size; i++) s << v[i] << ' ';
		return s;
	}

	inline const vector_type& data() const
	{
		return this->_array;
	}

	static constexpr size_t size()
	{
		return _size;
	}

private:
	vector_type _array;
	static constexpr size_t _size = traits::bytes / sizeof(value_type) ;
};

} // end namespace


////////////////////////////
///// Functions Float<4>
////////////////////////////

namespace boost::numeric::simd
{


inline
simd_t< float , sse_intrinsics_tag >
operator+(simd_t< float , sse_intrinsics_tag >::const_reference r, const simd_t< float , sse_intrinsics_tag >& vec)
{
	return vec.operator +(r);
}

inline
simd_t< float , sse_intrinsics_tag >
operator-(simd_t< float , sse_intrinsics_tag >::const_reference r, const simd_t< float , sse_intrinsics_tag >& vec)
{
	return simd_t< float , sse_intrinsics_tag >(r) - vec;
}


inline
simd_t< float , sse_intrinsics_tag >
operator*(simd_t< float , sse_intrinsics_tag >::const_reference r, const simd_t< float , sse_intrinsics_tag >& vec)
{
	return vec.operator *(r);
}

inline simd_t< float , sse_intrinsics_tag >
operator/(simd_t< float , sse_intrinsics_tag >::const_reference r, const simd_t< float , sse_intrinsics_tag >& vec)
{
	return simd_t< float , sse_intrinsics_tag >(r) / vec;
}

} // end namespace




///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////


namespace boost::numeric::simd
{

/*! \brief simd_t class for 2-packed double precision floating point operations with sse intrinsics
 *
 * 128-bit - 16-byte - sse - 2 x double
 *
 *
 *  127:63     63:32
 * --------- ---------
 *   v[1]      v[0]
 *
 */
template<>
class simd_t< double , sse_intrinsics_tag >
{

public:
	//using vector_type = __m128; // typedef struct __declspec(align(16)) { float f[4]; }

	using value_type = double;
	using pointer = value_type*;
	using const_pointer = const value_type*;
	using reference  = value_type&;
	using const_reference  = const value_type&;
	using intrinsics_tag = sse_intrinsics_tag;
	using traits = intrinsics_traits<intrinsics_tag>;
	using vector_type = __m128d; // __array_type(traits::bits);

	// Constructors
	constexpr explicit simd_t() : _array() {}
	explicit simd_t(const_reference value)    { _array = _mm_set1_pd(value); /*__function(traits::bits, set1, s,   value );*/ }  //
	simd_t(const simd_t& value)     { _array = value._array;}
	simd_t(const vector_type& value) { _array = value; }

	explicit simd_t(const_pointer value) { _array = _mm_load_pd(value); }

	inline void load(const_pointer p) { _array = _mm_load_pd(p); }
	inline void loadu(const_pointer p) { _array = _mm_loadu_pd(p); }
	inline void store(pointer p) { _mm_store_pd(p,_array); }
	inline void storeu(pointer p) { _mm_storeu_pd(p,_array); }
	inline void stream(pointer p) { _mm_stream_pd(p,_array); }

	static inline void fence(void)            { _mm_sfence(); }
	static inline void prefetch_nt(const_pointer p)    { _mm_prefetch (reinterpret_cast<char const*>(p), _MM_HINT_NTA); }
	static inline void prefetch_t0(const_pointer p)    { _mm_prefetch (reinterpret_cast<char const*>(p), _MM_HINT_T0); }
	static inline void prefetch_t1(const_pointer p)    { _mm_prefetch (reinterpret_cast<char const*>(p), _MM_HINT_T1); }
	static inline void prefetch_t2(const_pointer p)    { _mm_prefetch (reinterpret_cast<char const*>(p), _MM_HINT_T2); }

	// todo hier
	simd_t(const value_type a, const value_type b) { _array = _mm_setr_pd(a,b); }

	// Destructor
	~simd_t() = default;

	// Assign
	inline simd_t& operator =(const vector_type& v) { _array = v; return *this; }
	inline simd_t& operator =(const_reference v) { _array = _mm_set1_pd(v); return *this; }
	inline simd_t& operator =(const simd_t & v)  { _array = v._array; return *this; }


	// Arithmetic
	inline simd_t operator-(void) const { return (*this) * (-1); }
	inline simd_t operator+(void) const { return *this; }

	inline void operator+=(const simd_t &  v) { _array = _mm_add_pd(_array,v._array); }
	inline void operator-=(const simd_t &  v) { _array = _mm_sub_pd(_array,v._array); }
	inline void operator*=(const simd_t &  v) { _array = _mm_mul_pd(_array,v._array); }
	inline void operator/=(const simd_t &  v) { _array = _mm_div_pd(_array,v._array); }

	inline simd_t& operator++() { _array = _mm_add_pd(_array,_mm_set1_pd(1)); return *this;}
	inline simd_t& operator--() { _array = _mm_sub_pd(_array,_mm_set1_pd(1)); return *this;}


	inline simd_t operator+(const simd_t &  v) const { return simd_t(_mm_add_pd(_array,v._array)); }
	inline simd_t operator-(const simd_t &  v) const { return simd_t(_mm_sub_pd(_array,v._array)); }
	inline simd_t operator*(const simd_t &  v) const { return simd_t(_mm_mul_pd(_array,v._array)); }
	inline simd_t operator/(const simd_t &  v) const { return simd_t(_mm_div_pd(_array,v._array)); }

	inline void operator+=(const_reference v) { this->operator +=(simd_t(v)); }
	inline void operator-=(const_reference v) { this->operator -=(simd_t(v)); }
	inline void operator*=(const_reference v) { this->operator *=(simd_t(v)); }
	inline void operator/=(const_reference v) { this->operator /=(simd_t(v)); }


	inline simd_t operator+(const_reference v) const { return this->operator +(simd_t(v)); }
	inline simd_t operator-(const_reference v) const { return this->operator -(simd_t(v)); }
	inline simd_t operator*(const_reference v) const { return this->operator *(simd_t(v)); }
	inline simd_t operator/(const_reference v) const { return this->operator /(simd_t(v)); }


	// Compare


	inline bool operator==(const simd_t & v)  const { return (*this)[0] == v[0] && (*this)[1] == v[1];}
	inline bool operator!=(const simd_t & v)  const { return (*this)[0] != v[0] && (*this)[1] != v[1];}
	inline bool operator> (const simd_t & v)  const { return (*this)[0] >  v[0] && (*this)[1] >  v[1];}
	inline bool operator>=(const simd_t & v)  const { return (*this)[0] >= v[0] && (*this)[1] >= v[1];}
	inline bool operator< (const simd_t & v)  const { return (*this)[0] <  v[0] && (*this)[1] <  v[1];}
	inline bool operator<=(const simd_t & v)  const { return (*this)[0] <= v[0] && (*this)[1] <= v[1];}



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

	inline const vector_type& data() const
	{
		return this->_array;
	}

	static constexpr size_t size()
	{
		return _size;
	}

private:
	vector_type _array;
	static constexpr size_t _size = traits::bytes / sizeof(value_type) ;
};

} // end namespace

////////////////////////////
/// Functions Double2
////////////////////////////
namespace boost::numeric::simd
{

inline
simd_t< double , sse_intrinsics_tag >
operator+(simd_t< double , sse_intrinsics_tag >::const_reference r, const simd_t< double , sse_intrinsics_tag >& vec)
{
	return vec.operator +(r);
}

inline
simd_t< double , sse_intrinsics_tag >
operator-(simd_t< double , sse_intrinsics_tag >::const_reference r, const simd_t< double , sse_intrinsics_tag >& vec)
{
	return simd_t< double , sse_intrinsics_tag >(r) - vec;
}


inline
simd_t< double , sse_intrinsics_tag >
operator*(simd_t< double , sse_intrinsics_tag >::const_reference r, const simd_t< double , sse_intrinsics_tag >& vec)
{
	return vec.operator *(r);
}

inline simd_t< double , sse_intrinsics_tag >
operator/(simd_t< double , sse_intrinsics_tag >::const_reference r, const simd_t< double , sse_intrinsics_tag >& vec)
{
	return simd_t< double , sse_intrinsics_tag >(r) / vec;
}

} // namespace boost::numeric::simd


#endif // BOOST_NUMERIC_SIMD_SSE_H
