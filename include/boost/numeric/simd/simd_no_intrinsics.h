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

#ifndef BOOST_NUMERIC_SIMD_N_H
#define BOOST_NUMERIC_SIMD_N_H

#include "simd_traits.h"

#include <ostream>
#include <iostream>
#include <cmath>
#include <cassert>
#include <array>
#include <initializer_list>
#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include <complex>
#include <type_traits>

////////////////////////////////////////////////////////
////////////////////////////////////////////////////////
////////////////////////////////////////////////////////

namespace boost::numeric::simd
{


template<class value_type_ , std::size_t N_>
class simdn_t
{
public:
	using intrinsics_tag = no_intrinsics_tag;
	using value_type = value_type_;

	static constexpr std::size_t N = N_;

	using base_type  = std::array<value_type, N>;

	using       reference = typename base_type::reference;
	using const_reference = typename base_type::const_reference;

	using       pointer = typename base_type::pointer;
	using const_pointer = typename base_type::const_pointer;

	using       iterator = typename base_type::iterator;
	using const_iterator = typename base_type::const_iterator;


	constexpr explicit simdn_t() : _array() {}
	explicit simdn_t(const_pointer pvalue) : _array() { std::copy(pvalue, pvalue+N, this->begin()); }

	inline void load(const_pointer p) { std::copy(p, p+N, this->begin()); }

	inline void loadu(const_pointer p) { load(p); }
	inline void store(pointer p)       { std::copy(this->begin(), this->end(), p);   }
	inline void storeu(pointer p)      { store(p);  }
	inline void stream(pointer p)      { store(p);  }
	static inline void fence(void)     { }


	explicit simdn_t(value_type value) : _array() { std::fill(this->begin(), this->end(), value); }

	simdn_t(const base_type& other) : _array(other) {}

	simdn_t(const simdn_t& other) : _array(other._array) {}
	simdn_t(simdn_t&& other) : _array(std::move(other._array)) {}

	template<class ... ValueTypes>
	explicit simdn_t(value_type const& v, ValueTypes&& ... types)
		: _array()
	{
		Setter<value_type, ValueTypes...>::set(this->_array, v, std::move(types)...);
//		set2(v1, v2, std::move(types)...);
	}

	inline simdn_t& operator=(value_type value) { std::fill(this->begin(), this->end(), value); return *this;}
	inline simdn_t& operator=(const simdn_t& other) { _array = other._array; return *this;}
	inline simdn_t& operator=(simdn_t&& other) { _array = std::move(other._array); return *this;}




	inline bool operator==(const simdn_t& other) const { return std::equal(this->begin(), this->end(), other.begin(), std::equal_to     <value_type>()); }
	inline bool operator!=(const simdn_t& other) const { return std::equal(this->begin(), this->end(), other.begin(), std::not_equal_to <value_type>()); }
	inline bool operator<=(const simdn_t& other) const { return std::equal(this->begin(), this->end(), other.begin(), std::less_equal   <value_type>()); }
	inline bool operator< (const simdn_t& other) const { return std::equal(this->begin(), this->end(), other.begin(), std::less         <value_type>()); }
	inline bool operator>=(const simdn_t& other) const { return std::equal(this->begin(), this->end(), other.begin(), std::greater_equal<value_type>()); }
	inline bool operator> (const simdn_t& other) const { return std::equal(this->begin(), this->end(), other.begin(), std::greater      <value_type>()); }

	inline bool operator==(const_reference value) const { return this->operator ==(simdn_t(value));}
	inline bool operator!=(const_reference value) const { return this->operator !=(simdn_t(value));}
	inline bool operator<=(const_reference value) const { return this->operator <=(simdn_t(value));}
	inline bool operator< (const_reference value) const { return this->operator < (simdn_t(value));}
	inline bool operator>=(const_reference value) const { return this->operator >=(simdn_t(value));}
	inline bool operator> (const_reference value) const { return this->operator > (simdn_t(value));}


	inline simdn_t operator-(void) const { return (*this) * (-1); }
	inline simdn_t operator+(void) const { return *this; }
	inline simdn_t operator*(value_type v) const { simdn_t res; std::transform(this->begin(), this->end(), res.begin(), std::bind(std::multiplies<value_type>(),std::placeholders::_1, v  )); return res;}
	inline simdn_t operator/(value_type v) const { simdn_t res; std::transform(this->begin(), this->end(), res.begin(), std::bind(std::multiplies<value_type>(),std::placeholders::_1, 1/v)); return res;}
	inline simdn_t operator+(value_type v) const { simdn_t res; std::transform(this->begin(), this->end(), res.begin(), std::bind(std::plus      <value_type>(),std::placeholders::_1, v  )); return res;}
	inline simdn_t operator-(value_type v) const { simdn_t res; std::transform(this->begin(), this->end(), res.begin(), std::bind(std::minus     <value_type>(),std::placeholders::_1, v  )); return res;}

	inline void operator*=(value_type v) { std::for_each(this->begin(),this->end(), [v](reference a) { a*=v; }); }
	inline void operator/=(value_type v) { std::for_each(this->begin(),this->end(), [v](reference a) { a/=v; }); }
	inline void operator+=(value_type v) { std::for_each(this->begin(),this->end(), [v](reference a) { a+=v; }); }
	inline void operator-=(value_type v) { std::for_each(this->begin(),this->end(), [v](reference a) { a-=v; }); }

	inline simdn_t& operator++() { *this = *this + 1; return *this;}
	inline simdn_t& operator--() { *this = *this - 1; return *this;}



	inline simdn_t operator*(const simdn_t& v) const { simdn_t res; std::transform(this->begin(), this->end(), v.begin(), res.begin(), std::multiplies<value_type>()); return res;}
	inline simdn_t operator/(const simdn_t& v) const { simdn_t res; std::transform(this->begin(), this->end(), v.begin(), res.begin(), std::divides   <value_type>()); return res;}
	inline simdn_t operator+(const simdn_t& v) const { simdn_t res; std::transform(this->begin(), this->end(), v.begin(), res.begin(), std::plus      <value_type>()); return res;}
	inline simdn_t operator-(const simdn_t& v) const { simdn_t res; std::transform(this->begin(), this->end(), v.begin(), res.begin(), std::minus     <value_type>()); return res;}

	inline void operator*=(const simdn_t& v) { std::transform(this->begin(), this->end(), v.begin(), this->begin(), std::multiplies<value_type>()); }
	inline void operator/=(const simdn_t& v) { std::transform(this->begin(), this->end(), v.begin(), this->begin(), std::divides   <value_type>()); }
	inline void operator+=(const simdn_t& v) { std::transform(this->begin(), this->end(), v.begin(), this->begin(), std::plus      <value_type>()); }
	inline void operator-=(const simdn_t& v) { std::transform(this->begin(), this->end(), v.begin(), this->begin(), std::minus     <value_type>()); }

	inline reference operator[](unsigned i) { return _array[i]; }
	inline const_reference operator[](unsigned i) const { return _array[i]; }


	inline const std::array<value_type, N>& data() const { return this->_array; }

	friend
	std::ostream &operator<<(std::ostream & s, const simdn_t & v) {// Component-wise output stream
		for (size_t i=0; i<N; i++) s << v[i] << ' ';
		return s;
	}

	static constexpr size_t size() { return N; }


	inline       iterator begin()       { return _array.begin(); }
	inline const_iterator begin() const { return _array.begin(); }

	inline       iterator end()       { return _array.end(); }
	inline const_iterator end() const { return _array.end(); }

private:

	template<class ValueType, class ... ValueTypes>
	struct Setter
	{
		using NextSetter = Setter<ValueTypes...>;
		static_assert( N > sizeof...(ValueTypes), "Static Error in simdn_t::Setter: too many arguments in simdn_t ctor.");
		static constexpr auto M = N - sizeof...(ValueTypes)-1;

		static void set(std::array<value_type, N>& array, ValueType const& value, ValueTypes&& ... types)
		{
			std::get<M>(array) = value;
			NextSetter::set(array, std::move(types)...);
		}
	};

	template<class ValueType>
	struct Setter<ValueType>
	{
		static void set(std::array<value_type, N>& array, value_type const& value)
		{
			std::get<N-1>(array) = value;
		}
	};


//	void set1(value_type value)
//	{
//		_array[N-1] = value;
//	}

//	template<class ... ValueTypes>
//	void set1(value_type value, ValueTypes&& ... types)
//	{
//		_array[N - sizeof...(types)-1] = value;
//		set1(std::move(types)...);
//	}

//	template<class ... ValueTypes>
//	void set2(value_type v1, value_type v2, ValueTypes&& ... types)
//	{
//		_array[N - sizeof...(types)-1] = v1;
//		_array[N - sizeof...(types)-2] = v2;
//		set1(std::move(types)...);
//	}





	alignas(32) std::array<value_type, N> _array;
};

} // namespace boost::numeric::simd


namespace boost::numeric::simd
{

template<class T, size_t N>
inline
simdn_t<T,N>
operator+(typename simdn_t<T,N>::value_type r, const simdn_t<T,N>& vec)
{
	return vec.operator +(r);
}


template<class T, size_t N>
inline
simdn_t<T,N>
operator-(typename simdn_t<T,N>::value_type r, const simdn_t<T,N>& vec)
{
	simdn_t<T,N> rvec(r);
	return rvec - vec;
}

template<class T, size_t N>
inline
simdn_t<T,N>
operator*(typename simdn_t<T,N>::value_type r, const simdn_t<T,N>& vec)
{
	return vec.operator *(r);
}


template<class T, size_t N>
inline
simdn_t<T,N>
operator/(typename simdn_t<T,N>::value_type r, const simdn_t<T,N>& vec)
{
	return  simdn_t<T,N>(r) / vec;
}

} // namespace boost::numeric::simd

#endif


