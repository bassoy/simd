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


#ifndef BOOST_NUMERIC_SIMD_COMPLEX_H
#define BOOST_NUMERIC_SIMD_COMPLEX_H

#include <complex>
#include <cassert>

namespace boost::numeric::simd
{

template<class __simd_type>
class csimd_t
{	
public:

	using simd_type = __simd_type;
	using intrinsics_tag = typename simd_type::intrinsics_tag;

	using value_type = std::complex<typename simd_type::value_type>;

	using       reference = value_type&; // value_type&;
	using const_reference = const value_type&;

	using       pointer = value_type*;
	using const_pointer = const value_type*;


	explicit csimd_t() : _real(), _imag() {}

	explicit csimd_t(const_pointer p) :
		_real(), _imag()
	{
		assert(p != nullptr);
		this->load(p);
	}

	explicit csimd_t(const_reference v) :
		_real(v.real()), _imag(v.imag())
	{}

	csimd_t(const simd_type& real, const simd_type& imag) : _real(real), _imag(imag) {}

	csimd_t(const csimd_t& other) : _real(other._real), _imag(other._imag) {}
	csimd_t(csimd_t&& other) : _real(std::move(other._real)),_imag(std::move(other._imag)) {}

	template<class ... ValueTypes>
	explicit csimd_t(ValueTypes&& ... types)
		: _real(), _imag()
	{
		static_assert(sizeof...(types) <= M, "Number of input variables must be smaller than M");
		if(sizeof...(types) == 1) { auto t = {types...}; this->operator =(*t.begin()); return; }
		set(std::move(types)...);
	}

	inline void load(const_pointer p)
	{
		assert(p != nullptr);
		for(size_t j = 0; j < M; ++j)
		{
			_real[j] = *p; ++p;
			_imag[j] = *p; ++p;
		}
	}

	inline csimd_t& operator=(const_reference v) { _real = v.real(); _imag = v.imag(); return *this;}
	inline csimd_t& operator=(const csimd_t& other) { _real = other._real; _imag = other._imag; return *this;}
	inline csimd_t& operator=(csimd_t&& other) { _real = std::move(other._real); _imag = std::move(other._imag); return *this;}

	inline bool operator==(const csimd_t& other) const { return _real == other._real && _imag == other._imag; }
	inline bool operator!=(const csimd_t& other) const { return _real != other._real && _imag != other._imag; }
	inline bool operator<=(const csimd_t& other) const { return _real <= other._real && _imag <= other._imag; }
	inline bool operator< (const csimd_t& other) const { return _real <  other._real && _imag <  other._imag; }
	inline bool operator>=(const csimd_t& other) const { return _real >= other._real && _imag >= other._imag; }
	inline bool operator> (const csimd_t& other) const { return _real >  other._real && _imag >  other._imag; }

	inline csimd_t operator*(const csimd_t& w) const
	{
		csimd_t y;
		const csimd_t& v = *this;

		y._real = v._real * w._real - v._imag * w._imag;
		y._imag = v._real * w._imag + v._imag * w._real;

		return y;
	}

	inline csimd_t operator/(const csimd_t& w) const
	{
		csimd_t y;
		const csimd_t& v = *this;

		simd_type x = 1.0 / (w._real * w._real + w._imag * w._imag);

		y._real = (v._real * w._real + v._imag * w._imag) * x;
		y._imag = (v._imag * w._real - v._real * w._imag) * x;

		return y;
	}

	inline csimd_t operator+(const csimd_t& w) const
	{
		csimd_t y;
		const csimd_t& v = *this;

		y._real = v._real + w._real;
		y._imag = v._imag + w._imag;

		return y;
	}


	inline csimd_t operator-(const csimd_t& w) const
	{
		csimd_t y;
		const csimd_t& v = *this;

		y._real = v._real + w._real;
		y._imag = v._imag + w._imag;

		return y;
	}


	inline void operator*=(const csimd_t& w)
	{
		csimd_t& y = *this;
		const csimd_t& v = *this;

		y._real = v._real * w._real - v._imag * w._imag;
		y._imag = v._real * w._imag + v._imag * w._real;

//		return y;
	}

	inline void operator/=(const csimd_t& w)
	{
		csimd_t& y = *this;
		const csimd_t& v = *this;

		simd_type x = 1.0 / (w._real * w._real + w._imag * w._imag);

		y._real = (v._real * w._real + v._imag * w._imag) * x;
		y._imag = (v._imag * w._real - v._real * w._imag) * x;

//		return y;
	}

	inline void operator+=(const csimd_t& w)
	{
		csimd_t& y = *this;
		const csimd_t& v = *this;

		y._real = v._real + w._real;
		y._imag = v._imag + w._imag;

//		return y;
	}


	inline void operator-=(const csimd_t& w)
	{
		csimd_t& y = *this;
		const csimd_t& v = *this;

		y._real = v._real + w._real;
		y._imag = v._imag + w._imag;

//		return y;
	}


	inline value_type operator[](size_t i) const { return value_type(_real[i], _imag[i]); }

	inline const simd_type& real() const { return this->_real; }
	inline const simd_type& imag() const { return this->_imag; }

	inline simd_type& real() { return this->_real; }
	inline simd_type& imag() { return this->_imag; }


	friend
	std::ostream &operator<<(std::ostream & s, const csimd_t & v) {// Component-wise output stream
		for (unsigned i=0; i<M-1; i++){
			s << "(" << v.real()[i] << "," << v.imag()[i] << "), ";
		}
		s << "(" << v.real()[M-1] << "," << v.imag()[M-1] << ")";
		return s;
	}

	static constexpr unsigned size() { return M; }

private:
	static constexpr unsigned M = simd_type::size();

	template<class ... ValueTypes>
	inline void set(value_type&& value, ValueTypes ... types)
	{
		static constexpr unsigned K = M - unsigned(sizeof...(types)) - 1u;
		_real[K] = value.real();
		_imag[K] = value.imag();
		set(std::move(types)...);
	}

	inline void set(value_type&& value) { _real[M-1] = value.real(); _imag[M-1] = value.imag(); }



	alignas(32) simd_type _real;
	alignas(32) simd_type _imag;
};

} // namespace boost::numeric::simd 

#endif // SIMD_COMPLEX_H
