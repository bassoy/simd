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

#ifndef BOOST_NUMERIC_SIMD_ALIGN_ALLOCATOR_H
#define BOOST_NUMERIC_SIMD_ALIGN_ALLOCATOR_H

//

#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <memory>
#include <new>
#include <iostream>
#include <vector>
#include <algorithm>



namespace boost::numeric::simd
{

template<size_t i, size_t n>
struct __is_power_of_two { const static size_t value = (n&1u)^__is_power_of_two<i-1,(n>>1) >::value; };

template<size_t n>
struct __is_power_of_two<1,n> { const static size_t value = n&1u; };


/*! \brief checks if n is power of two
 *
 * (n)_10 = (b_64 ... b_2 b_1)_2
 *
 * Forumla: is_true =  ( b_64 xor b_63 xor ... xor b_1 ) == 1
 *
*/
template<size_t n>
struct is_power_of_two { const static bool value = (1u == __is_power_of_two<64u, n>::value); };



template<typename T, std::size_t __align = 128>
class align_allocator_t
{
	static constexpr std::size_t alignment = __align;
	static_assert(__align != 0, "alignment cannot be equal zero");
	static_assert(alignof(T) <= alignment, "Alignment cannot be chosen smaller than the natural alignment of T");
	static_assert(is_power_of_two<alignment>::value, "Alignment must be a power of two.");
	static_assert((alignment%alignof(T)) == 0, "Alignment must be a multiple of the alignment of T.");

	static constexpr std::size_t natural = sizeof(void *) > alignof(long double) ? sizeof(void *) : (alignof(long double) > alignof(long long) ? alignof(long double) : alignof(long long));
	static constexpr std::size_t extra_bytes = alignment > natural ? ( alignment + sizeof(std::ptrdiff_t) ): 0ul;
	//static constexpr std::ptrdiff_t mask = ~(alignment - 1ul);
	static constexpr std::size_t mask = ~(alignment - 1ul);
public:
	typedef std::size_t     size_type;
	typedef std::ptrdiff_t  difference_type;
	typedef T*              pointer;
	typedef const T*        const_pointer;
	typedef T&              reference;
	typedef const T&        const_reference;
	typedef T               value_type;

	template<typename T1>
	struct rebind
	{
		using other = align_allocator_t<T1, alignment>;
	};

	constexpr align_allocator_t()
		: _vector_aligned{}
		, _vector_non_aligned{}
	{}

	align_allocator_t(const align_allocator_t&)
		: _vector_aligned ()
		, _vector_non_aligned ()
	{}

	align_allocator_t(align_allocator_t&& other)
		: _vector_aligned (std::move(other._vector_aligned))
		, _vector_non_aligned (std::move(other._vector_non_aligned))
	{}

	template<typename T1>
	align_allocator_t(const align_allocator_t<T1, alignment>&) throw()
	{
		assert(0);
	}


	align_allocator_t& operator=(const align_allocator_t& other)
	{
		_vector_aligned = other._vector_aligned;
		_vector_non_aligned = other._vector_non_aligned;
		assert(0);
		return *this;
	}

	align_allocator_t& operator=(align_allocator_t&& other)
	{
		_vector_aligned = std::move(other._vector_aligned);
		_vector_non_aligned = std::move(other._vector_non_aligned);
		return *this;
	}


	~align_allocator_t() = default;

	pointer address(reference r)
	{
		return std::addressof(r);
	}

	const_pointer address(const_reference r) const
	{
		return std::addressof(r);
	}

	// NB: __n is permitted to be 0.  The C++ standard says nothing
	// about what the return value is when __n == 0.
	pointer allocate(size_type n, const void* = 0)
	{
		if(n == 0ul) return nullptr;
		if(n > this->max_size())
			std::bad_alloc();

		pointer non_aligned = static_cast<pointer>(std::malloc( n * sizeof(T) + extra_bytes + 128 ));
		if(non_aligned == nullptr)
			std::bad_alloc();

		if( extra_bytes ==  0ul ){
			_vector_aligned.push_back( reinterpret_cast<size_type>(non_aligned) );
			_vector_non_aligned.push_back( reinterpret_cast<size_type>(non_aligned) );
			return non_aligned;
		}


		size_type non_aligned_size_type = reinterpret_cast<size_type>(non_aligned);

		// go extra_bytes in memory starting at ptr1, and then mask the location to get the aligned location
		size_type aligned_size_type = (non_aligned_size_type + extra_bytes) & mask;

		assert(aligned_size_type%alignment==0);
		assert(aligned_size_type > non_aligned_size_type);

		auto offset = aligned_size_type - non_aligned_size_type;
		non_aligned_size_type = aligned_size_type - offset;

//		std::cout << "allocate: offset  = " << offset << std::endl;
//		std::cout << "allocate: non_aligned = " << std::hex << non_aligned_size_type << std::dec << std::endl;
//		std::cout << "allocate: aligned = " << std::hex << aligned_size_type << std::dec << std::endl << std::endl;

		_vector_aligned.push_back( reinterpret_cast<size_type>(aligned_size_type) );
		_vector_non_aligned.push_back( reinterpret_cast<size_type>(non_aligned_size_type) );


		// converting the aligned location into a pointer.
		return reinterpret_cast<pointer>(aligned_size_type);

	}

	// __p is not permitted to be a null pointer.
	void deallocate(pointer aligned, size_type)
	{
		assert(aligned != nullptr);

		assert(!_vector_aligned.empty());
		if( extra_bytes ==  0ul ){
			std::free(aligned);
		}
		else {			
			auto aligned_size_type = reinterpret_cast<size_type>(aligned);
			assert(aligned_size_type%alignment==0);
			auto it = std::find( this->_vector_aligned.begin(), this->_vector_aligned.end(), aligned_size_type );
			assert( it != this->_vector_aligned.end());
			auto pos = it - this->_vector_aligned.begin();
			auto non_aligned_size_type = this->_vector_non_aligned.at( pos ) ;
			assert(aligned_size_type > non_aligned_size_type);
			auto non_aligned = reinterpret_cast<pointer>(non_aligned_size_type);
			assert(non_aligned != nullptr);
			_vector_aligned.erase(it);
			_vector_non_aligned.erase(_vector_non_aligned.begin() + pos);
//			auto non_aligned_size_type2 = reinterpret_cast<size_type>(_non_aligned);

//			auto offset = aligned_size_type - non_aligned_size_type;

//			std::cout << "deallocate: offset  = " << offset << std::endl;
//			std::cout << "deallocate: non_aligned = " << std::hex << non_aligned_size_type << std::dec << std::endl;
//			std::cout << "deallocate: aligned = " << std::hex << aligned_size_type << std::dec << std::endl << std::endl;

//			assert(non_aligned_size_type2 == non_aligned_size_type);
			std::free(non_aligned);
		}
	}

	bool
	operator==(align_allocator_t const& r)
	{ return this->_vector_aligned == r._vector_aligned; }

	bool
	operator!=(align_allocator_t const& r)
	{ return this->_vector_aligned != r._vector_aligned; }

	std::vector<size_type> const& vector_aligned() const { return _vector_aligned; }
	std::vector<size_type> const& vector_non_aligned() const { return _vector_non_aligned; }

	constexpr size_type max_size() const throw()
	{
		return size_type(-1) / sizeof(T);
	}

	template <typename other_pointer>
	void construct(other_pointer __p, const T& __val)
	{
		::new((void *)__p) T(__val);
	}

	template<typename other_pointer, typename... _Args>
	void construct(other_pointer __p, _Args&&... __args)
	{
		::new((void *)__p) T(std::forward<_Args>(__args)...);
	}

	template <typename other_pointer>
	void destroy(other_pointer __p) { __p->~T(); }

private:
//	size_type _offset;
//	pointer _non_aligned;
	std::vector<size_type> _vector_aligned;
	std::vector<size_type> _vector_non_aligned;


};

} // namespace boost::numeric::ublas

template<typename T1, typename T2, size_t A>
inline bool
operator==(const boost::numeric::simd::align_allocator_t<T1, A>&, const boost::numeric::simd::align_allocator_t<T2, A>&)
{ return false; }

template<typename T, size_t A1, size_t A2>
inline bool
operator==(const boost::numeric::simd::align_allocator_t<T, A1>&, const boost::numeric::simd::align_allocator_t<T, A2>&)
{ return false; }

template<typename T1, typename T2, size_t A1, size_t A2>
inline bool
operator==(const boost::numeric::simd::align_allocator_t<T1, A1>&, const boost::numeric::simd::align_allocator_t<T2, A2>&)
{ return false; }


//template<typename T, size_t A>
//inline bool
//operator!=(const boost::numeric::simd::align_allocator_t<T, A>& l, const boost::numeric::simd::align_allocator_t<T, A>& r)
//{ return l.offset() != r.offset(); }

template<typename T1, typename T2, size_t A>
inline bool
operator!=(const boost::numeric::simd::align_allocator_t<T1, A>&, const boost::numeric::simd::align_allocator_t<T2, A>&)
{ return true; }

template<typename T, size_t A1, size_t A2>
inline bool
operator!=(const boost::numeric::simd::align_allocator_t<T, A1>&, const boost::numeric::simd::align_allocator_t<T, A2>&)
{ return true; }

template<typename T1, typename T2, size_t A1, size_t A2>
inline bool
operator!=(const boost::numeric::simd::align_allocator_t<T1, A1>&, const boost::numeric::simd::align_allocator_t<T2, A2>&)
{ return true; }


#endif
