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

#ifndef BOOST_NUMERIC_SIMD_TRAITS_H
#define BOOST_NUMERIC_SIMD_TRAITS_H

#include <type_traits>

namespace boost::numeric::simd
{

struct avx512_intrinsics_tag;
struct avx_intrinsics_tag;
struct sse_intrinsics_tag;
struct no_intrinsics_tag;


template<class intrinsics_tag>
struct intrinsics_traits;


template<>
struct intrinsics_traits<boost::numeric::simd::avx512_intrinsics_tag>
{
	static constexpr std::size_t bits   = 512ul;
	static constexpr std::size_t bytes  = bits>>3u;

};

template<>
struct intrinsics_traits<boost::numeric::simd::avx_intrinsics_tag>
{
	static constexpr std::size_t bits   = 256ul;
	static constexpr std::size_t bytes  = bits>>3u;

};

template<>
struct intrinsics_traits<boost::numeric::simd::sse_intrinsics_tag>
{
	static constexpr std::size_t bits   = 128ul;
	static constexpr std::size_t bytes  = bits>>3u;
};

template<>
struct intrinsics_traits<boost::numeric::simd::no_intrinsics_tag>
{
	static constexpr std::size_t bits   = 0ul;
	static constexpr std::size_t bytes  = 0ul;
};

} // namespace


#endif // BOOST_NUMERIC_SIMD_TRAITS_H
