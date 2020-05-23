#include <boost/numeric/simd/align_allocator.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <type_traits>

TEST(AlignAllocator, Construction)
{
	using value_type = float;
	constexpr std::size_t align = 32ul;
	using allocator_type = boost::numeric::simd::align_allocator_t<value_type,align>;
	using vector_type_aligned = std::vector<value_type,allocator_type>;

	vector_type_aligned v(3);

	EXPECT_EQ(0ul, reinterpret_cast<std::ptrdiff_t>(v.data()) & (align-1ul)  );

}
