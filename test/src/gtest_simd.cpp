#include <boost/numeric/simd/simd.h>
#include <boost/numeric/simd/align_allocator.h>


#include <gtest/gtest.h>

#include <ostream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdlib>


TEST(SimdTest,ctor)
{
	{
		using simd = boost::numeric::simd::cdouble4;
		using vtype = typename simd::value_type;

		simd vec1;
		vec1 = 0.0;
		for(auto i = 0ul; i < vec1.size(); ++i) EXPECT_TRUE(vec1[i] == std::complex<double>(0.0f));

		simd vec2(5);
		for(auto i = 0ul; i < vec2.size(); ++i) EXPECT_TRUE(vec2[i] == std::complex<double>(5.0,0));

		simd vec3(vtype(0,0),vtype(1,1),vtype(2,2),vtype(3,3));
//		std::cout << "vec3 = " << vec3 << std::endl;
		for(auto i = 0.0f; i < float(vec3.size()); ++i) EXPECT_TRUE(vec3[i] == std::complex<double>(i,i));
	}


	{
		using simd = boost::numeric::simd::float16;
		simd vec1;
		vec1 = 0.0;
		for(auto i = 0ul; i < vec1.size(); ++i) EXPECT_FLOAT_EQ(vec1[i], 0.0f);

		simd vec2(5);
		for(auto i = 0ul; i < vec2.size(); ++i) EXPECT_FLOAT_EQ(vec2[i], 5.0f);

        simd vec3(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
		for(auto i = 0.0f; i < float(vec3.size()); ++i) EXPECT_FLOAT_EQ(vec3[i], i);
	}


	{
		using simd = boost::numeric::simd::double8;
		simd vec1;
		vec1 = 0.0;
		for(auto i = 0ul; i < vec1.size(); ++i) EXPECT_FLOAT_EQ(vec1[i], 0.0f);

		simd vec2(5);
		for(auto i = 0ul; i < vec2.size(); ++i) EXPECT_FLOAT_EQ(vec2[i], 5.0f);

        simd vec3(0,1,2,3,4,5,6,7);
		for(auto i = 0.0f; i < float(vec3.size()); ++i) EXPECT_FLOAT_EQ(vec3[i], i);
	}




	{
		using simd = boost::numeric::simd::float8;
		simd vec1;
		vec1 = 0.0f;
		for(auto i = 0ul; i < vec1.size(); ++i) EXPECT_FLOAT_EQ(vec1[i], 0.0f);

		simd vec2(5);
		for(auto i = 0ul; i < vec2.size(); ++i) EXPECT_FLOAT_EQ(vec2[i], 5.0f);

		simd vec3(0,1,2,3,4,5,6,7);
		for(auto i = 0.0f; i < float(vec3.size()); ++i) EXPECT_FLOAT_EQ(vec3[i], i);
	}

	{
		using simd = boost::numeric::simd::double4;
		simd vec1;
		vec1 = 0.0f;
		for(auto i = 0ul; i < vec1.size(); ++i) EXPECT_FLOAT_EQ(vec1[i], 0.0);

		simd vec2(5);
		for(auto i = 0ul; i < vec2.size(); ++i) EXPECT_FLOAT_EQ(vec2[i], 5.0);

		simd vec3(0,1,2,3);
		for(auto i = 0.0; i < float(vec3.size()); ++i) EXPECT_FLOAT_EQ(vec3[i], i);
	}


	{
		using simd = boost::numeric::simd::float4;
		simd vec1;
		vec1 = 0.0f;
		for(auto i = 0ul; i < vec1.size(); ++i) EXPECT_FLOAT_EQ(vec1[i], 0.0);

		simd vec2(5);
		for(auto i = 0ul; i < vec2.size(); ++i) EXPECT_FLOAT_EQ(vec2[i], 5.0);

		simd vec3(0,1,2,3);
		for(auto i = 0.0; i < float(vec3.size()); ++i) EXPECT_FLOAT_EQ(vec3[i], i);
	}


	{
		using simd = boost::numeric::simd::double2;
		simd vec1(0.0f);
		for(auto i = 0ul; i < vec1.size(); ++i) EXPECT_FLOAT_EQ(vec1[i], 0.0);

		simd vec2(5);
		for(auto i = 0ul; i < vec2.size(); ++i) EXPECT_FLOAT_EQ(vec2[i], 5.0);

		simd vec3(0,1);
		for(auto i = 0.0; i < float(vec3.size()); ++i) EXPECT_FLOAT_EQ(vec3[i], i);
	}





	{
		using simd = boost::numeric::simd::simdn_t<float,8>;
		simd vec1;
		vec1 = 0.0f;
		for(auto i = 0ul; i < vec1.size(); ++i) EXPECT_FLOAT_EQ(vec1[i], 0.0f);

		simd vec2(5.0f);
		for(auto i = 0ul; i < vec2.size(); ++i) EXPECT_FLOAT_EQ(vec2[i], 5.0f);

		simd vec3(0.0f,1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f);
		for(auto i = 0.0f; i < float(vec3.size()); ++i) EXPECT_FLOAT_EQ(vec3[i], i);
	}

	{
		using simd = boost::numeric::simd::simdn_t<double,4>;
		simd vec1;
		vec1 = 0.0f;
		for(auto i = 0ul; i < vec1.size(); ++i) EXPECT_FLOAT_EQ(vec1[i], 0.0);

		simd vec2(5.0);
		for(auto i = 0ul; i < vec2.size(); ++i) EXPECT_FLOAT_EQ(vec2[i], 5.0);

		simd vec3(0.0,1.0,2.0,3.0);
		for(auto i = 0.0; i < float(vec3.size()); ++i) EXPECT_FLOAT_EQ(vec3[i], i);
	}


	{
		using simd = boost::numeric::simd::simdn_t<float,4>;
		simd vec1;
		vec1 = 0.0f;
		for(auto i = 0ul; i < vec1.size(); ++i) EXPECT_FLOAT_EQ(vec1[i], 0.0f);

		simd vec2(5.0f);
		for(auto i = 0ul; i < vec2.size(); ++i) EXPECT_FLOAT_EQ(vec2[i], 5.0f);

		simd vec3(0.0f,1.0f,2.0f,3.0f);
		for(auto i = 0.0f; i < float(vec3.size()); ++i) EXPECT_FLOAT_EQ(vec3[i], i);
	}

	{
		using simd = boost::numeric::simd::simdn_t<double,2>;
		simd vec1;
		vec1 = 0.0f;
		for(auto i = 0ul; i < vec1.size(); ++i) EXPECT_FLOAT_EQ(vec1[i], 0.0);

		simd vec2(5.0);
		for(auto i = 0ul; i < vec2.size(); ++i) EXPECT_FLOAT_EQ(vec2[i], 5.0);

		simd vec3(0.0,1.0);
		for(auto i = 0.0; i < float(vec3.size()); ++i) EXPECT_FLOAT_EQ(vec3[i], i);
	}


}

template<class SimdT>
struct SimdCopyCtorTester
{
	using value_type = typename SimdT::value_type;

	static void run(value_type num)
	{
		SimdT v1;
		v1 = num;
		auto v2 = v1;
		for(auto i = 0ul; i < v1.size(); ++i) EXPECT_FLOAT_EQ(v2[i], num);

		auto v3 = v1.data();
		for(auto i = 0ul; i < v1.size(); ++i) EXPECT_FLOAT_EQ(v3[i], num);

		auto v4 = SimdT(num);
		for(auto i = 0ul; i < v1.size(); ++i) EXPECT_FLOAT_EQ(v4[i], num);
	}
};

TEST(simd,copy_ctor)
{
	SimdCopyCtorTester<boost::numeric::simd::float16>::run(5.0f);
	SimdCopyCtorTester<boost::numeric::simd::float8 >::run(5.0f);
	SimdCopyCtorTester<boost::numeric::simd::float4 >::run(5.0f);

	SimdCopyCtorTester<boost::numeric::simd::double8 >::run(5.0f);
	SimdCopyCtorTester<boost::numeric::simd::double4 >::run(5.0f);
	SimdCopyCtorTester<boost::numeric::simd::double2 >::run(5.0f);

	SimdCopyCtorTester<boost::numeric::simd::simdn_t<float,16>>::run(5.0f);
	SimdCopyCtorTester<boost::numeric::simd::simdn_t<float, 8>>::run(5.0f);
	SimdCopyCtorTester<boost::numeric::simd::simdn_t<float, 4>>::run(5.0f);

	SimdCopyCtorTester<boost::numeric::simd::simdn_t<double, 8>>::run(5.0f);
	SimdCopyCtorTester<boost::numeric::simd::simdn_t<double, 4>>::run(5.0f);
	SimdCopyCtorTester<boost::numeric::simd::simdn_t<double, 2>>::run(5.0f);
}


template<class SimdT>
struct SimdCopyOperatorTester
{
	using value_type = typename SimdT::value_type;

	static void run(value_type num)
	{
		SimdT v1;
		v1 = num;
		SimdT v2;
		v2 = v1;
		for(auto i = 0ul; i < v1.size(); ++i) EXPECT_FLOAT_EQ(v2[i], num);

		SimdT v3;
		v3 = v1.data();
		for(auto i = 0ul; i < v1.size(); ++i) EXPECT_FLOAT_EQ(v3[i], num);

		SimdT v4;
		v4 = SimdT(num);
		for(auto i = 0ul; i < v1.size(); ++i) EXPECT_FLOAT_EQ(v4[i], num);
	}
};


TEST(simd,copy_operator)
{
	SimdCopyOperatorTester<boost::numeric::simd::float16>::run(5.0f);
	SimdCopyOperatorTester<boost::numeric::simd::float8 >::run(5.0f);
	SimdCopyOperatorTester<boost::numeric::simd::float4 >::run(5.0f);

	SimdCopyOperatorTester<boost::numeric::simd::double8 >::run(5.0f);
	SimdCopyOperatorTester<boost::numeric::simd::double4 >::run(5.0f);
	SimdCopyOperatorTester<boost::numeric::simd::double2 >::run(5.0f);

	SimdCopyOperatorTester<boost::numeric::simd::simdn_t<float,16>>::run(5.0f);
	SimdCopyOperatorTester<boost::numeric::simd::simdn_t<float, 8>>::run(5.0f);
	SimdCopyOperatorTester<boost::numeric::simd::simdn_t<float, 4>>::run(5.0f);

	SimdCopyOperatorTester<boost::numeric::simd::simdn_t<double, 8>>::run(5.0f);
	SimdCopyOperatorTester<boost::numeric::simd::simdn_t<double, 4>>::run(5.0f);
	SimdCopyOperatorTester<boost::numeric::simd::simdn_t<double, 2>>::run(5.0f);
}


template <class SimdT>
struct SimdArithmeticScalarTester
{
	using value_type = typename SimdT::value_type;

	static void run()
	{
		value_type j = 0.0;
		size_t i = 0ul;
		SimdT v1, v2;
		for(auto i = 0ul, j = 2ul; i < SimdT::size(); ++i, j+=2)
			v1[i] = j;
		v2 = v1;

		v1 += 2;
		for(i = 0ul, j = 2.0; i < v1.size(); ++i, j+=2) EXPECT_FLOAT_EQ(v1[i], j+2);

		v1 = v2;
		v1 -= 2;
		for(i = 0ul, j = 2.0; i < v1.size(); ++i, j+=2) EXPECT_FLOAT_EQ(v1[i], j-2);

		v1 = v2;
		v1 *= 2;
		for(i = 0ul, j = 2.0; i < v1.size(); ++i, j+=2) EXPECT_FLOAT_EQ(v1[i], j*2);

		v1 = v2;
		v1 /= 2;
		for(i = 0ul, j = 2.0; i < v1.size(); ++i, j+=2) EXPECT_FLOAT_EQ(v1[i], j/2);

		v1 = v2 + 2;
		for(i = 0ul, j = 2.0; i < v1.size(); ++i, j+=2) EXPECT_FLOAT_EQ(v1[i], j+2);

		v1 = v2 - 2;
		for(i = 0ul, j = 2.0; i < v1.size(); ++i, j+=2) EXPECT_FLOAT_EQ(v1[i], j-2);

		v1 = v2 * 2;
		for(i = 0ul, j = 2.0; i < v1.size(); ++i, j+=2) EXPECT_FLOAT_EQ(v1[i], j*2);

		v1 = v2 / 2;
		for(i = 0ul, j = 2.0; i < v1.size(); ++i, j+=2) EXPECT_FLOAT_EQ(v1[i], j/2);


		v1 = 2.0f + v2;
		for(i = 0ul, j = 2.0; i < v1.size(); ++i, j+=2) EXPECT_FLOAT_EQ(v1[i], j+2);

		v1 = 2.0f - v2;
		for(i = 0ul, j = 2.0; i < v1.size(); ++i, j+=2) EXPECT_FLOAT_EQ(v1[i], 2-j);

		v1 = 2.0f * v2;
		for(i = 0ul, j = 2.0; i < v1.size(); ++i, j+=2) EXPECT_FLOAT_EQ(v1[i], j*2);

		v1 = 2.0 / v2;
		for(i = 0ul, j = 2.0; i < v1.size(); ++i, j+=2) EXPECT_FLOAT_EQ(v1[i], 2/j);

	}
};



TEST(simd,arithmetic_scalar)
{
	SimdArithmeticScalarTester<boost::numeric::simd::float16>::run();
	SimdArithmeticScalarTester<boost::numeric::simd::float8 >::run();
	SimdArithmeticScalarTester<boost::numeric::simd::float4 >::run();

	SimdArithmeticScalarTester<boost::numeric::simd::double8 >::run();
	SimdArithmeticScalarTester<boost::numeric::simd::double4 >::run();
	SimdArithmeticScalarTester<boost::numeric::simd::double2 >::run();

	SimdArithmeticScalarTester<boost::numeric::simd::simdn_t<float,16>>::run();
	SimdArithmeticScalarTester<boost::numeric::simd::simdn_t<float, 8>>::run();
	SimdArithmeticScalarTester<boost::numeric::simd::simdn_t<float, 4>>::run();

	SimdArithmeticScalarTester<boost::numeric::simd::simdn_t<double, 8>>::run();
	SimdArithmeticScalarTester<boost::numeric::simd::simdn_t<double, 4>>::run();
	SimdArithmeticScalarTester<boost::numeric::simd::simdn_t<double, 2>>::run();
}

template <class SimdT>
struct SimdArithmeticTester
{
	using value_type = typename SimdT::value_type;

	static void run()
	{
		SimdT v1, v2;
		for(auto i = 0ul, j = 2ul; i < SimdT::size(); ++i, j+=2)
			v1[i] = j;
		v2 = v1;

		v1 += v2;
		for(auto i = 0ul, j = 2ul; i < v1.size(); ++i, j+=2) EXPECT_FLOAT_EQ(v1[i], j+j);

		v1 = v2;
		v1 -= v2;
		for(auto i = 0ul, j = 2ul; i < v1.size(); ++i, j+=2) EXPECT_FLOAT_EQ(v1[i], j-j);

		v1 = v2;
		v1 *= v2;
		for(auto i = 0ul, j = 2ul; i < v1.size(); ++i, j+=2) EXPECT_FLOAT_EQ(v1[i], j*j);

		v1 = v2;
		v1 /= v2;
		for(auto i = 0ul, j = 2ul; i < v1.size(); ++i, j+=2) EXPECT_FLOAT_EQ(v1[i], j/j);

	}
};


TEST(simd,arithmetic)
{
	SimdArithmeticTester<boost::numeric::simd::float16>::run();
	SimdArithmeticTester<boost::numeric::simd::float8 >::run();
	SimdArithmeticTester<boost::numeric::simd::float4 >::run();

	SimdArithmeticTester<boost::numeric::simd::double8 >::run();
	SimdArithmeticTester<boost::numeric::simd::double4 >::run();
	SimdArithmeticTester<boost::numeric::simd::double2 >::run();

	SimdArithmeticTester<boost::numeric::simd::simdn_t<float,16>>::run();
	SimdArithmeticTester<boost::numeric::simd::simdn_t<float, 8>>::run();
	SimdArithmeticTester<boost::numeric::simd::simdn_t<float, 4>>::run();

	SimdArithmeticTester<boost::numeric::simd::simdn_t<double, 8>>::run();
	SimdArithmeticTester<boost::numeric::simd::simdn_t<double, 4>>::run();
	SimdArithmeticTester<boost::numeric::simd::simdn_t<double, 2>>::run();
}


template <class SimdT>
struct SimdCompareTester
{
	using value_type = typename SimdT::value_type;

	static void run()
	{
		SimdT v1, v2;
		for(auto i = 0ul; i < SimdT::size(); i+=2)
			v1[i] = i+2;
		v2 = v1+2;

		EXPECT_TRUE (v2 == v2);
		EXPECT_TRUE (v2 <= v2);
		EXPECT_TRUE (v2 >= v2);

		EXPECT_FALSE(v2 != v2);
		EXPECT_FALSE(v2  > v2);
		EXPECT_FALSE(v2  < v2);

		EXPECT_TRUE (v2 == (v1 + 2.0f));
		EXPECT_TRUE (v2 >= (v1 + 2.0f));
		EXPECT_TRUE (v2 <= (v1 + 2.0f));

		EXPECT_FALSE(v2 != (v1 + 2.0f));
		EXPECT_FALSE(v2 <  (v1 + 2.0f));
		EXPECT_FALSE(v2 >  (v1 + 2.0f));
	}
};



TEST(simd,compare)
{
	SimdCompareTester<boost::numeric::simd::float16>::run();
	SimdCompareTester<boost::numeric::simd::float8 >::run();
	SimdCompareTester<boost::numeric::simd::float4 >::run();

	SimdCompareTester<boost::numeric::simd::double8 >::run();
	SimdCompareTester<boost::numeric::simd::double4 >::run();
	SimdCompareTester<boost::numeric::simd::double2 >::run();

	SimdCompareTester<boost::numeric::simd::simdn_t<float,16>>::run();
	SimdCompareTester<boost::numeric::simd::simdn_t<float, 8>>::run();
	SimdCompareTester<boost::numeric::simd::simdn_t<float, 4>>::run();

	SimdCompareTester<boost::numeric::simd::simdn_t<double, 8>>::run();
	SimdCompareTester<boost::numeric::simd::simdn_t<double, 4>>::run();
	SimdCompareTester<boost::numeric::simd::simdn_t<double, 2>>::run();
}

template <class SimdT>
struct SimdArrayTester
{
	using simd_type = SimdT;
	using value_type = typename simd_type::value_type;
	using dynarray = std::vector<simd_type, boost::numeric::simd::align_allocator_t<simd_type, 64ul> >;

	static void run()
	{
		const simd_type one(1.0f);
		const simd_type two(2.0f);

		dynarray array(100, one);

		EXPECT_TRUE ( std::all_of(array.begin(), array.end(), [] ( const simd_type&  vec) { return vec == 1;} ) );
		EXPECT_FALSE( std::all_of(array.begin(), array.end(), [] ( const simd_type&  vec) { return vec != 1;} ) );

		array = dynarray(200, two);

		EXPECT_TRUE ( std::all_of(array.begin(), array.end(), [] ( const simd_type&  vec) { return vec > 1;} ) );
		EXPECT_FALSE( std::all_of(array.begin(), array.end(), [] ( const simd_type&  vec) { return vec < 1;} ) );

		auto result = std::accumulate(array.begin(), array.end(), simd_type(float(0)));

		EXPECT_TRUE( result == two * array.size() );
	}




};


TEST(simd,array)
{
	SimdArrayTester<boost::numeric::simd::float16>::run();
	SimdArrayTester<boost::numeric::simd::float8 >::run();
	SimdArrayTester<boost::numeric::simd::float4 >::run();

	SimdArrayTester<boost::numeric::simd::double8 >::run();
	SimdArrayTester<boost::numeric::simd::double4 >::run();
	SimdArrayTester<boost::numeric::simd::double2 >::run();

	SimdArrayTester<boost::numeric::simd::simdn_t<float,16>>::run();
	SimdArrayTester<boost::numeric::simd::simdn_t<float, 8>>::run();
	SimdArrayTester<boost::numeric::simd::simdn_t<float, 4>>::run();

	SimdArrayTester<boost::numeric::simd::simdn_t<double, 8>>::run();
	SimdArrayTester<boost::numeric::simd::simdn_t<double, 4>>::run();
	SimdArrayTester<boost::numeric::simd::simdn_t<double, 2>>::run();

}




