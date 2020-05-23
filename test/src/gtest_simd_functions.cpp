
#include <boost/numeric/simd/simd_functions.h>

#include <gtest/gtest.h>

const double pi = 3.1415926535897932384626433832795028841971693993751058;
int loops = 1e1;

TEST(SimdVectorFunctions,cos)
{

	{
		using type = boost::numeric::simd::double4;
		type vec1(pi/3.0, 0.0, 0.0, 0.0);
		type res(0.5, 1, 1, 1);
		type vec2;

		vec2 = boost::numeric::simd::cos(vec1);

		for(int i = 0; i < 4; i++){
			EXPECT_DOUBLE_EQ(res[i], vec2[i]);
		}
	}

	{
		using type = boost::numeric::simd::float8;
		type vec1(pi/3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
		type res(0.5, 1, 1, 1, 1, 1, 1, 1);
		type vec2;

		vec2 = boost::numeric::simd::cos(vec1);

		for(int i = 0; i < 8; i++){
			EXPECT_FLOAT_EQ(res[i], vec2[i]);
		}
	}

	{
		using type = boost::numeric::simd::float4;
		type vec1(pi/3.0, 0.0, 0.0, 0.0);
		type res(0.5, 1, 1, 1);
		type vec2;

		vec2 = boost::numeric::simd::cos(vec1);

		for(int i = 0; i < 4; i++){
			EXPECT_FLOAT_EQ(res[i], vec2[i]);
		}
	}

}

TEST(CVectorFunctions,sin)
{

	{
		using type = boost::numeric::simd::double4;
		type vec1(pi, pi/2.0, pi/6.0, 0.0);
		type res(0, 1, 0.5, 0);
		type vec2 = boost::numeric::simd::sin(vec1);

		for(int i = 0; i < 4; i++){
			EXPECT_DOUBLE_EQ(res[i], vec2[i]);
		}
	}

	{
		using type = boost::numeric::simd::float8;
		type vec1(pi, pi/2.0, pi/6.0, 0.0, 0.0, 0.0, 0.0, 0.0);
		type res(0, 1, 0.5, 0, 0, 0, 0, 0);
		type vec2;

		vec2 = boost::numeric::simd::sin(vec1);

		for(int i = 0; i < 8; i++){
			EXPECT_FLOAT_EQ(res[i], vec2[i]);
		}
	}

	{
		using type = boost::numeric::simd::float4;
		type vec1(pi, pi/2.0, pi/6.0, 0.0);
		type res(0, 1, 0.5, 0);
		type vec2;

		vec2 = boost::numeric::simd::sin(vec1);

		for(int i = 0; i < 4; i++){
			EXPECT_FLOAT_EQ(res[i], vec2[i]);
		}
	}

}

TEST(CVectorFunctions,tan)
{
	{
		using type = boost::numeric::simd::double4;
		type vec1(pi, 2.0*pi, pi/4.0, 3.0*pi/4.0);
		type res(0, 0, 1, -1);
		type vec2;

		vec2 = boost::numeric::simd::tan(vec1);

		for(int i = 0; i < 4; i++){
			EXPECT_DOUBLE_EQ(res[i], vec2[i]);
		}
	}

	{
		using type = boost::numeric::simd::float16;
		type vec1(pi, 2.0*pi, pi/4.0, 3.0*pi/4.0, 0.0, 0.0, 0.0, 0.0,   pi, 2.0*pi, pi/4.0, 3.0*pi/4.0, 0.0, 0.0, 0.0, 0.0);
		type res (0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0);
		type vec2;

		vec2 = boost::numeric::simd::tan(vec1);

		for(auto i = 0u; i < res.size(); i++) EXPECT_FLOAT_EQ(res[i], vec2[i]);
	}

	{
		using type = boost::numeric::simd::double8;
		type vec1(pi, 2.0*pi, pi/4.0, 3.0*pi/4.0, 0.0, 0.0, 0.0, 0.0);
		type res(0, 0, 1, -1, 0, 0, 0, 0);
		type vec2;

		vec2 = boost::numeric::simd::tan(vec1);

		for(int i = 0; i < 8; i++) EXPECT_DOUBLE_EQ(res[i], vec2[i]);
	}

	{
		using type = boost::numeric::simd::float8;
		type vec1(pi, 2.0*pi, pi/4.0, 3.0*pi/4.0, 0.0, 0.0, 0.0, 0.0);
		type res(0, 0, 1, -1, 0, 0, 0, 0);
		type vec2;

		vec2 = boost::numeric::simd::tan(vec1);

		for(int i = 0; i < 8; i++) EXPECT_FLOAT_EQ(res[i], vec2[i]);
	}

	{
		using type = boost::numeric::simd::float4;
		type vec1(pi, 2.0*pi, pi/4.0, 3.0*pi/4.0);
		type res(0, 0, 1, -1);
		type vec2;

		vec2 = boost::numeric::simd::tan(vec1);

		for(int i = 0; i < 4; i++){
			EXPECT_FLOAT_EQ(res[i], vec2[i]);
		}
	}
}



TEST(CVectorFunctions,cmult)
{
	//complex (4 complex numbers per vector)
	{
		using simd = boost::numeric::simd::cdouble4;
		using vtype = typename simd::value_type;

		simd vec1(vtype(0,0),vtype(1,1),vtype(2,2),vtype(3,3));
		simd vec2(vtype(0,0),vtype(1,1),vtype(2,2),vtype(3,3));
		simd vec3;

		vec3 = vec1 * vec2;

		simd res(vtype(0,0),vtype(0,2),vtype(0,8),vtype(0,18));
		for(auto i = 0ul; i < vec3.size(); ++i) EXPECT_TRUE(vec3[i] == res[i]);
	}

	//complex (4 complex numbers per vector)
	{
		using simd = boost::numeric::simd::cfloat8;
		using vtype = typename simd::value_type;

		simd vec1(vtype(0,0),vtype(1,1),vtype(2,2),vtype(3,3), vtype(4,4),vtype(5,5),vtype(6,6),vtype(7,7));
		simd vec2(vtype(0,0),vtype(1,1),vtype(2,2),vtype(3,3), vtype(4,4),vtype(5,5),vtype(6,6),vtype(7,7));
		simd vec3;

		vec3 = vec1 * vec2;
//		simd res(vtype(0,0),vtype(0,2),vtype(0,8),vtype(0,18));
//		for(auto i = 0ul; i < vec3.size(); ++i) EXPECT_TRUE(vec3[i] == res[i]);
	}

}

TEST(CVectorFunctions,cdiv)
{
	//complex (4 complex numbers per vector)
	{
		using simd = boost::numeric::simd::cdouble4;
		using vtype = typename simd::value_type;

		simd vec1(vtype(0,0),vtype(1,1),vtype(2,2),vtype(3,3));
		simd vec2(vtype(-1,-1),vtype(1,1),vtype(2,2),vtype(3,3));
		simd vec3;

		vec3 = vec1 / vec2;

		simd res(vtype(0,0),vtype(1,0),vtype(1,0),vtype(1,0));
		//std::cout << vec3 << " == " << res << std::endl;
		for(auto i = 0ul; i < vec3.size(); ++i) EXPECT_TRUE(vec3[i] == res[i]);
	}

	{
		using simd = boost::numeric::simd::cfloat8;
		using vtype = typename simd::value_type;

		simd vec1(vtype(0,0),vtype(1,1),vtype(2,2),vtype(3,3), vtype(4,4),vtype(5,5),vtype(6,6),vtype(7,7));
		simd vec2(vtype(1,1),vtype(1,1),vtype(2,2),vtype(3,3), vtype(4,4),vtype(5,5),vtype(6,6),vtype(7,7));
		simd vec3;

		vec3 = vec1 / vec2;

//		simd res(vtype(0,0),vtype(0,2),vtype(0,8),vtype(0,18));
//		for(auto i = 0ul; i < vec3.size(); ++i) EXPECT_TRUE(vec3[i] == res[i]);
	}
}


TEST(CVectorFunctions,exp)
{
	{
		using type = boost::numeric::simd::double4;
		type vec1(0.0);
		type res(1);
		type vec2;

		vec2 = boost::numeric::simd::exp(vec1);

		for(int i = 0; i < 4; i++){
			EXPECT_DOUBLE_EQ(res[i], vec2[i]);
		}
	}


	{
		using type = boost::numeric::simd::float16;
		type vec1(0.0);
		type res(1.0);
		type vec2;

		vec2 = boost::numeric::simd::exp(vec1);

		for(int i = 0; i < 8; i++){
			EXPECT_FLOAT_EQ(res[i], vec2[i]);
		}
	}


	{
		using type = boost::numeric::simd::float8;
		type vec1(0.0);
		type res(1.0);
		type vec2;

		vec2 = boost::numeric::simd::exp(vec1);

		for(int i = 0; i < 8; i++){
			EXPECT_FLOAT_EQ(res[i], vec2[i]);
		}
	}

	{
		using type = boost::numeric::simd::float4;
		type vec1(0.0);
		type res(1.0);
		type vec2;

		vec2 = boost::numeric::simd::exp(vec1);

		for(int i = 0; i < 4; i++){
			EXPECT_FLOAT_EQ(res[i], vec2[i]);
		}
	}
}

TEST(CVectorFunctions,sum)
{

	{
		using type = boost::numeric::simd::double4;
		type vec1(1);
		EXPECT_DOUBLE_EQ(vec1.size(), boost::numeric::simd::sum(vec1));
	}

	{
		using type = boost::numeric::simd::float16;
		type vec1(1);
		EXPECT_FLOAT_EQ(vec1.size(), boost::numeric::simd::sum(vec1));
	}

	{
		using type = boost::numeric::simd::float8;
		type vec1(1);
		EXPECT_FLOAT_EQ(vec1.size(), boost::numeric::simd::sum(vec1));
	}

	{
		using type = boost::numeric::simd::float4;
		type vec1(1);
		EXPECT_FLOAT_EQ(vec1.size(), boost::numeric::simd::sum(vec1));
	}

}

TEST(CVectorFunctions,sqrt)
{
	{
		using type = boost::numeric::simd::float16;
		type vec1, ref;
		for(auto i = 0u; i < vec1.size(); ++i) { vec1[i] = i*i; ref[i] = i; }


		auto vec2 = boost::numeric::simd::sqrt(vec1);
		EXPECT_TRUE(vec2 == ref);

		vec2 = boost::numeric::simd::rsqrt(vec1);

		vec2 = 1.0f;


		EXPECT_EQ(boost::numeric::simd::sum(vec2), vec2.size());
	}


	{
		using type = boost::numeric::simd::float8;
		type vec1, ref;
		for(auto i = 0u; i < vec1.size(); ++i) { vec1[i] = i*i; ref[i] = i; }

		auto vec2 = boost::numeric::simd::sqrt(vec1);
		EXPECT_TRUE(vec2 == ref);

		vec2 = boost::numeric::simd::rsqrt(vec1);

		vec2 = 1.0f;


		EXPECT_EQ(boost::numeric::simd::sum(vec2), vec2.size());
	}

	{
		using type = boost::numeric::simd::double4;
		type vec1(1*1, 2*2, 3*3, 4*4);

		type vec2 = boost::numeric::simd::sqrt(vec1);
		EXPECT_TRUE(vec2 == type(1,2,3,4));

		vec2 = boost::numeric::simd::rsqrt(vec1);
		vec2 = 1.0f;

		EXPECT_EQ(boost::numeric::simd::sum(vec2), vec2.size());
	}

	{
		using type = boost::numeric::simd::float4;
		type vec1(1*1, 2*2, 3*3, 4*4);

		type vec2 = boost::numeric::simd::sqrt(vec1);
		EXPECT_TRUE(vec2 == type(1,2,3,4));

		vec2 = boost::numeric::simd::rsqrt(vec1);
		vec2 = 1.0f;

		EXPECT_EQ(boost::numeric::simd::sum(vec2), vec2.size());

	}


	{
		using type = boost::numeric::simd::double2;
		type vec1(1*1, 2*2);

		type vec2 = boost::numeric::simd::sqrt(vec1);
		EXPECT_TRUE(vec2 == type(1,2));

//		vec2 = boost::numeric::simd::rsqrt(vec1);
		vec2 = 1.0f;

		EXPECT_EQ(boost::numeric::simd::sum(vec2), vec2.size());
	}



	{
		using type = boost::numeric::simd::float8;
		type vec1(1*1,2*2,3*3,4*4,5*5,6*6, 7*7,8*8);

		auto vec2 = boost::numeric::simd::sqrt(vec1);
		EXPECT_TRUE(vec2 == type(1,2,3,4,5,6,7,8));

		vec2 = boost::numeric::simd::rsqrt(vec1);

		vec2 = 1.0f;


		EXPECT_EQ(boost::numeric::simd::sum(vec2), vec2.size());
	}

	{
		using type = boost::numeric::simd::double4;
		type vec1(1*1, 2*2, 3*3, 4*4);

		type vec2 = boost::numeric::simd::sqrt(vec1);
		EXPECT_TRUE(vec2 == type(1,2,3,4));

		vec2 = boost::numeric::simd::rsqrt(vec1);
		vec2 = 1.0f;

		EXPECT_EQ(boost::numeric::simd::sum(vec2), vec2.size());
	}

	{
		using type = boost::numeric::simd::float4;
		type vec1(1*1, 2*2, 3*3, 4*4);

		type vec2 = boost::numeric::simd::sqrt(vec1);
		EXPECT_TRUE(vec2 == type(1,2,3,4));

		vec2 = boost::numeric::simd::rsqrt(vec1);
		vec2 = 1.0f;

		EXPECT_EQ(boost::numeric::simd::sum(vec2), vec2.size());

	}


	{
		using type = boost::numeric::simd::double2;
		type vec1(1*1, 2*2);

		type vec2 = boost::numeric::simd::sqrt(vec1);
		EXPECT_TRUE(vec2 == type(1,2));

//		vec2 = boost::numeric::simd::rsqrt(vec1);
		vec2 = 1.0f;

		EXPECT_EQ(boost::numeric::simd::sum(vec2), vec2.size());
	}

}
