# C++ SIMD types for Intel CPUs
 
This is a header-only library which uses Intel intrinsics for (complex) SIMD data types.
It automatically selects the right types and intrinsics according to the enabled flags such `-mavx`

## Build Requirements

- C++14 conform compiler

### Build Requirements for Unit-Tests 

- Google Unit-Test

## Library Usage

Simply include `boost/numeric/simd.h` and use `-msse`, `-mavx` or `-mavx512f` flags

You can always use
- `double2`, `double4`, `double8`
- `float4`, `float8`, `float16`
- `cdouble2`, `cdouble4`, `cdouble8` as complex types with values encoded in double precision
- `cfloat4`, `cfloat8`, `cfloat16` as complex types with values encoded in single precision

even if your machine does not support the intrinsics with that length.
Those types will mimic the intrinsics.

## Example

```cpp
int main(){
  using namespace boost::numeric;
  auto a = simd::double4 {1,2,3,4};
  auto b = simd::double4 {5,6,7,8};
  auto c = a + b / 2;
}

```
