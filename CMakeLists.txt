cmake_minimum_required(VERSION 3.20)

# Select a supported CUDA host compiler before project() enables CUDA.
# With GCC 16 or newer, prefer g++-15; otherwise enable the compiler override
# and use C++20 for CUDA, since newer libstdc++ headers require it.
find_program(_cxx_compiler NAMES $ENV{CXX} c++ g++)
if(_cxx_compiler)
    execute_process(COMMAND ${_cxx_compiler} -dumpversion
                    OUTPUT_VARIABLE _gcc_version OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
endif()
if(_gcc_version VERSION_GREATER_EQUAL 16)
    find_program(_gcc15 NAMES g++-15)
    if(_gcc15)
        set(CMAKE_CUDA_HOST_COMPILER "${_gcc15}")
    else()
        set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler -std=c++20")
    endif()
endif()

project(w4a16_dequant_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_STANDARD 17)
else()
    set(CMAKE_CUDA_STANDARD 20)
endif()
set(CMAKE_CUDA_ARCHITECTURES native)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

enable_testing()

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data)

add_library(w4 INTERFACE)
target_include_directories(w4 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(w4 INTERFACE CUDA::cudart)

add_executable(test_oracle tests/test_oracle.cpp)
target_link_libraries(test_oracle PRIVATE w4)
target_compile_options(test_oracle PRIVATE -Wall -Wextra)
add_test(NAME oracle COMMAND test_oracle)

add_executable(test_gpu tests/test_gpu.cu)
target_link_libraries(test_gpu PRIVATE w4 CUDA::cublas)
target_compile_options(test_gpu PRIVATE -Wall -Wextra)
add_test(NAME gpu COMMAND test_gpu)

add_executable(bench_baseline benchmarks/bench_baseline.cu)
target_link_libraries(bench_baseline PRIVATE w4 CUDA::cublas)
target_compile_options(bench_baseline PRIVATE -Wall -Wextra)
target_compile_definitions(bench_baseline PRIVATE
  W4_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  W4_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

add_executable(bench_fused benchmarks/bench_fused.cu)
target_link_libraries(bench_fused PRIVATE w4 CUDA::cublas)
target_compile_options(bench_fused PRIVATE -Wall -Wextra)
target_compile_definitions(bench_fused PRIVATE
  W4_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  W4_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

add_executable(bench_mx benchmarks/bench_mx.cu)
target_link_libraries(bench_mx PRIVATE w4 CUDA::cublas)
target_compile_options(bench_mx PRIVATE -Wall -Wextra)
target_compile_definitions(bench_mx PRIVATE
  W4_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
  W4_BUILD_TYPE="${CMAKE_BUILD_TYPE}")
