# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)

option(USE_ROCM "Use ROCm" OFF)
option(USE_HIPBLASLT "Use HIPBLASLT" ON)
# Temp unsupport  aottriton\ck backend and Use ROCBLAS
option(USE_ROCBLAS "Use ROCBLAS" OFF)

if(NOT USE_ROCM)
  if(((EXISTS "/opt/dtk/") OR (EXISTS $ENV{ROCM_PATH})) AND NOT (EXISTS "/bin/nvcc"))
    message("hcu detected.")
    set(USE_ROCM ON)
  endif()
endif()

if (USE_ROCM)
  add_compile_definitions(__HIP_CLANG_ONLY__=1)
  if (NOT USE_HIPBLASLT AND NOT USE_ROCBLAS)
    message(FATAL_ERROR "Need specify at least one GEMM library to use: HIPBLASLT or ROCBLAS")
  endif()
  unset(USE_CUDA)
else()
  set(USE_CUDA TRUE)
endif()


# Language options
if(USE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if (CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
      set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90 100 120)
    else ()
      set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)
    endif()
  endif()
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
  endif()
  
  # Hide non-necessary symbols in shared object.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libtransformer_engine.version")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libtransformer_engine.version")
  
  # Transformer Engine library
  project(transformer_engine LANGUAGES CUDA CXX)
  
  # CUDA Toolkit
  find_package(CUDAToolkit REQUIRED)
  if (CUDAToolkit_VERSION VERSION_LESS 12.0)
    message(FATAL_ERROR "CUDA 12.0+ is required, but found CUDA ${CUDAToolkit_VERSION}")
  endif()
  
  # cuDNN frontend API
  set(CUDNN_FRONTEND_INCLUDE_DIR
      "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cudnn-frontend/include")
  if(NOT EXISTS "${CUDNN_FRONTEND_INCLUDE_DIR}")
      message(FATAL_ERROR
              "Could not find cuDNN frontend API at ${CUDNN_FRONTEND_INCLUDE_DIR}. "
              "Try running 'git submodule update --init --recursive' "
              "within the Transformer Engine source.")
  endif()
  include(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cudnn-frontend/cmake/cuDNN.cmake)
else()
  set(CMAKE_CXX_STANDARD 17)
  project(transformer_engine LANGUAGES HIP CXX)
  
  # Disable Asserts In Code (Can't use asserts on HIP stack.)
  add_definitions(-DNDEBUG)
  add_definitions(-DUSE_ROCM)
  # Change clang++ to hipcc 
  SET(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")  
  
  if(NOT DEFINED ENV{NVTE_ROCM_ARCH})
    SET(CMAKE_HIP_ARCHITECTURES gfx906;gfx926;gfx928;gfx936)
  else()
    SET(CMAKE_HIP_ARCHITECTURES $ENV{NVTE_ROCM_ARCH})
  endif()
  
  # build error will be dup-ed parallel-jobs times
  # set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -parallel-jobs=4")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -g")
  endif()
  
  list(APPEND CMAKE_MODULE_PATH "/opt/dtk")
endif()

set(message_line "-------------------------------------------------------------")
message("${message_line}")
message(STATUS "USE_ROCM ${USE_ROCM}")
if(USE_ROCM)
  message(STATUS "CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")
  message(STATUS "USE_HIPBLASLT ${USE_HIPBLASLT} USE_ROCBLAS ${USE_ROCBLAS}")
endif()

# Python
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Configure Transformer Engine library
include_directories(${PROJECT_SOURCE_DIR}/..)
set(transformer_engine_SOURCES)

if(USE_CUDA)
  list(APPEND transformer_engine_SOURCES
       cudnn_utils.cpp
       transformer_engine.cpp
       common.cu
       transpose/cast_transpose.cu
       transpose/transpose.cu
       transpose/cast_transpose_fusion.cu
       transpose/transpose_fusion.cu
       transpose/multi_cast_transpose.cu
       activation/gelu.cu
       fused_attn/fused_attn_f16_max512_seqlen.cu
       fused_attn/fused_attn_f16_arbitrary_seqlen.cu
       activation/relu.cu
       activation/swiglu.cu
       fused_attn/fused_attn_fp8.cu
       fused_attn/fused_attn.cpp
       fused_attn/utils.cu
       gemm/cublaslt_gemm.cu
       normalization/common.cpp
       normalization/layernorm/ln_api.cpp
       normalization/layernorm/ln_bwd_semi_cuda_kernel.cu
       normalization/layernorm/ln_fwd_cuda_kernel.cu
       normalization/rmsnorm/rmsnorm_api.cpp
       normalization/rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu
       normalization/rmsnorm/rmsnorm_fwd_cuda_kernel.cu
       permutation/permutation.cu
       util/cast.cu
       util/padding.cu
       util/cuda_driver.cpp
       util/cuda_runtime.cpp
       util/rtc.cpp
       util/system.cpp
       swizzle/swizzle.cu
       fused_softmax/scaled_masked_softmax.cu
       fused_softmax/scaled_upper_triang_masked_softmax.cu
       fused_softmax/scaled_aligned_causal_masked_softmax.cu
       fused_rope/fused_rope.cu
       recipe/delayed_scaling.cu
       comm_gemm_overlap/userbuffers/ipcsocket.cc
       comm_gemm_overlap/userbuffers/userbuffers-host.cpp
       comm_gemm_overlap/userbuffers/userbuffers.cu
       comm_gemm_overlap/comm_gemm_overlap.cpp)
  add_library(transformer_engine SHARED ${transformer_engine_SOURCES})
else()
  list(APPEND transformer_engine_SOURCES
       cudnn_utils.cpp
       transformer_engine.cpp
       common.cu
       transpose/cast_transpose.cu
       transpose/transpose.cu
       transpose/cast_transpose_fusion.cu
       transpose/transpose_fusion.cu
       transpose/multi_cast_transpose.cu
       activation/gelu.cu
       fused_attn/fused_attn_f16_max512_seqlen.cu
       fused_attn/fused_attn_f16_arbitrary_seqlen.cu
       activation/relu.cu
       activation/swiglu.cu
       fused_attn/fused_attn_fp8.cu
       fused_attn/fused_attn.cpp
       fused_attn/utils.cu
       gemm/cublaslt_gemm.cu
       normalization/common.cpp
       normalization/layernorm/ln_api.cpp
       normalization/layernorm/ln_bwd_semi_cuda_kernel.cu
       normalization/layernorm/ln_fwd_cuda_kernel.cu
       normalization/rmsnorm/rmsnorm_api.cpp
       normalization/rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu
       normalization/rmsnorm/rmsnorm_fwd_cuda_kernel.cu
       permutation/permutation.cu
       util/cast.cu
       util/padding.cu
       util/cuda_driver.cpp
       util/cuda_runtime.cpp
       util/rtc.cpp
       util/system.cpp
       swizzle/swizzle.cu
       fused_softmax/scaled_masked_softmax.cu
       fused_softmax/scaled_upper_triang_masked_softmax.cu
       fused_softmax/scaled_aligned_causal_masked_softmax.cu
       fused_rope/fused_rope.cu
       recipe/delayed_scaling.cu
       comm_gemm_overlap/userbuffers/ipcsocket.cc
       comm_gemm_overlap/userbuffers/userbuffers-host.cpp
       comm_gemm_overlap/userbuffers/userbuffers.cu
       comm_gemm_overlap/comm_gemm_overlap.cpp)
  # process source code files
  message("${message_line}")
  message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

  set(TE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
  set(THIRDPARTY ${TE}/3rdparty)
  list(APPEND CMAKE_MODULE_PATH "${THIRDPARTY}/hipify_torch/cmake")
  include(Hipify)
  message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

  set(header_include_dir
      ${CMAKE_CURRENT_SOURCE_DIR}/comm_gemm_overlap/userbuffers
      ${CMAKE_CURRENT_SOURCE_DIR}/activation 
      ${CMAKE_CURRENT_SOURCE_DIR}/include 
      ${CMAKE_CURRENT_SOURCE_DIR}/transpose
      ${CMAKE_CURRENT_SOURCE_DIR}/util
      ${CMAKE_CURRENT_SOURCE_DIR}/normalization
      ${CMAKE_CURRENT_SOURCE_DIR}/normalization/rmsnorm
      ${CMAKE_CURRENT_SOURCE_DIR}/normalization/layernorm 
      ${CMAKE_CURRENT_SOURCE_DIR})
  message(STATUS "HIPIFY CUDA_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
  message(STATUS "HIPIFY HEADER_INCLUDE_DIR: ${header_include_dir}")
  hipify(CUDA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
      HEADER_INCLUDE_DIR ${header_include_dir}
      IGNORES "*/amd_detail/*"
      IGNORES "*/fused_attn/*"
      CUSTOM_MAP_FILE "${TE}/hipify_custom_map.json"
  )
  get_hipified_list("${transformer_engine_SOURCES}" te_hip_sources)
  message("${message_line}")
  message(STATUS "nvte hipified sources: ${te_hip_sources}")

  add_library(transformer_engine SHARED ${te_hip_sources})
endif()

# Configure dependencies
if (USE_CUDA)
  target_include_directories(transformer_engine PUBLIC
                             "${CMAKE_CURRENT_SOURCE_DIR}/include")
  
  # Configure dependencies
  target_link_libraries(transformer_engine PUBLIC
                        CUDA::cublas
                        CUDA::cudart)
  target_include_directories(transformer_engine PRIVATE
                             ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  target_include_directories(transformer_engine PRIVATE "${CUDNN_FRONTEND_INCLUDE_DIR}")
else()
  # Aotriton is currently unsupported 
  set(AotritonAndCk_fused_attn "unsupported")

  find_package(hip)
  list(APPEND transformer_engine_LINKER_LIBS hip::host hip::device roctx64)
  if(USE_HIPBLASLT)
    find_package(hipblaslt)
    find_package(hipblas REQUIRED PATHS ${ROCM_PATH})
    target_compile_definitions(transformer_engine PUBLIC USE_HIPBLASLT)
    list(APPEND transformer_engine_LINKER_LIBS roc::hipblaslt hipblas)
  endif()
  if(USE_ROCBLAS)
    find_package(rocblas)
    target_compile_definitions(transformer_engine PUBLIC USE_ROCBLAS)
    list(APPEND transformer_engine_LINKER_LIBS roc::rocblas)
  endif()
  target_link_libraries(transformer_engine PUBLIC ${transformer_engine_LINKER_LIBS})
endif()


# Compiling Userbuffers with native MPI bootstrapping requires linking against MPI
option(NVTE_UB_WITH_MPI "Bootstrap Userbuffers with MPI" OFF)
if (NVTE_UB_WITH_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(transformer_engine PUBLIC MPI::MPI_CXX)
    target_include_directories(transformer_engine PRIVATE ${MPI_CXX_INCLUDES})
    target_compile_definitions(transformer_engine PUBLIC NVTE_UB_WITH_MPI)
endif()

if (USE_CUDA)
  # Hack to enable dynamic loading in cuDNN frontend
  target_compile_definitions(transformer_engine PUBLIC NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)
endif()

# Helper functions to make header files with C++ strings
function(make_string_header STRING STRING_NAME)
    configure_file(util/string_header.h.in
                   "string_headers/${STRING_NAME}.h"
                   @ONLY)
endfunction()
function(make_string_header_from_file file_ STRING_NAME)
    file(READ "${file_}" STRING)
    configure_file(util/string_header.h.in
                   "string_headers/${STRING_NAME}.h"
                   @ONLY)
endfunction()

# Header files with C++ strings
if(USE_CUDA)
  list(GET CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES 0 cuda_include_path)
  make_string_header("${cuda_include_path}"
                     string_path_cuda_include)
  make_string_header_from_file(transpose/rtc/cast_transpose_fusion.cu
                               string_code_transpose_rtc_cast_transpose_fusion_cu)
  make_string_header_from_file(transpose/rtc/cast_transpose.cu
                               string_code_transpose_rtc_cast_transpose_cu)
  make_string_header_from_file(transpose/rtc/transpose.cu
                               string_code_transpose_rtc_transpose_cu)
  make_string_header_from_file(utils.cuh
                               string_code_utils_cuh)
else()
  make_string_header_from_file(utils_hip.cuh
                               string_code_utils_cuh)
  make_string_header_from_file(transpose/rtc/cast_transpose_fusion.hip
                              string_code_transpose_rtc_cast_transpose_fusion_cu)
  make_string_header_from_file(transpose/rtc/cast_transpose.hip
                              string_code_transpose_rtc_cast_transpose_cu)
  make_string_header_from_file(transpose/rtc/transpose.hip
                              string_code_transpose_rtc_transpose_cu)
  make_string_header_from_file(amd_detail/hip_float8.h
                               string_code_amd_detail_hip_float8_h)
  make_string_header_from_file(amd_detail/hip_f8_impl.h
                               string_code_amd_detail_hip_f8_impl_h)
endif()


make_string_header_from_file(util/math.h
                             string_code_util_math_h)
target_include_directories(transformer_engine PRIVATE
                           "${CMAKE_CURRENT_BINARY_DIR}/string_headers")

# Compiler options
set_source_files_properties(fused_softmax/scaled_masked_softmax.cu
                            fused_softmax/scaled_upper_triang_masked_softmax.cu
                            fused_softmax/scaled_aligned_causal_masked_softmax.cu
                            PROPERTIES
                            COMPILE_OPTIONS "--use_fast_math")
option(NVTE_BUILD_ACTIVATION_WITH_FAST_MATH "Compile activation kernels with --use_fast_math option" OFF)
if (NVTE_BUILD_ACTIVATION_WITH_FAST_MATH)
  set_source_files_properties(activation/gelu.cu
                              activation/relu.cu
                              activation/swiglu.cu
                              PROPERTIES
                              COMPILE_OPTIONS "--use_fast_math")
endif()

if(USE_CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
else()
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -O3")
  set(HIP_HCC_FLAGS "${CMAKE_HIP_FLAGS} -mavx2 -mf16c -mfma -std=c++17")
  # Ask hcc to generate device code during compilation so we can use
  # host linker to link.
  set(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} -fno-gpu-rdc -Wno-defaulted-function-deleted")
  foreach(rocm_arch ${CMAKE_HIP_ARCHITECTURES})
    # if CMAKE_CXX_FLAGS has --offload-arch set already, better to rm first
    set(HIP_HCC_FLAGS "${HIP_HCC_FLAGS} --offload-arch=${rocm_arch}")
  endforeach()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_HCC_FLAGS}")
endif()

# Number of parallel build jobs
if(ENV{MAX_JOBS})
  set(BUILD_JOBS_STR "$ENV{MAX_JOBS}")
elseif(ENV{NVTE_BUILD_MAX_JOBS})
  set(BUILD_JOBS_STR "$ENV{NVTE_BUILD_MAX_JOBS}")
else()
  set(BUILD_JOBS_STR "max")
endif()
message(STATUS "Parallel build jobs: ${BUILD_JOBS_STR}")

# Number of threads per parallel build job
set(BUILD_THREADS_PER_JOB $ENV{NVTE_BUILD_THREADS_PER_JOB})
if (NOT BUILD_THREADS_PER_JOB)
  set(BUILD_THREADS_PER_JOB 1)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads ${BUILD_THREADS_PER_JOB}")
message(STATUS "Threads per parallel build job: ${BUILD_THREADS_PER_JOB}")

# Install library
install(TARGETS transformer_engine DESTINATION .)
