
# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#
# This example depends on the CUTLASS Library
#
# if (CUTLASS_ENABLE_LIBRARY)

# # Planar Complex GEMM example
# cutlass_example_add_executable(
#   10_planar_complex
#   planar_complex.cu
# )

# target_link_libraries(
#   10_planar_complex
#   PRIVATE
#   cutlass_lib
#   cutlass_tools_util_includes
#   cuda
# )

# endif()


cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cutlass_lib_example VERSION 0.2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CUDA_COMPILER)
  # 全局添加编译选项
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

if (CUTLASS_DIR)
  message(STATUS "Using CUTLASS specified at ${CUTLASS_DIR}.")
  list(APPEND CMAKE_PREFIX_PATH ${CUTLASS_DIR})
endif()

find_package(NvidiaCutlass 2.0 REQUIRED)

message(STATUS "CUTLASS: ${NvidiaCutlass_DIR}")

add_executable(hhz_example)


target_sources(hhz_example PRIVATE planar_complex.cu)


target_include_directories(
  hhz_example
  PRIVATE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  # nvidia::cutlass::library::includes
  )

target_link_libraries(
  hhz_example 
  PRIVATE
  # nvidia::cutlass::cutlass
  nvidia::cutlass::library
  # nvidia::cutlass::tools::util
  )
