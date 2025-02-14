#include <iostream>
#include <cassert>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"

#include "cutlass/numeric_types.h"
#include "packed_stride.hpp"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"

#define CUTLASS_CHECK(x) assert((x) == cutlass::Status::kSuccess)

using cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmem_epilogue =
  typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    cutlass::epilogue::collective::EpilogueTileAuto,
    cutlass::half_t, cutlass::half_t,
    void, cutlass::layout::ColumnMajor, 8,
    cutlass::half_t, cutlass::layout::ColumnMajor, 8,
    cutlass::epilogue::NoSmemWarpSpecialized,
    cutlass::epilogue::fusion::LinearCombination<
      cutlass::half_t,
      cutlass::half_t,
      void,
      cutlass::half_t
    >
  >::CollectiveOp;

using cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmem_mainloop =
  typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t,
    cute::Shape<cute::_128, cute::_128, cute::_64>,
    cute::Shape<cute::_2, cute::_1, cute::_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmem_epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong
  >::CollectiveOp;

// Gemm operator cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmem
using cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmem_base = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int,int,int,int>,
    cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmem_mainloop,
    cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmem_epilogue,
    cutlass::gemm::PersistentScheduler>;

// Define named type
struct cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmem :
  public cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmem_base { };


  
int main() {
  using namespace cute;

  using CollectiveMainloop = cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmem_mainloop;
  using CollectiveEpilogue = cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmem_epilogue;
  using GemmKernel = cutlass3x_sm90_tensorop_h64x128x16gemm_f16_f16_f16_void_f16_128x128x64_2x1x1_0_ttn_align8_warpspecialized_pingpong_epi_nosmem;

  int batch = 1;
  int m = 4096, n = 4096, k = 4096;
  cutlass::half_t *a_ptr = nullptr, *b_ptr = nullptr, *c_ptr = nullptr;
  cudaMalloc(&a_ptr, batch * m * k * sizeof(*a_ptr));
  cudaMalloc(&b_ptr, batch * n * k * sizeof(*b_ptr));
  cudaMalloc(&c_ptr, batch * n * m * sizeof(*c_ptr));

#if 0
  std::vector<cutlass::half_t> data_in(4);
  for (int i = 0; i < data_in.size(); ++i)
    data_in[i] = cutlass::half_t(i);
  cudaMemcpy(a_ptr, data_in.data(), m * k * sizeof(*a_ptr), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  for (int i = 0; i < data_in.size(); ++i)
    data_in[i] = cutlass::half_t(i + 1);
  cudaMemcpy(b_ptr, data_in.data(), n * k * sizeof(*b_ptr), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
#else
  auto init_val = cutlass::half_t(1);
  cuMemsetD16((CUdeviceptr)a_ptr, (uint16_t&)init_val, m * k * batch);
  cuMemsetD16((CUdeviceptr)b_ptr, (uint16_t&)init_val, n * k * batch);
#endif

  using Stride_A = typename GemmKernel::StrideA;
  using Stride_B = typename GemmKernel::StrideB;
  using Stride_C = typename GemmKernel::StrideC;

  typename GemmKernel::ProblemShape prob_shape{m, n, k, batch};
  auto strideA = cutlass::make_cute_packed_stride(Stride_A{}, {m, k, batch});
  auto strideB = cutlass::make_cute_packed_stride(Stride_B{}, {n, k, batch});
  auto strideC = cutlass::make_cute_packed_stride(Stride_C{}, {m, n, batch});

  typename GemmKernel::MainloopArguments mainloop_args{a_ptr, strideA, b_ptr, strideB};
  typename GemmKernel::EpilogueArguments epilogue_args{{cutlass::half_t(1), cutlass::half_t(0)}, c_ptr, strideC, c_ptr, strideC};

  using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  typename GemmKernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, prob_shape, mainloop_args, epilogue_args};
  args.scheduler.raster_order = cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90::RasterOrderOptions::Heuristic; // AlongM:AlongM:Heuristic
  args.scheduler.max_swizzle_size = 8; // 1:2:4:8

  GemmOp gemm_op;
  size_t workspace_size = gemm_op.get_workspace_size(args);
  void *workspace = nullptr;
  if (workspace_size > 0)
    cudaMalloc(&workspace, workspace_size);

  CUTLASS_CHECK(gemm_op.can_implement(args));
  CUTLASS_CHECK(gemm_op.initialize(args, workspace));

  gemm_op.run(args, workspace, nullptr);
  gemm_op.run(args, workspace, nullptr);

  int nstep = 1000;
  cudaEvent_t hStart, hEnd;
  cudaEventCreate(&hStart);
  cudaEventCreate(&hEnd);
  cudaEventRecord(hStart);
  for (int i = 0; i < nstep; ++i)
    gemm_op.run(args, workspace, nullptr);
  cudaEventRecord(hEnd);
  cudaDeviceSynchronize();
  float ms = -1;
  cudaEventElapsedTime(&ms, hStart, hEnd);
  ms /= nstep;

  float tflops = (uint64_t(n) * m * k * batch * 2.0 * 1e-12 / (ms * 1e-3));

  std::vector<cutlass::half_t> hData(n * m * batch);
  cudaMemcpy(hData.data(), c_ptr, hData.size() * sizeof(hData[0]), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  bool passed = true;
  for (int i = 0; i < hData.size(); ++i) {
#if 0
    printf("%g\n", float(hData[i]));
#else
    if (abs(float(hData[i]) - k) > 1e-3) {
      passed = false;
      break;
    }
#endif
  }
  if (passed)
    printf("PASSED with workspace size = %ld! TFlops = %g (cost = %g msec)\n", workspace_size, tflops, ms);
  return 0;
}

