#include <cutlass/numeric_types.h>
#include <cutlass/util/host_tensor.h>



#include "cute/tensor.hpp"
#include "cute/atom/mma_atom.hpp"


#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "packed_stride.hpp"

#include "helper.h"



///////////////////////////////////////////////////////////////////////////////////////////////////


// Gemm operator cutlass_tensorop_d884gemm_128x64_16x3_nn_align1
using cutlass_tensorop_d884gemm_128x64_16x3_nn_align1_base =
  typename cutlass::gemm::kernel::DefaultGemmUniversal<
    double, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,    // transposed B operand
    double, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 1,    // transposed A operand
    double, cutlass::layout::RowMajor,
    double,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 16>,
    cutlass::gemm::GemmShape<64, 32, 16>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    
    cutlass::epilogue::thread::LinearCombination<
      double,
      1,
      double,
      double
    >
,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    3,
    cutlass::arch::OpMultiplyAdd
>::GemmKernel;

// Define named type
struct cutlass_tensorop_d884gemm_128x64_16x3_nn_align1 :
  public cutlass_tensorop_d884gemm_128x64_16x3_nn_align1_base { };


///////////////////////////////////////////////////////////////////////////////////////////////////


int main() {

  using GemmKernel = cutlass_tensorop_d884gemm_128x64_16x3_nn_align1;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Define the GEMM operation
  // using Gemm = 


  //
  // Define the problem size
  //
  int batch = 1;

  int M = 512;
  int N = 256;
  int K = 128;

  float alpha = 1.25f;
  float beta = -1.25f;

  //
  // Allocate device memory
  //

  cutlass::HostTensor<double, cutlass::layout::ColumnMajor> A({M, K});
  cutlass::HostTensor<double, cutlass::layout::ColumnMajor> B({K, N});
  cutlass::HostTensor<double, cutlass::layout::ColumnMajor> C({M, N});

  
  // auto strideA = cutlass::make_cute_packed_stride(Stride_A{}, {m, k, batch});
  // auto strideB = cutlass::make_cute_packed_stride(Stride_B{}, {n, k, batch});
  // auto strideC = cutlass::make_cute_packed_stride(Stride_C{}, {m, n, batch});



  // int split_k_slices = 1;

  // typename Gemm::Arguments arguments{{M, N, K, batch},       // <- problem size of matrix multiplication
  //                                    A.device_ref(),  // <- reference to matrix A on device
  //                                    B.device_ref(),  // <- reference to matrix B on device
  //                                    C.device_ref(),  // <- reference to matrix C on device
  //                                    C.device_ref(),  // <- reference to matrix D on device
  //                                    {alpha, beta},  // <- tuple of alpha and beta
  //                                    split_k_slices}; 

  













                                             
  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);


  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;
  
  // Check the problem size is supported or not 
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);


  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);


  // 创建 CUDA 事件用于计时
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 记录起始事件
  cudaEventRecord(start);
  


  //
  // Launch GEMM on the device
  //

  // status = gemm_op({
  //   {M, N, K},
  //   {ptrA, lda},            // TensorRef to A device tensor
  //   {ptrB, ldb},            // TensorRef to B device tensor
  //   {ptrC, ldc},            // TensorRef to C device tensor
  //   {ptrD, ldd},            // TensorRef to D device tensor - may be the same as C
  //   {alpha, beta}           // epilogue operation arguments
  // });

  status = gemm_op();
  CUTLASS_CHECK(status);

  // 记录结束事件
  cudaEventRecord(stop);

  // 同步等待 GPU 完成任务
  cudaEventSynchronize(stop);

  // 计算并输出经过的时间（单位：毫秒）
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "GEMM 执行时间: " << milliseconds << " ms" << std::endl;

  // 检查返回状态
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "GEMM 操作失败！" << std::endl;
    return -1;
  }


  return 0;
}