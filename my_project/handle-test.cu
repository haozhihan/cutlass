#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

// CUTLASS Library API
// #include "cutlass/library/library.h"

#include "cutlass/library/handle.h"

// Host tensor 用于管理主机与设备内存
#include "cutlass/util/host_tensor.h"

// 用于参考 GEMM 结果（可选）
// #include "cutlass/util/reference/host/gemm.h"

int main() {

  // 定义 GEMM 问题尺寸：D = alpha * A*B + beta * C
  int m = 128;
  int n = 128;
  int k = 128;

  // 创建 HostTensor 用于 A, B, C, D 四个矩阵
  // 这里使用 RowMajor 布局，数据类型为 float
  cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_A({m, k});
  cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_B({k, n});
  cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_C({m, n});
  cutlass::HostTensor<float, cutlass::layout::RowMajor> tensor_D({m, n});

  // 初始化 A 和 B，C 可设为 0
  for (int i = 0; i < tensor_A.size(); ++i) {
    tensor_A.host_data()[i] = 1.0f;
  }
  for (int i = 0; i < tensor_B.size(); ++i) {
    tensor_B.host_data()[i] = 2.0f;
  }
  for (int i = 0; i < tensor_C.size(); ++i) {
    tensor_C.host_data()[i] = 0.0f;
  }

  // 将数据拷贝到设备
  tensor_A.sync_device();
  tensor_B.sync_device();
  tensor_C.sync_device();

  // 创建 CUTLASS library 的 handle 对象
  cutlass::library::Handle handle;

  // 选择 GEMM 模式，此处使用普通 GEMM 模式
  cutlass::library::GemmUniversalMode mode = cutlass::library::GemmUniversalMode::kGemm;

  //
  // 构造 gemm_universal 的参数结构体
  //
  // 注意：GemmUniversalArguments 的具体成员变量可能会随着 CUTLASS 版本有所调整，
  // 以下示例构造了一个典型的参数结构体。
  //

  cutlass::library::GemmUniversalArguments gemm_args;

  // 设置问题尺寸
  gemm_args.problem_size = cutlass::gemm::GemmCoord(m, n, k);

  // 设置标量系数
  gemm_args.alpha = 1.0;
  gemm_args.beta  = 0.0;

  // 设置 A, B, C, D 指针
  gemm_args.A = static_cast<void const*>(tensor_A.device_data());
  gemm_args.B = static_cast<void const*>(tensor_B.device_data());
  gemm_args.C = static_cast<void const*>(tensor_C.device_data());
  gemm_args.D = static_cast<void*>(tensor_D.device_data());

  // 设置矩阵 strides (以 RowMajor 布局为例)
  gemm_args.lda = k;   // A: m x k
  gemm_args.ldb = n;   // B: k x n
  gemm_args.ldc = n;   // C: m x n
  gemm_args.ldd = n;   // D: m x n

  // 对于一些 kernel，可能还需要设置额外的参数，这里以简单的情况为例

  // 调用 gemm_universal 执行 GEMM
  cutlass::Status status = handle.gemm_universal(mode, gemm_args, /*split_k_slices=*/false);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS gemm_universal 调用失败: " 
              << cutlassGetStatusString(status) << std::endl;
    return -1;
  }

  // 将结果从设备同步回主机
  tensor_D.sync_host();

  // 简单打印部分结果以验证
  std::cout << "结果矩阵 D[0] = " << tensor_D.host_data()[0] << std::endl;

  return 0;
}
