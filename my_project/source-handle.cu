Status Handle::gemm_universal(

    GemmUniversalMode mode,                   /// indicates the mode in which the kUniversal GEMM is launched
  
    int M,                                    /// GEMM M dimension
    int N,                                    /// GEMM N dimension
    int K,                                    /// GEMM K dimension
    
    int cluster_m,                            /// cluster shape M dimension
    int cluster_n,                            /// cluster shape N dimension
    int cluster_k,                            /// cluster shape K dimension
    int cluster_m_fallback,                   /// Fallback cluster shape M dimension
    int cluster_n_fallback,                   /// Fallback cluster shape N dimension
    int cluster_k_fallback,                   /// Fallback cluster shape K dimension
    
  
    NumericTypeID element_compute,            /// Data type of internal accumulation
  
    NumericTypeID element_scalar,             /// Data type of alpha/beta scalars
  
    void const *alpha,                        /// Pointer to alpha scalar
  
    NumericTypeID element_A,                  /// Data type of A matrix elements
    LayoutTypeID layout_A,                    /// Layout of A matrix
    ComplexTransform transform_A,             /// Complex transformation applied to A matrix - ignored for real-valued matrices
    void const * ptr_A,                       /// Pointer to A matrix in Global Memory
    int64_t lda,                              /// Leading dimension of A matrix
  
    NumericTypeID element_B,                  /// Data type of B matrix elements
    LayoutTypeID layout_B,                    /// Layout of B matrix
    ComplexTransform transform_B,             /// Complex transformation applied to B matrix - ignored for real-valued matrices
    void const * ptr_B,                       /// Pointer to B matrix in Global Memory
    int64_t ldb,                              /// Leading dimension of B matrix
  
    void const * beta,                        /// Pointer to beta scalar
  
    NumericTypeID element_C,                  /// Data type of C matrix
    LayoutTypeID layout_C,                    /// Layout of D matrix
    void const * ptr_C,                       /// Pointer to C matrix
    int64_t ldc,                              /// Leading dimension of C matrix
  
    NumericTypeID element_D,                  /// Data type of D matrix
    LayoutTypeID layout_D,                    /// Layout of D matrix
    void * ptr_D,                             /// Pointer to D matrix
    int64_t ldd,                              /// Leading dimension of D matrix
  
    int batch_count,                          /// Batch count or number of split-K slices
  
    int64_t batch_stride_A,                   /// Batch stride of A operand
    int64_t batch_stride_B,                   /// Batch stride of B operand
    int64_t batch_stride_C,                   /// Batch stride of C operand
    int64_t batch_stride_D                    /// Batch stride of D operand
  ) {
  
    //
    // Find the operation
    //
  
    GemmFunctionalKey key(
      provider_,
      GemmKind::kUniversal,
      element_compute,
      element_scalar,
      element_A,
      layout_A,
      transform_A,
      element_B,
      layout_B,
      transform_B,
      element_C,
      layout_C,
      element_D,
      layout_D
    );
  
    auto operators_it = Singleton::get().operation_table.gemm_operations.find(key);
  
    if (operators_it == Singleton::get().operation_table.gemm_operations.end()) {
      return cutlass::Status::kErrorNotSupported;
    }
  
    if (operators_it->second.empty()) {
      return cutlass::Status::kErrorNotSupported;
    }
  
    //
    // Compute the largest alignment restriction the kernel can satisfy.
    //
  
    // Maximum alignment expectation among all kernels (in units of bytes)
    int const kMaximumAlignmentSize = 16;
  
    void const *ptr_A_check = ptr_A;
    void const *ptr_B_check = ptr_B;
    void const *ptr_C_check = ptr_C;
    void *      ptr_D_check = ptr_D;
  
    // Ignore alignment of pointers to pointers. We can't check this from the host,
    // as each batch index has its own pointer in device memory.
    if (mode == GemmUniversalMode::kArray) {
      ptr_A_check = nullptr;
      ptr_B_check = nullptr;
      ptr_C_check = nullptr;
      ptr_D_check = nullptr;
    }
  
    int alignment = gemm_problem_alignment(
      M, N, K,
      element_A, ptr_A_check, lda, 0,
      element_B, ptr_B_check, ldb, 0,
      element_C, ptr_C_check, ldc, 0,
      ptr_D_check, ldd, 0, kMaximumAlignmentSize
    );
  
    //
    // Find the best kernel in descending order of preference.
    //
  
    GemmPreferenceKey preference_key(compute_capability(), alignment);
  
    Operation const *operation = find_gemm_operation(operators_it, preference_key);
  
    if (!operation) {
      return cutlass::Status::kErrorNotSupported;
    }
  
    last_operation_ = operation;
  
    //
    // Configure operation
    //
  
    GemmUniversalConfiguration configuration{
      mode,
      {M, N, K},
      {cluster_m, cluster_n, cluster_k}, 
      {cluster_m_fallback, cluster_n_fallback, cluster_k_fallback}, 
      batch_count,
      lda,
      ldb,
      ldc,
      ldd
    };
  
    // Query host work space size
    uint64_t host_workspace_size_needed = operation->get_host_workspace_size(&configuration);
  
    if (uint64_t(kHostWorkspaceSize) < host_workspace_size_needed) {
      return cutlass::Status::kErrorNotSupported;
    }
  
    char host_workspace[kHostWorkspaceSize];
  
    GemmUniversalArguments arguments{
      {M, N, K},
      {cluster_m, cluster_n, cluster_k}, 
      {cluster_m_fallback, cluster_n_fallback, cluster_k_fallback}, 
      batch_count,
      ptr_A,
      ptr_B,
      ptr_C,
      ptr_D,
      alpha,
      beta,
      scalar_pointer_mode_,
      lda,
      ldb,
      ldc,
      ldd,
      batch_stride_A,
      batch_stride_B,
      batch_stride_C,
      batch_stride_D
    };
  
    // Query device workspace size
    uint64_t device_workspace_size_needed = operation->get_device_workspace_size(&configuration, &arguments);
  
    if (uint64_t(workspace_size_) < device_workspace_size_needed) {
      return cutlass::Status::kErrorNotSupported;
    }
  
    // Initialize host and device workspaces
    Status status = operation->initialize(
      &configuration,
      host_workspace,
      workspace_,
      stream_);
  
    if (status != cutlass::Status::kSuccess) {
      return status;
    }
  
    // Run the operator
  
    return operation->run(&arguments, host_workspace, workspace_, stream_);
  }
  