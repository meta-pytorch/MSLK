/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>

namespace mslk::gemm {

#if CUDART_VERSION >= 13000

namespace {

inline void checkCublasStatus(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("cuBLAS API failed with status %d\n", status);
    throw std::logic_error("cuBLAS API failed");
  }
}

} // namespace

at::Tensor bf16x9_gemm(
    at::Tensor A, // FP32
    at::Tensor B, // FP32
    std::optional<at::Tensor> output = std::nullopt) {
  auto m = A.size(0);
  auto n = B.size(0);
  auto k = A.size(1);

  TORCH_CHECK(A.is_cuda() && A.is_contiguous());
  TORCH_CHECK(B.is_cuda() && B.is_contiguous());
  TORCH_CHECK(A.dtype() == at::kFloat);
  TORCH_CHECK(B.dtype() == at::kFloat);

  if (output.has_value()) {
    auto output_tensor = output.value();
    TORCH_CHECK(output_tensor.is_cuda());
    TORCH_CHECK(output_tensor.is_contiguous());
    TORCH_CHECK(
        output_tensor.numel() == m * n,
        "output_tensor.numel=",
        output_tensor.numel(),
        ", m=",
        m,
        ", n=",
        n);
    TORCH_CHECK(output_tensor.options().dtype() == at::kFloat);
  }

  // Use cuBLAS legacy API for FP32 GEMM
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasSetStream(handle, at::cuda::getCurrentCUDAStream());

  // Enable BF16x9 emulation for performance on Tensor Core GPUs
  checkCublasStatus(
      cublasSetMathMode(handle, CUBLAS_FP32_EMULATED_BF16X9_MATH));

  float alpha = 1.0f;
  float beta = 0.0f;

  // Create output tensor
  auto Y = output.value_or(at::empty({m, n}, A.options().dtype(at::kFloat)));

  // PyTorch tensors are row-major, cuBLAS expects column-major
  // We want to compute (in row-major): C[M,N] = A[M,K] @ B[N,K]^T
  // where C[i,j] = sum_k A[i,k] * B[j,k]
  //
  // For cuBLAS (column-major), we compute: C = B @ A^T
  // - B[N,K] row-major appears as B^T[K,N] column-major
  // - A[M,K] row-major appears as A^T[K,M] column-major
  // - C[M,N] row-major appears as C^T[N,M] column-major
  // So: C^T = B @ A^T means (B^T)^T @ (A^T)^T = B @ A^T (in column-major)
  auto lda = k; // Leading dimension of A (row-major stride)
  auto ldb = k; // Leading dimension of B (row-major stride)
  auto ldc = n; // Leading dimension of C (row-major stride)

  checkCublasStatus(cublasSgemm(
      handle,
      CUBLAS_OP_T, // Transpose B: B[N,K] -> B^T[K,N]
      CUBLAS_OP_N, // Don't transpose A: A[M,K] (treated as A^T[K,M])
      n, // number of rows of result (N)
      m, // number of columns of result (M)
      k, // common dimension (K)
      &alpha,
      B.data_ptr<float>(), // B[N,K] row-major
      ldb,
      A.data_ptr<float>(), // A[M,K] row-major
      lda,
      &beta,
      Y.data_ptr<float>(), // C[M,N] row-major
      ldc));

  return Y;
}

#else

at::Tensor bf16x9_gemm(
    at::Tensor A, // FP32
    at::Tensor B, // FP32
    std::optional<at::Tensor> output = std::nullopt) {
  throw std::runtime_error("Only supported on CUDA>=13");
}

#endif

} // namespace mslk::gemm
