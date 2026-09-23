/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <c10/hip/HIPStream.h>

#include <limits>

#ifdef HIPIFY_V2
#define getCurrentHIPStream getCurrentCUDAStream
#endif

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/utility/data_type.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_xdl_cshuffle_v3.hpp"
#include "mslk/utils/device/launch_checks.h"

// Define commonly used types.
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using MultiplyMultiply = ck::tensor_operation::element_wise::MultiplyMultiply;

using ADataType = ck::f8_t;
using BDataType = ck::f8_t;
using D0DataType = float;
using D1DataType = float;
using DsDataType = ck::Tuple<D0DataType, D1DataType>;
using EDataType = ck::bhalf_t;
using AccDataType = float;
using CShuffleDataType = float;

using ALayout = Row;
using BLayout = Col;
using D0Layout = Row;
using D1Layout = Col;
using DsLayout = ck::Tuple<D0Layout, D1Layout>;
using ELayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CDEElementOp = MultiplyMultiply;

using ComputeType = ck::f8_t;

template <
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int WAVE_TILE_M,
    int WAVE_TILE_N,
    int WAVE_MAP_M,
    int WAVE_MAP_N,
    typename ABLOCK_TRANSFER,
    typename BBLOCK_TRANSFER,
    typename CBLOCK_TRANSFER,
    typename CBLOCK_SPV,
    int CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
    int CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
    ck::BlockGemmPipelineScheduler LOOP_SCHED,
    ck::BlockGemmPipelineVersion PIPELINE_VERSION,
    ck::tensor_operation::device::GemmSpecialization GEMM_SPEC =
        ck::tensor_operation::device::GemmSpecialization::MNPadding>
using DeviceGemmHelper =
    ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl_CShuffle_V3<
        ALayout,
        BLayout,
        DsLayout,
        ELayout,
        ADataType,
        BDataType,
        DsDataType,
        EDataType,
        AccDataType,
        CShuffleDataType,
        AElementOp,
        BElementOp,
        CDEElementOp,
        GEMM_SPEC,
        BLOCK_SIZE, // Block Size
        MBLOCK, // M per Block
        NBLOCK, // N per Block
        KBLOCK, // K per Block
        16, // AK1
        16, // BK1
        WAVE_TILE_M, // M per Xdl
        WAVE_TILE_N, // N per Xdl
        WAVE_MAP_M, // Mxdl per Wave
        WAVE_MAP_N, // Nxdl per Wave
        ABLOCK_TRANSFER,
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        16,
        16,
        0,
        BBLOCK_TRANSFER,
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        16,
        16,
        0,
        CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
        CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
        CBLOCK_TRANSFER,
        CBLOCK_SPV,
        LOOP_SCHED,
        PIPELINE_VERSION,
        ComputeType>;

template <typename DeviceGemmInstance>
at::Tensor f8f8bf16_rowwise_batched_impl(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {
  // Get input information.
  const int64_t B64 = XQ.size(0);
  const int64_t M64 = XQ.size(1);
  const int64_t N64 = WQ.size(1);
  const int64_t K64 = WQ.size(2);
  constexpr auto ck_index_max = std::numeric_limits<ck::index_t>::max();
  TORCH_CHECK(
      B64 <= ck_index_max && M64 <= ck_index_max && N64 <= ck_index_max &&
          K64 <= ck_index_max,
      "MSLK-080 CK batched GEMM dimensions must fit ck::index_t; got B=",
      B64,
      ", M=",
      M64,
      ", N=",
      N64,
      ", K=",
      K64);
  const auto B = static_cast<ck::index_t>(B64);
  const auto M = static_cast<ck::index_t>(M64);
  const auto N = static_cast<ck::index_t>(N64);
  const auto K = static_cast<ck::index_t>(K64);

  int StrideA = K;
  int StrideB = K;
  int StrideD0 = 0;
  int StrideD1 = 0;
  int StrideE = N;

  const int64_t BatchStrideA64 = M64 * K64;
  const int64_t BatchStrideB64 = N64 * K64;
  const int64_t BatchStrideE64 = M64 * N64;
  TORCH_CHECK(
      BatchStrideA64 <= ck_index_max && BatchStrideB64 <= ck_index_max &&
          BatchStrideE64 <= ck_index_max,
      "MSLK-080 CK batched GEMM strides must fit ck::index_t; got A=",
      BatchStrideA64,
      ", B=",
      BatchStrideB64,
      ", E=",
      BatchStrideE64);
  const auto BatchStrideA = static_cast<ck::index_t>(BatchStrideA64);
  const auto BatchStrideB = static_cast<ck::index_t>(BatchStrideB64);
  const auto BatchStrideD0 = N;
  const auto BatchStrideD1 = M;
  const auto BatchStrideE = static_cast<ck::index_t>(BatchStrideE64);

  // Create gemm launcher and arguments.
  auto gemm = DeviceGemmInstance{};
  auto invoker = gemm.MakeInvoker();

  auto a_element_op = AElementOp{};
  auto b_element_op = BElementOp{};
  auto cde_element_op = CDEElementOp{};

  auto argument = gemm.MakeArgument(
      reinterpret_cast<ADataType*>(XQ.data_ptr()),
      reinterpret_cast<BDataType*>(WQ.data_ptr()),
      std::array<const void*, 2>{
          reinterpret_cast<D0DataType*>(w_scale.data_ptr()),
          reinterpret_cast<D1DataType*>(x_scale.data_ptr())},
      reinterpret_cast<EDataType*>(Y.data_ptr()),
      M,
      N,
      K,
      B,
      StrideA,
      StrideB,
      std::array<ck::index_t, 2>{StrideD0, StrideD1},
      StrideE,
      BatchStrideA,
      BatchStrideB,
      std::array<ck::index_t, 2>{BatchStrideD0, BatchStrideD1},
      BatchStrideE,
      a_element_op,
      b_element_op,
      cde_element_op);

  auto stream = at::cuda::getCurrentHIPStream().stream();
  invoker.Run(argument, StreamConfig{stream, false});

  return Y;
}
