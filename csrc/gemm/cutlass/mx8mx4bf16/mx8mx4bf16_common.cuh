/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define CUTLASS_NAMESPACE mslk

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/packed_stride.hpp>

// clang-format off
// The fixed ordering of the headers is required for CUTLASS 3.2+
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>     // @manual
#include <cutlass/gemm/device/gemm_universal_adapter.h>       // @manual
#include <cutlass/epilogue/collective/collective_builder.hpp> // @manual
// clang-format on

namespace mslk::gemm {

namespace cutlass = cutlass_mslk;

using MXFP8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using MXFP4 = cutlass::mx_float4_t<cutlass::float_e2m1_t>;

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

template <
    int TB_M,
    int TB_N,
    int TBS_M,
    int TBS_N,
    int TBS_K>
at::Tensor _mx8mx4bf16(
    at::Tensor XQ, // MX FP8 (e4m3)
    at::Tensor WQ, // MX FP4 (e2m1)
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output) {
  c10::cuda::CUDAGuard deviceGuard(XQ.device());

  const int M = XQ.size(0);
  const int N = WQ.size(0);
  const int K = XQ.size(1);

  using ElementA = MXFP8;
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutATag_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutATag>::type;
  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = MXFP4;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutBTag_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutBTag>::type;
  constexpr int AlignmentB = 128;

  using ElementScale = float;
  using ElementCompute = float;
  using ElementAccumulator = float;

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutputTag = cutlass::layout::RowMajor;
  using LayoutOutputTag_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutOutputTag>::type;
  constexpr int AlignmentOutput =
      128 / cutlass::sizeof_bits<ElementOutput>::value;

  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  constexpr int TileShapeK = 256;

  using MmaTileShape =
      cute::Shape<cute::Int<TB_M>, cute::Int<TB_N>, cute::Int<TileShapeK>>;
  using ClusterShape =
      cute::Shape<cute::Int<TBS_M>, cute::Int<TBS_N>, cute::Int<TBS_K>>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          MmaTileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementAccumulator,
          void,
          void,
          0,
          ElementOutput,
          LayoutOutputTag_Transpose,
          AlignmentOutput,
          cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementB,
          LayoutBTag_Transpose,
          AlignmentB,
          ElementA,
          LayoutATag_Transpose,
          AlignmentA,
          ElementAccumulator,
          MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>,
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutA =
      decltype(cute::make_layout(cute::make_shape(0, 0, 0), StrideA{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutB =
      decltype(cute::make_layout(cute::make_shape(0, 0, 0), StrideB{}));
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
  using StrideOutput = typename Gemm::GemmKernel::StrideC;
  using LayoutOutput =
      decltype(cute::make_layout(cute::make_shape(0, 0, 0), StrideOutput{}));

  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  StrideA stride_A =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  StrideB stride_B =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  StrideOutput stride_output = cutlass::make_cute_packed_stride(
      StrideOutput{}, cute::make_shape(N, M, 1));

  LayoutA layout_A = make_layout(cute::make_shape(M, K, 1), stride_A);
  LayoutB layout_B = make_layout(cute::make_shape(N, K, 1), stride_B);
  LayoutOutput layout_output =
      make_layout(cute::make_shape(N, M, 1), stride_output);
  LayoutSFA layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(M, N, K, 1));
  LayoutSFB layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(M, N, K, 1));

  using DataTypeA = typename ElementA::DataType;
  using DataTypeB = typename ElementB::DataType;
  using SFTypeA = typename ElementA::ScaleFactorType;
  using SFTypeB = typename ElementB::ScaleFactorType;

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {N, M, K, 1},
      {// Mainloop arguments
       reinterpret_cast<DataTypeB*>(WQ.data_ptr()),
       stride_B,
       reinterpret_cast<DataTypeA*>(XQ.data_ptr()),
       stride_A,
       reinterpret_cast<SFTypeB*>(w_scale.data_ptr()),
       layout_SFB,
       reinterpret_cast<SFTypeA*>(x_scale.data_ptr()),
       layout_SFA},
      {// Epilogue arguments
       {1, 0},
       reinterpret_cast<ElementOutput*>(output.data_ptr()),
       stride_output,
       reinterpret_cast<ElementOutput*>(output.data_ptr()),
       stride_output}};

  Gemm gemm;

  size_t workspace_size = Gemm::get_workspace_size(arguments);

  at::Tensor workspace =
      at::empty(workspace_size, XQ.options().dtype(at::kByte));

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  status = gemm.initialize(arguments, workspace.data_ptr());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize");
  }

  status = gemm(at::cuda::getCurrentCUDAStream());

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("cutlass cannot run") +
        cutlass::cutlassGetStatusString(status));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

#endif

} // namespace mslk::gemm

#undef CUTLASS_NAMESPACE
