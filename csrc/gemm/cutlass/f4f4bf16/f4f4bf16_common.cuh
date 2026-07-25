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
#include <cutlass/gemm/dispatch_policy.hpp>                   // @manual
#include <cutlass/epilogue/collective/collective_builder.hpp> // @manual
// clang-format on

namespace mslk::gemm {

namespace cutlass = cutlass_mslk;

using MXFP4 = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using NVFP4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
// MXFP4 with 1x16 block size
using MXFP4_16 = cutlass::mx_float4_16_t<cutlass::float_e2m1_t>;

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

// Scale-factor vector size for the SM103 ultra block-scaled schedules. NVFP4
// and MXFP4_16 quantize 1x16 blocks (Vs16); MXFP4 quantizes 1x32 blocks (Vs32).
// On SM103 the Auto schedule always resolves to Vs32, so the schedule must be
// named explicitly per format (unlike Sm100, which derives it from the element
// type).
template <typename T>
struct ultra_sf_vec_size;
template <typename F4>
struct ultra_sf_vec_size<cutlass::nv_float4_t<F4>> {
  static constexpr int value = 16;
};
template <typename F4>
struct ultra_sf_vec_size<cutlass::mx_float4_16_t<F4>> {
  static constexpr int value = 16;
};
template <typename F4>
struct ultra_sf_vec_size<cutlass::mx_float4_t<F4>> {
  static constexpr int value = 32;
};

// SM103 ultra schedules require CUDA 13+. Alias them to the Sm100 Auto schedule
// on older toolkits so UseUltra=true instances still compile (they are never
// dispatched pre-CUDA-13; see the runtime guard in f4f4bf16.cu).
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 13000)
using _UltraArchTag = cutlass::arch::Sm103;
using _Ultra1SmVs16 =
    cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103;
using _Ultra2SmVs16 =
    cutlass::gemm::KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103;
using _Ultra1SmVs32 =
    cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs32Sm103;
using _Ultra2SmVs32 =
    cutlass::gemm::KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs32Sm103;
#else
using _UltraArchTag = cutlass::arch::Sm100;
using _Ultra1SmVs16 = cutlass::gemm::collective::KernelScheduleAuto;
using _Ultra2SmVs16 = cutlass::gemm::collective::KernelScheduleAuto;
using _Ultra1SmVs32 = cutlass::gemm::collective::KernelScheduleAuto;
using _Ultra2SmVs32 = cutlass::gemm::collective::KernelScheduleAuto;
#endif

template <
    typename InputType,
    int TB_M,
    int TB_N,
    int TBS_M,
    int TBS_N,
    int TBS_K,
    bool UseUltra = false>
at::Tensor _f4f4bf16(
    at::Tensor XQ, // FP4
    at::Tensor WQ, // FP4
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    std::optional<at::Tensor> global_scale) {
  c10::cuda::CUDAGuard deviceGuard(XQ.device());

  const int M = XQ.size(0);
  const int N = WQ.size(0);
  const int K = XQ.size(1) * 2; // Since K is packed

  // The SM103 ultra block-scaled builder hard-requires TileShape K = 768
  // (static_assert in sm103_blockscaled_umma_builder.inl); it cannot be tuned.
  constexpr int TileShapeK =
      UseUltra ? 768 : (128 * 8 / cutlass::sizeof_bits<InputType>::value);

  using ElementA = InputType;
  using LayoutATag = cutlass::layout::RowMajor;
  using LayoutATag_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutATag>::type;
  constexpr int AlignmentA = 32;

  using ElementB = InputType;
  using LayoutBTag = cutlass::layout::ColumnMajor;
  using LayoutBTag_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutBTag>::type;
  constexpr int AlignmentB = 32;

  // TODO: Verify if bfloat16 is enough
  using ElementScale = float;
  using ElementCompute = float;
  using ElementAccumulator = float;

  using ElementOutput = cutlass::bfloat16_t;
  using LayoutOutputTag = cutlass::layout::RowMajor;
  using LayoutOutputTag_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutOutputTag>::type;
  constexpr int AlignmentOutput =
      128 /
      cutlass::sizeof_bits<
          ElementOutput>::value; // Memory access granularity/alignment of C
                                 // matrix in units of elements (up to 16 bytes)

  using ArchTag =
      cute::conditional_t<UseUltra, _UltraArchTag, cutlass::arch::Sm100>;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

  using ElementData = typename InputType::DataType;
  using ElementSF = typename InputType::ScaleFactorType;
  using ElementPair = cute::tuple<ElementData, ElementSF>;
  // The SM103 ultra CollectiveBuilder expects a (data, scale) tuple; the Sm100
  // builder takes the bare wrapper type and derives the scale type itself.
  using ElementForBuilder =
      cute::conditional_t<UseUltra, ElementPair, InputType>;

  // SM103 ultra: name the block-scaled schedule explicitly. Vs is a property of
  // the format (NVFP4 / MXFP4_16 -> Vs16, MXFP4 -> Vs32); 2SM when TB_M == 256
  // and cluster M is even, else 1SM. Unused (aliased to Auto) when !UseUltra.
  constexpr bool kUltra2Sm = (TB_M == 256) && (TBS_M % 2 == 0);
  constexpr int kUltraVs = ultra_sf_vec_size<InputType>::value;
  using _UltraSchedule = cute::conditional_t<
      (kUltraVs == 16),
      cute::conditional_t<kUltra2Sm, _Ultra2SmVs16, _Ultra1SmVs16>,
      cute::conditional_t<kUltra2Sm, _Ultra2SmVs32, _Ultra1SmVs32>>;
  using KernelSchedule = cute::conditional_t<
      UseUltra,
      _UltraSchedule,
      cutlass::gemm::collective::KernelScheduleAuto>;
  using MmaTileShape = cute::Shape<
      cute::Int<TB_M>,
      cute::Int<TB_N>,
      cute::Int<TileShapeK>>; // Threadblock-level MMA
                              // tile size
  using ClusterShape = cute::Shape<
      cute::Int<TBS_M>,
      cute::Int<TBS_N>,
      cute::Int<TBS_K>>; // Shape of the
                         // threadblocks in a
                         // cluster

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          MmaTileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementAccumulator,
          void, // No C input - we are doing C = A @ B, not C = A @ B + beta*C
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
          ElementForBuilder,
          LayoutBTag_Transpose,
          AlignmentB,
          ElementForBuilder,
          LayoutATag_Transpose,
          AlignmentA,
          ElementAccumulator,
          MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule // Auto on Sm100; explicit ultra schedule on Sm103
          >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, // Indicates ProblemShape
      CollectiveMainloop,
      CollectiveEpilogue,
      void>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using LayoutA =
      decltype(cute::make_layout(cute::make_shape(0, 0, 0), StrideA{}));
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::
      LayoutSFA; // Scale Factor tensors have an interleaved layout. Bring
                 // Layout instead of stride.
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using LayoutB =
      decltype(cute::make_layout(cute::make_shape(0, 0, 0), StrideB{}));
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::
      LayoutSFB; // Scale Factor tensors have an interleaved layout. Bring
                 // Layout instead of stride.
  using StrideOutput = typename Gemm::GemmKernel::StrideC;
  using LayoutOutput =
      decltype(cute::make_layout(cute::make_shape(0, 0, 0), StrideOutput{}));

  // For SFA and SFB tensors layouts
  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  // For SFD tensor layout
  using Sm100BlockScaledOutputConfig =
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

  if constexpr (std::is_same_v<
                    InputType,
                    cutlass::nv_float4_t<cutlass::float_e2m1_t>>) {
    TORCH_CHECK(global_scale.has_value(), "global_scale is required in nvfp4.");
    auto& fusion_args = arguments.epilogue.thread;
    fusion_args.alpha_ptr =
        static_cast<ElementCompute const*>(global_scale.value().data_ptr());
  }

  Gemm gemm;

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  at::Tensor workspace =
      at::empty(workspace_size, XQ.options().dtype(at::kByte));

  // Check the problem size is supported or not
  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement");
  }

  // Initialize CUTLASS kernel with arguments and workspace pointer
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
