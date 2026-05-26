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

#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12080)

namespace mslk::gemm {

namespace cutlass = cutlass_mslk;

inline int64_t _mx8mx4_grouped_byte_align(int64_t offset) {
  int64_t remainder = offset % 16;
  if (remainder != 0) {
    offset += (16 - remainder);
  }
  return offset;
}

template <
    typename ProblemShape,
    typename ElementA,
    typename ElementB,
    typename ElementC,
    typename ScaleDtype,
    typename StrideA,
    typename StrideB,
    typename StrideC,
    typename LayoutSFA,
    typename LayoutSFB,
    typename Sm1xxBlkScaledConfig>
__global__ void set_mx8mx4_grouped_2d3d_args_kernel(
    int64_t G,
    int64_t M,
    int64_t N,
    int64_t K,
    ProblemShape* problem_shape_ptr,
    ElementA* xq,
    const ElementA** xq_ptr,
    ElementB* wq,
    const ElementB** wq_ptr,
    ScaleDtype* x_scale,
    const ScaleDtype** x_scale_ptr,
    ScaleDtype* w_scale,
    const ScaleDtype** w_scale_ptr,
    ElementC* output,
    ElementC** output_ptr,
    StrideA* stride_a_ptr,
    StrideB* stride_b_ptr,
    StrideC* stride_c_ptr,
    int32_t* offsets,
    LayoutSFA* layout_SFA,
    LayoutSFB* layout_SFB) {
  const uint32_t group_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (group_index >= G) {
    return;
  }

  __shared__ int non_zero_counter;
  if (threadIdx.x == 0) {
    non_zero_counter = 0;
  }

  problem_shape_ptr[group_index] = ProblemShape(0, 0, 0);
  __syncthreads();

  const int32_t prev_group_end_offset =
      (group_index == 0) ? 0 : offsets[group_index - 1];
  const int32_t curr_group_end_offset = offsets[group_index];
  const int32_t M_group_size = curr_group_end_offset - prev_group_end_offset;

  if (M_group_size <= 0) {
    return;
  }

  CUDA_KERNEL_ASSERT(
      curr_group_end_offset <= M &&
      "for mx8mx4 2d-3d grouped GEMM, group end offsets must be <= M\n");

  auto round_up = [](int64_t x, int64_t y) { return ((x + y - 1) / y) * y; };

  constexpr int64_t kScaleFactorBlockSize = 32;
  const int64_t K_scale_rounded = round_up(K / kScaleFactorBlockSize, 4);
  const int64_t N_rounded = round_up(N, 128);

  int64_t scale_group_offset_M = 0;
  for (int i = 0; i < group_index; i++) {
    const int64_t group_i_size =
        (i == 0) ? offsets[i] : offsets[i] - offsets[i - 1];
    scale_group_offset_M += round_up(group_i_size, 128);
  }

  const int64_t group_offset_M = prev_group_end_offset;
  const int non_zero_idx = atomicAdd(&non_zero_counter, 1);

  // A is MXFP8: one byte per logical K element.
  const int64_t xq_offset = group_offset_M * K;
  // B is MXFP4: two logical K elements packed into one storage element.
  const int64_t wq_offset = group_index * N * K / 2;
  const int64_t output_offset = group_offset_M * N;
  const int64_t x_scale_offset = scale_group_offset_M * K_scale_rounded;
  const int64_t w_scale_offset = group_index * N_rounded * K_scale_rounded;

  // The mixed MX8xMX4 kernel uses the transposed problem form:
  //   D^T = W @ X^T, with logical output still stored as [M, N].
  problem_shape_ptr[non_zero_idx] = ProblemShape(N, M_group_size, K);

  xq_ptr[non_zero_idx] = xq + xq_offset;
  wq_ptr[non_zero_idx] = wq + wq_offset;
  x_scale_ptr[non_zero_idx] = x_scale + x_scale_offset;
  w_scale_ptr[non_zero_idx] = w_scale + w_scale_offset;
  output_ptr[non_zero_idx] = output + output_offset;

  stride_a_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(int(M_group_size), int(K), 1));
  stride_b_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(int(N), int(K), 1));
  stride_c_ptr[non_zero_idx] = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(int(N), int(M_group_size), 1));

  layout_SFA[non_zero_idx] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(int(M_group_size), int(N), int(K), 1));
  layout_SFB[non_zero_idx] = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(int(M_group_size), int(N), int(K), 1));
}

template <int TB_M, int TB_N, int TB_K, int TBS_M, int TBS_N, int TBS_K>
at::Tensor mx8mx4bf16_grouped_impl(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor output,
    int64_t G,
    at::Tensor offsets) {
  c10::cuda::CUDAGuard deviceGuard(XQ.device());

  using ProblemShape =
      cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
  using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
  using ElementC = cutlass::bfloat16_t;

  using LayoutA = typename cutlass::layout::LayoutTranspose<
      cutlass::layout::RowMajor>::type;
  using LayoutB = typename cutlass::layout::LayoutTranspose<
      cutlass::layout::ColumnMajor>::type;
  using LayoutC = typename cutlass::layout::LayoutTranspose<
      cutlass::layout::RowMajor>::type;

  constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  constexpr int AlignmentB = 128;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

  using ElementAccumulator = float;
  using ArchTag = cutlass::arch::Sm100;
  using TileShape =
      cute::Shape<cute::Int<TB_M>, cute::Int<TB_N>, cute::Int<TB_K>>;
  using ClusterShape =
      cute::Shape<cute::Int<TBS_M>, cute::Int<TBS_N>, cute::Int<TBS_K>>;

  using KernelSchedule = cute::conditional_t<
      (TB_M == 256) && (TBS_M % 2 == 0),
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100>;
  using EpilogueSchedule = cute::conditional_t<
      (TB_M == 256) && (TBS_M % 2 == 0),
      cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm,
      cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator,
          ElementAccumulator,
          void,
          void,
          0,
          ElementC,
          LayoutC*,
          AlignmentC,
          EpilogueSchedule>::CollectiveOp;

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          cutlass::arch::OpClassBlockScaledTensorOp,
          ElementB,
          LayoutB*,
          AlignmentB,
          ElementA,
          LayoutA*,
          AlignmentA,
          ElementAccumulator,
          TileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideD;

  using ScaleDtype = typename ElementA::ScaleFactorType;
  using DataTypeA = typename ElementA::DataType;
  using DataTypeB = typename ElementB::DataType;

  using LayoutSFA =
      typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB =
      typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using Sm1xxBlkScaledConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  const int64_t problem_size_offset = 0;
  int64_t problem_size_buffer = _mx8mx4_grouped_byte_align(
      G * sizeof(ProblemShape::UnderlyingProblemShape));

  const int64_t xq_offset = problem_size_offset + problem_size_buffer;
  int64_t xq_size_buffer = _mx8mx4_grouped_byte_align(G * sizeof(ElementA**));

  const int64_t wq_offset = xq_offset + xq_size_buffer;
  int64_t wq_size_buffer = _mx8mx4_grouped_byte_align(G * sizeof(ElementB**));

  const int64_t x_scale_offset = wq_offset + wq_size_buffer;
  int64_t x_scale_buffer = _mx8mx4_grouped_byte_align(G * sizeof(ScaleDtype**));

  const int64_t w_scale_offset = x_scale_offset + x_scale_buffer;
  int64_t w_scale_buffer = _mx8mx4_grouped_byte_align(G * sizeof(ScaleDtype**));

  const int64_t output_offset = w_scale_offset + w_scale_buffer;
  int64_t output_buffer = _mx8mx4_grouped_byte_align(G * sizeof(ElementC**));

  const int64_t stride_a_offset = output_offset + output_buffer;
  int64_t stride_a_buffer = _mx8mx4_grouped_byte_align(G * sizeof(StrideA));

  const int64_t stride_b_offset = stride_a_offset + stride_a_buffer;
  int64_t stride_b_buffer = _mx8mx4_grouped_byte_align(G * sizeof(StrideB));

  const int64_t stride_c_offset = stride_b_offset + stride_b_buffer;
  int64_t stride_c_buffer = _mx8mx4_grouped_byte_align(G * sizeof(StrideC));

  const int64_t layout_SFA_offset = stride_c_offset + stride_c_buffer;
  int64_t layout_SFA_buffer = _mx8mx4_grouped_byte_align(G * sizeof(LayoutSFA));

  const int64_t layout_SFB_offset = layout_SFA_offset + layout_SFA_buffer;
  int64_t layout_SFB_buffer = _mx8mx4_grouped_byte_align(G * sizeof(LayoutSFB));

  int64_t total_buffer_size = layout_SFB_offset + layout_SFB_buffer;
  at::Tensor kernel_args =
      at::empty({total_buffer_size}, XQ.options().dtype(at::kByte));
  char* kernel_args_ptr = reinterpret_cast<char*>(kernel_args.data_ptr());

  auto* problem_shape_ptr =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(
          kernel_args_ptr + problem_size_offset);
  const ElementA** xq_ptr =
      reinterpret_cast<const ElementA**>(kernel_args_ptr + xq_offset);
  const ElementB** wq_ptr =
      reinterpret_cast<const ElementB**>(kernel_args_ptr + wq_offset);
  const ScaleDtype** x_scale_ptr =
      reinterpret_cast<const ScaleDtype**>(kernel_args_ptr + x_scale_offset);
  const ScaleDtype** w_scale_ptr =
      reinterpret_cast<const ScaleDtype**>(kernel_args_ptr + w_scale_offset);
  ElementC** output_ptr =
      reinterpret_cast<ElementC**>(kernel_args_ptr + output_offset);
  StrideA* stride_a_ptr =
      reinterpret_cast<StrideA*>(kernel_args_ptr + stride_a_offset);
  StrideB* stride_b_ptr =
      reinterpret_cast<StrideB*>(kernel_args_ptr + stride_b_offset);
  StrideC* stride_c_ptr =
      reinterpret_cast<StrideC*>(kernel_args_ptr + stride_c_offset);
  LayoutSFA* layout_SFA =
      reinterpret_cast<LayoutSFA*>(kernel_args_ptr + layout_SFA_offset);
  LayoutSFB* layout_SFB =
      reinterpret_cast<LayoutSFB*>(kernel_args_ptr + layout_SFB_offset);

  const int64_t M = XQ.size(0);
  const int64_t N = WQ.size(-2);
  const int64_t K = WQ.size(-1) * 2;

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  set_mx8mx4_grouped_2d3d_args_kernel<
      ProblemShape::UnderlyingProblemShape,
      ElementA,
      ElementB,
      ElementC,
      ScaleDtype,
      StrideA,
      StrideB,
      StrideC,
      LayoutSFA,
      LayoutSFB,
      Sm1xxBlkScaledConfig><<<1, G, 0, stream>>>(
      G,
      M,
      N,
      K,
      problem_shape_ptr,
      reinterpret_cast<ElementA*>(XQ.data_ptr()),
      xq_ptr,
      reinterpret_cast<ElementB*>(WQ.data_ptr()),
      wq_ptr,
      reinterpret_cast<ScaleDtype*>(x_scale.data_ptr()),
      x_scale_ptr,
      reinterpret_cast<ScaleDtype*>(w_scale.data_ptr()),
      w_scale_ptr,
      reinterpret_cast<ElementC*>(output.data_ptr()),
      output_ptr,
      stride_a_ptr,
      stride_b_ptr,
      stride_c_ptr,
      reinterpret_cast<int32_t*>(offsets.data_ptr()),
      layout_SFA,
      layout_SFB);

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  hw_info.sm_count =
      min(cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
              hw_info.device_id),
          2147483647);

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {int(G), problem_shape_ptr, nullptr},
      {reinterpret_cast<const DataTypeB**>(wq_ptr),
       stride_b_ptr,
       reinterpret_cast<const DataTypeA**>(xq_ptr),
       stride_a_ptr,
       reinterpret_cast<const ScaleDtype**>(w_scale_ptr),
       layout_SFB,
       reinterpret_cast<const ScaleDtype**>(x_scale_ptr),
       layout_SFA},
      {{}, nullptr, stride_c_ptr, output_ptr, stride_c_ptr},
      hw_info};

  Gemm gemm;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace =
      at::empty(workspace_size, XQ.options().dtype(at::kByte));

  cutlass::Status status = gemm.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot implement mx8mx4 grouped");
  }

  status = gemm.initialize(
      arguments, reinterpret_cast<uint8_t*>(workspace.data_ptr()));
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error("cutlass cannot initialize mx8mx4 grouped");
  }

  status = gemm(at::cuda::getCurrentCUDAStream());
  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(
        std::string("cutlass cannot run mx8mx4 grouped: ") +
        cutlass::cutlassGetStatusString(status));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output;
}

} // namespace mslk::gemm

#endif

#undef CUTLASS_NAMESPACE
