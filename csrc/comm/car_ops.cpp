/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <mslk/comm/car.h> // @manual
#include <mslk/utils/torch/op_registration.h> // @manual
#include <torch/library.h>

namespace mslk::comm {

TORCH_LIBRARY_FRAGMENT(mslk, m) {
  m.def(
      "nccl_init(int rank, int world_size, str rendevouz, int comm_idx=0) -> ()");
  m.impl("nccl_init", nccl_init);

  m.def("nccl_get_unique_id() -> Tensor");
  m.impl("nccl_get_unique_id", nccl_get_unique_id);

  m.def(
      "nccl_comm_init_rank(int world_size, int rank, Tensor id_, int comm_idx=0) -> ()");
  m.impl("nccl_comm_init_rank", nccl_comm_init_rank);

  m.def("nccl_allgather(Tensor(a!) dst, Tensor src, int comm_idx=0) -> ()");

  m.def(
      "nccl_alltoall_single(Tensor(a!) dst, Tensor src, int world_size, int comm_idx=0) -> ()");
  m.def("nccl_alltoall(Tensor(a!)[] dst, Tensor[] src, int comm_idx=0) -> ()");

  m.def("nccl_one2many(int[] dst_ranks, Tensor[] src, int comm_idx=0) -> ()");
  m.def(
      "nccl_many2one(Tensor(a!)[] dst, int[] src_ranks, int comm_idx=0) -> ()");
  m.def(
      "nccl_broadcast(Tensor send, Tensor(a!) recv, int root, int comm_idx=0) -> ()");
  m.def("nccl_reducescatter(Tensor(a!) dst, Tensor src, int comm_idx=0) -> ()");

  m.def(
      "nccl_allreduce(Tensor(a!) dst, Tensor src, Tensor? bias=None, int comm_idx=0) -> ()");
  // car: customized all reduce
  m.def("car_tensor() -> Tensor");
  m.impl("car_tensor", car_tensor);

  m.def("car_ipc_handle(Tensor buffer) -> Tensor");
  m.impl("car_ipc_handle", car_ipc_handle);

  m.def(
      "car_init(int rank, int world_size, Tensor local_barrier, Tensor[] all_barrier_handles, Tensor local_buffer, Tensor[] all_buffer_handles) -> ()");
  m.impl("car_init", car_init);

  m.def(
      "one_shot_car_allreduce(Tensor(a!) dst, Tensor src, Tensor? bias=None, int comm_idx=0, bool enable_pipelining=False) -> ()");

  m.def(
      "two_shot_car_allreduce(Tensor(a!) dst, Tensor src, Tensor? bias=None, int comm_idx=0, bool enable_pipelining=False) -> ()");

  m.def(
      "car_reducescatter(Tensor(a!) dst, Tensor src, bool split_last_dim=False, int comm_idx=0) -> ()");
}

TORCH_LIBRARY_IMPL(mslk, CUDA, m) {
  DISPATCH_TO_CUDA("nccl_allreduce", nccl_allreduce);
  DISPATCH_TO_CUDA("nccl_allgather", nccl_allgather);
  DISPATCH_TO_CUDA("nccl_alltoall_single", nccl_alltoall_single);
  DISPATCH_TO_CUDA("nccl_alltoall", nccl_alltoall);
  DISPATCH_TO_CUDA("nccl_one2many", nccl_one2many);
  DISPATCH_TO_CUDA("nccl_many2one", nccl_many2one);
  DISPATCH_TO_CUDA("nccl_broadcast", nccl_broadcast);
  DISPATCH_TO_CUDA("nccl_reducescatter", nccl_reducescatter);
  DISPATCH_TO_CUDA("one_shot_car_allreduce", one_shot_car_allreduce);
  DISPATCH_TO_CUDA("two_shot_car_allreduce", two_shot_car_allreduce);
  DISPATCH_TO_CUDA("car_reducescatter", car_reducescatter);
}

// Shape registration functions for car operators.
void nccl_allreduce_meta(
    at::Tensor /* dst */,
    at::Tensor /* src */,
    std::optional<at::Tensor> /* bias */,
    int64_t /* comm_idx */) {
  return;
}

void nccl_allgather_meta(
    at::Tensor /* dst */,
    at::Tensor /* src */,
    int64_t /* comm_idx */) {
  return;
}

void nccl_alltoall_single_meta(
    at::Tensor /* dst */,
    at::Tensor /* src */,
    int64_t /* world_size */,
    int64_t /* comm_idx */) {
  return;
}

void nccl_alltoall_meta(
    std::vector<at::Tensor> /* dsts */,
    std::vector<at::Tensor> /* srcs */,
    int64_t /* comm_idx */) {
  return;
}

void nccl_one2many_meta(
    const std::vector<int64_t>& /* dst_ranks */,
    std::vector<at::Tensor> /* srcs */,
    int64_t /* comm_idx */) {
  return;
}

void nccl_many2one_meta(
    std::vector<at::Tensor> /* dsts */,
    const std::vector<int64_t>& /* src_ranks */,
    int64_t /* comm_idx */) {
  return;
}

void nccl_broadcast_meta(
    at::Tensor /*send*/,
    at::Tensor /*recv*/,
    int64_t /*root*/,
    int64_t /*comm_idx*/) {
  return;
}

void nccl_reducescatter_meta(
    at::Tensor /* dst */,
    at::Tensor /* src */,
    int64_t /* comm_idx */) {
  return;
}

void one_shot_car_allreduce_meta(
    at::Tensor /* dst */,
    at::Tensor /* src */,
    std::optional<at::Tensor> /* bias */,
    int64_t /* comm_idx */,
    bool /* enable_pipelining */) {
  return;
}

void two_shot_car_allreduce_meta(
    at::Tensor /* dst */,
    at::Tensor /* src */,
    std::optional<at::Tensor> /* bias */,
    int64_t /* comm_idx */,
    bool /* enable_pipelining */) {
  return;
}

void car_reducescatter_meta(
    at::Tensor /* dst */,
    at::Tensor /* src */,
    bool /* split_last_dim */,
    int64_t /* comm_idx */) {
  return;
}

TORCH_LIBRARY_IMPL(mslk, Meta, m) {
  DISPATCH_TO_META("nccl_allreduce", nccl_allreduce_meta);
  DISPATCH_TO_META("nccl_allgather", nccl_allgather_meta);
  DISPATCH_TO_META("nccl_alltoall_single", nccl_alltoall_single_meta);
  DISPATCH_TO_META("nccl_alltoall", nccl_alltoall_meta);
  DISPATCH_TO_META("nccl_one2many", nccl_one2many_meta);
  DISPATCH_TO_META("nccl_many2one", nccl_many2one_meta);
  DISPATCH_TO_META("nccl_broadcast_meta", nccl_broadcast_meta);
  DISPATCH_TO_META("nccl_reducescatter", nccl_reducescatter_meta);
  DISPATCH_TO_META("one_shot_car_allreduce", one_shot_car_allreduce_meta);
  DISPATCH_TO_META("two_shot_car_allreduce", two_shot_car_allreduce_meta);
  DISPATCH_TO_META("car_reducescatter", car_reducescatter_meta);
}

} // namespace mslk::comm
