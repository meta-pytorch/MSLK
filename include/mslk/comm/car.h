/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <cstdint>
#include <string>

namespace mslk::comm {

void nccl_init(
    int64_t rank,
    int64_t world_size,
    std::string rendevouz,
    int64_t comm_idx);

at::Tensor nccl_get_unique_id();

void nccl_comm_init_rank(
    int64_t world_size,
    int64_t rank,
    at::Tensor id_,
    int64_t comm_idx);

void nccl_allreduce(
    at::Tensor dst,
    at::Tensor src,
    std::optional<at::Tensor> bias,
    int64_t comm_idx);

void nccl_allgather(at::Tensor dst, at::Tensor src, int64_t comm_idx);

void nccl_alltoall_single(
    at::Tensor dst,
    at::Tensor src,
    int64_t world_size,
    int64_t comm_idx);

void nccl_alltoall(
    std::vector<at::Tensor> dsts,
    std::vector<at::Tensor> srcs,
    int64_t comm_idx);

void nccl_one2many(
    const std::vector<int64_t>& dst_ranks,
    std::vector<at::Tensor> srcs,
    int64_t comm_idx);

void nccl_many2one(
    std::vector<at::Tensor> dsts,
    const std::vector<int64_t>& src_ranks,
    int64_t comm_idx);

void nccl_broadcast(
    at::Tensor send,
    at::Tensor recv,
    int64_t root,
    int64_t comm_idx);

void nccl_reducescatter(at::Tensor dst, at::Tensor src, int64_t comm_idx);

void one_shot_car_allreduce(
    at::Tensor dst,
    at::Tensor src,
    std::optional<at::Tensor> bias,
    int64_t comm_idx,
    bool enable_pipelining);

void two_shot_car_allreduce(
    at::Tensor dst,
    at::Tensor src,
    std::optional<at::Tensor> bias,
    int64_t comm_idx,
    bool enable_pipelining);

void car_reducescatter(
    at::Tensor dst,
    at::Tensor src,
    bool split_last_dim,
    int64_t comm_idx);

void car_init(
    int64_t rank,
    int64_t world_size,
    at::Tensor local_barrier,
    std::vector<at::Tensor> all_barrier_handles,
    at::Tensor local_buffer,
    std::vector<at::Tensor> all_buffer_handles);

at::Tensor car_ipc_handle(at::Tensor x);

at::Tensor car_tensor();

} // namespace mslk::comm
