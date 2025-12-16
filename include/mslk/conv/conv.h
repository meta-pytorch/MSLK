/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <vector>

#pragma once

namespace mslk::conv {

at::Tensor f8f8bf16_conv(
    at::Tensor activation,
    at::Tensor filter,
    at::Tensor scale,
    std::vector<int64_t> padding,
    std::vector<int64_t> stride,
    std::vector<int64_t> dilation);

} // namespace mslk::conv
