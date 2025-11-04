/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/core/dispatch/Dispatcher.h>

namespace mslk::utils::torch {

inline bool schemaExists(const std::string& qualified_name) {
  return c10::Dispatcher::singleton()
      .findSchema({qualified_name, ""})
      .has_value();
}

} // namespace mslk::utils::torch
