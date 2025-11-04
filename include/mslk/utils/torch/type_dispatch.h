/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/library.h>

////////////////////////////////////////////////////////////////////////////////
/// Dispatch Helper Macros
///
/// These macros cover bundled dispatch cases, similar to AT_DISPATCH_*_CASE
////////////////////////////////////////////////////////////////////////////////

#define MSLK_DISPATCH_INTEGRAL_TYPES_CASE(...)       \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define MSLK_DISPATCH_FLOATING_TYPES_CASE(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#if defined(USE_ROCM)

#define MSLK_DISPATCH_FLOAT_HALF_AND_FP8_CASE(...)     \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Float8_e4m3fnuz, __VA_ARGS__)

#else

#define MSLK_DISPATCH_FLOAT_HALF_AND_FP8_CASE(...)     \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Float8_e4m3fn, __VA_ARGS__)

#endif

#define MSLK_DISPATCH_FLOAT_AND_HALF_CASE(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)

#define MSLK_DISPATCH_FLOAT_AND_BFLOAT16_CASE(...)     \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define MSLK_DISPATCH_ALL_TYPES_BUT_HALF_CASE(...)        \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)    \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
  MSLK_DISPATCH_INTEGRAL_TYPES_CASE(__VA_ARGS__)

#define MSLK_DISPATCH_FLOAT_AND_DOUBLE_CASE(...)       \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__)

////////////////////////////////////////////////////////////////////////////////
/// Type Dispatch Macros
///
/// These macros are similar to AT_DISPATCH_*, but do not support
/// at::ScalarType::Double
////////////////////////////////////////////////////////////////////////////////

#define MSLK_DISPATCH_FLOAT_ONLY(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                             \
      TYPE, NAME, AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__))

#define MSLK_DISPATCH_FLOAT_AND_DOUBLE(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                   \
      TYPE, NAME, MSLK_DISPATCH_FLOAT_AND_DOUBLE_CASE(__VA_ARGS__))

#define MSLK_DISPATCH_FLOAT_AND_HALF(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, MSLK_DISPATCH_FLOAT_AND_HALF_CASE(__VA_ARGS__))

#define MSLK_DISPATCH_FLOAT_HALF_AND_BYTE(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                      \
      TYPE,                                                \
      NAME,                                                \
      MSLK_DISPATCH_FLOAT_AND_HALF_CASE(__VA_ARGS__)       \
          AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__))

#define MSLK_DISPATCH_FLOAT_HALF_FP8_AND_BYTE(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                          \
      TYPE,                                                    \
      NAME,                                                    \
      MSLK_DISPATCH_FLOAT_HALF_AND_FP8_CASE(__VA_ARGS__)       \
          AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__))

#define MSLK_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, MSLK_DISPATCH_FLOATING_TYPES_CASE(__VA_ARGS__))

#define MSLK_DISPATCH_FLOATING_TYPES_AND(SCALARTYPE, TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                                 \
      TYPE,                                                           \
      NAME,                                                           \
      MSLK_DISPATCH_FLOATING_TYPES_CASE(__VA_ARGS__)                  \
          AT_DISPATCH_CASE(SCALARTYPE, __VA_ARGS__))

#define MSLK_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, MSLK_DISPATCH_INTEGRAL_TYPES_CASE(__VA_ARGS__))

#define MSLK_DISPATCH_ALL_TYPES(TYPE, NAME, ...)     \
  AT_DISPATCH_SWITCH(                                \
      TYPE,                                          \
      NAME,                                          \
      MSLK_DISPATCH_FLOATING_TYPES_CASE(__VA_ARGS__) \
          MSLK_DISPATCH_INTEGRAL_TYPES_CASE(__VA_ARGS__))

#define MSLK_DISPATCH_ALL_TYPES_AND_DOUBLE(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(                                       \
      TYPE,                                                 \
      NAME,                                                 \
      MSLK_DISPATCH_FLOATING_TYPES_CASE(__VA_ARGS__)        \
          MSLK_DISPATCH_INTEGRAL_TYPES_CASE(__VA_ARGS__)    \
              AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__))
