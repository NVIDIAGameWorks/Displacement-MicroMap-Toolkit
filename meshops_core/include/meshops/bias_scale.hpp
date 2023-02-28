/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <algorithm>

// Class to use a bias and scale as a self-contained transformation object
template <class T>
struct BiasScale
{
  BiasScale()                       = default;
  BiasScale(const BiasScale& other) = default;
  BiasScale(const T& bias, const T& scale)
      : bias(bias)
      , scale(scale)
  {
  }
  template <class Vec>
  BiasScale(const Vec& vec)
      : bias(vec.x)
      , scale(vec.y)
  {
  }

  // Transform a value by the bias and scale
  template <class V>
  V operator*(const V& value) const
  {
    return value * scale + bias;
  }

  // Combines two BiasScales into one that performs the same transform that they would when chained
  BiasScale& operator*=(const BiasScale& other)
  {
    bias += other.bias * scale;
    scale *= other.scale;
    return *this;
  }

  BiasScale operator*(const BiasScale& other) const
  {
    BiasScale result(*this);
    result *= other;
    return result;
  }

  // Returns a BiasScale that scales about a point.
  static BiasScale centered_scale(const T& center, const T& scale)
  {
    return BiasScale((static_cast<T>(1) - scale) * center, scale);
  }

  // Returns a BiasScale that transforms values in the range min/max into the range 0 to 1.
  static BiasScale minmax_unit(const T& min, const T& max) { return BiasScale(min, max - min); }

  template <class Vec>
  static BiasScale minmax_unit(const Vec& vec)
  {
    return BiasScale(vec.x, vec.y - vec.x);
  }

  T unit_min() const { return bias; }          // BiasScale * 0 = 0 * scale + bias
  T unit_max() const { return scale + bias; }  // BiasScale * 1 = 1 * scale + bias

  BiasScale inverse() const { return BiasScale(-bias / scale, static_cast<T>(1) / scale); }

  // Avoid singularity transforms by limiting how small scale can get
  BiasScale degenerate_clamped(T epsilon = static_cast<T>(1e-6)) const
  {
    return BiasScale(bias, std::max(epsilon, scale));
  }

  T bias{0};
  T scale{1};
};

using BiasScalef = BiasScale<float>;