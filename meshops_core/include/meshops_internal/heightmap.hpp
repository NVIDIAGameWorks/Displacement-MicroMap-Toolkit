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
#include <string>
#include <vector>

#include "nvmath/nvmath.h"

//////////////////////////////////////////////////////////////////////////
/// Class to hold the height map texture and to fetch the value
struct HeightMap
{
  HeightMap() = default;
  HeightMap(int width, int height, const float* source)
      : width(width)
      , height(height)
      , data(source)
  {
  }

  // Load the texture in 16bit
  bool load(const std::string& filename);

  // Return the value [0..1] for texture coordinates [0..1]
  float              texelFetch(const nvmath::vec2f& texcoord) const;
  float              texelFetch(int x, int y) const;
  float              bilinearFetch(const nvmath::vec2f& texcoord) const;
  const float*       raw() const { return data ? data : fileData.data(); }
  int                width{0};
  int                height{0};
  int                components{0};
  const float*       data = nullptr;
  std::vector<float> fileData;
};
