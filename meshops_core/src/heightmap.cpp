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

#include <meshops_internal/heightmap.hpp>
#include <cmath>
#include <filesystem>
#include <iostream>
#include "imageio/imageio.hpp"

bool HeightMap::load(const std::string& filename)
{
  data = nullptr;
  fileData.clear();
  size_t               loaded_width, loaded_height, components;
  imageio::ImageIOData loaded_data = imageio::loadF(filename.c_str(), &loaded_width, &loaded_height, &components, 1);
  if(!loaded_data)
  {
    return false;
  }
  assert(components == 1);  // Double-checking myself
  if(loaded_width > std::numeric_limits<int>::max() || loaded_height > std::numeric_limits<int>::max())
  {
    // Too large!
    return false;
  }
  width                                = static_cast<int>(loaded_width);
  height                               = static_cast<int>(loaded_height);
  const float* const loaded_data_float = reinterpret_cast<const float*>(loaded_data);
  fileData.insert(fileData.begin(), loaded_data_float, loaded_data_float + (width * height));
  imageio::freeData(&loaded_data);
  data = fileData.data();
  return true;
}

float HeightMap::texelFetch(int x, int y) const
{
  // Wrap around; perform modulo so e.g. -3 mod 8 = 5.
  x = (x < 0 ? width : 0) + (x % width);
  y = (y < 0 ? height : 0) + (y % height);
  return data[(y * width) + x];
}

float HeightMap::texelFetch(const nvmath::vec2f& texcoord) const
{
  const int x = int(texcoord.x * float(width));
  const int y = int(texcoord.y * float(height));
  return texelFetch(x, y);
}

float HeightMap::bilinearFetch(const nvmath::vec2f& texcoord) const
{
  const float tx   = texcoord.x - 0.5f / float(width);   // Offset so that texels are centered
  const float ty   = texcoord.y - 0.5f / float(height);  // at half-integer coordinates
  const float gx   = (tx - std::floor(tx)) * float(width);
  const float gy   = (ty - std::floor(ty)) * float(height);
  const int   gxi0 = int(gx) % width;
  const int   gxi1 = (gxi0 + 1) % width;
  const int   gyi0 = int(gy) % height;
  const int   gyi1 = (gyi0 + 1) % height;

  const float t00 = data[(gyi0 * width) + gxi0];
  const float t10 = data[(gyi0 * width) + gxi1];
  const float t01 = data[(gyi1 * width) + gxi0];
  const float t11 = data[(gyi1 * width) + gxi1];

  const float i0 = nvmath::lerp(gx - float(gxi0), t00, t10);
  const float i1 = nvmath::lerp(gx - float(gxi0), t01, t11);
  const float r  = nvmath::lerp(gy - float(gyi0), i0, i1);
  return r;
}
