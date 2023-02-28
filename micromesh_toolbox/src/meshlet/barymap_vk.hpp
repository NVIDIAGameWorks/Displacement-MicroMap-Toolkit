/*
* SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "utilities/rbuffer.hpp"
#include <baryutils/baryutils.h>

namespace microdisp {

struct BaryLevelsMapVK
{
  struct Level
  {
    size_t coordsOffset  = 0;
    size_t headersOffset = 0;
    size_t dataOffset    = 0;

    size_t firstHeader  = 0;
    size_t headersCount = 0;
    size_t firstData    = 0;
    size_t dataCount    = 0;
  };

  RBuffer binding;
  RBuffer data;

  std::vector<Level> levels;

  const Level& getLevel(uint32_t subdivLevel, uint32_t topoBits, uint32_t maxLevelCount) const
  {
    return levels[subdivLevel + topoBits * maxLevelCount];
  }

  void init(nvvk::ResourceAllocator& alloc, VkCommandBuffer cmd, const baryutils::BaryLevelsMap& bmap);
  void deinit(nvvk::ResourceAllocator& alloc);
};

}  // namespace microdisp
