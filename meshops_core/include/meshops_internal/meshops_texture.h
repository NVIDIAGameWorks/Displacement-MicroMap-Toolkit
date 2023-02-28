//
// SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//

#pragma once

#include <meshops_internal/meshops_context.h>

namespace meshops {

inline bool textureHasReadWriteAccess(TextureUsageFlags flags)
{
  return (flags & (eTextureUsageBakerResamplingDestination | eTextureUsageBakerResamplingDistance)) != 0;
}

inline bool textureNeedsHost(TextureUsageFlags flags)
{
  return (flags & (eTextureUsageBakerHeightmapSource)) != 0;
}

inline bool textureNeedsDevice(TextureUsageFlags flags)
{
  return (flags
          & (eTextureUsageBakerResamplingDestination | eTextureUsageBakerResamplingDistance
             | eTextureUsageBakerResamplingSource | eTextureUsageRemesherImportanceSource))
         != 0;
}

class Texture_c
{
public:
  TextureConfig     m_config;
  TextureUsageFlags m_usageFlags;

  TextureVK   m_vk;
  nvvk::Image m_vkData;

  micromesh::MicromapValue                m_fillValue;
  std::vector<micromesh::Vector_uint32_2> m_mipSizes;
  std::vector<std::vector<uint8_t>>       m_mipData;  // only if host data exists

  // returns mip 0 data only if host copy exists
  const void* getImageData() const { return m_mipData.size() ? m_mipData[0].data() : nullptr; }

  void initMipSizes()
  {
    m_mipSizes.resize(m_config.mips);
    if(m_config.mips)
    {
      m_mipSizes[0] = {m_config.width, m_config.height};
    }

    for(uint32_t m = 1; m < m_config.mips; m++)
    {
      m_mipSizes[m] = {std::max(1u, m_mipSizes[m - 1].x / 2), std::max(1u, m_mipSizes[m - 1].y / 2)};
    }
  }

  inline bool hasReadWriteAccess() const { return textureHasReadWriteAccess(m_usageFlags); }
  inline bool needsDevice() const { return textureNeedsDevice(m_usageFlags); }
  inline bool needsHost() const { return textureNeedsHost(m_usageFlags); }
};

}  // namespace meshops
