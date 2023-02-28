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

#include <stdint.h>
#include <vulkan/vulkan_core.h>

// requires VK_KHR_push_descriptor

class PullPushFilter
{
public:
  // Maximum allowed number of mip levels. We'll always create this number
  // of views, even if the texture has no corresponding mips.
  // Must match pullpush.comp.
  // This is 15 so that the total descriptor count (2*MAX_MIP_LEVELS+2) is 32,
  // which is the maximum number of push descriptors supported by most
  // implementations of VK_KHR_push_descriptor.
  static const uint32_t MAX_MIP_LEVELS = 15;

  // Must match pullpush.comp.
  enum class Variant : uint32_t
  {
    eStandard    = 0,
    eNormals     = 1,
    eQuaternions = 2
  };

  class Pipes
  {
  private:
    VkPipeline pull = nullptr;
    VkPipeline push = nullptr;
    VkPipeline mips = nullptr;

    friend class PullPushFilter;
  };

  void init(VkDevice device);
  bool initialized() const { return m_device != nullptr; }
  void deinit();

  // pipes are specialized to a certain format
  // optionally generate mip-maps by averaging
  void initPipes(Pipes& pipes, Variant variant, VkShaderModule shaderModule, bool recomputeAveragedMips = true) const;
  void deinitPipes(Pipes& pipes) const;

  struct Views
  {
    // Views for the image to pull-push filter.
    VkImageView rgbaRead                            = nullptr;
    VkImageView rgbaReadWriteLevels[MAX_MIP_LEVELS] = {nullptr};
    // Views for an image that indicates the weight of each texel when
    // pull-push filtering. Mip 0 contains depth (all texels with a depth
    // less than a particular threshold get weight 1, and all others get weight
    // 0; the pull-push filter won't modify this level, so it can be reused),
    // while higher mips contain weights in 0...1.
    VkImageView depthWeightRead                            = nullptr;
    VkImageView depthWeightReadWriteLevels[MAX_MIP_LEVELS] = {nullptr};
  };

  // Describes an image. The image must have mipmaps.
  struct ImageInfo
  {
    uint32_t width;
    uint32_t height;

    uint32_t levelCount;
    VkImage  image;
    VkFormat imageFormat;
  };

  // Returns true on error.
  // Barrier before and after must be handled by user.
  // Assumes target Image was transitioned to VK_IMAGE_LAYOUT_GENERAL already.
  // imageRGBA and imageDepthWeight must have the same size and number of mips;
  // imageDepthWeight should be an R32F image. (See Views for more info.)
  bool process(VkCommandBuffer cmd, const Pipes& pipes, const ImageInfo& imageRGBA, const ImageInfo& imageDepthWeight, const Views& views) const;

  // utility function to create necessary views
  void initViews(Views& views, const ImageInfo& rgbaInfo, const ImageInfo& depthWeightInfo) const;
  void deinitViews(Views& views) const;

private:
  VkDevice  m_device      = nullptr;
  VkSampler m_readSampler = nullptr;  // Bilinear sampler for both the RGBA and depth-weight textures.

  VkPipelineLayout      m_pipeLayout  = nullptr;
  VkDescriptorSetLayout m_descrLayout = nullptr;
};
