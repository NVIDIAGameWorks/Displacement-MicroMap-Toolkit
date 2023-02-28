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

#include "pullpush_filter.hpp"
#include <algorithm>
#include <array>
#include <cassert>

// These #defines must match their values in pullpush.comp.

#define PULLPUSH_MAX_MIP_LEVELS 15
static_assert(PullPushFilter::MAX_MIP_LEVELS == PULLPUSH_MAX_MIP_LEVELS, "PullPushFilter::MAX_MIP_LEVELS mismatch");

// Fills mips 1...end, blending colors and weights based on higher mip weights.
#define PULLPUSH_MODE_PULL 0
// After pulling, fills mips, interpolating and blending based on weights, to
// fill empty spaces in mip 0 (but also overwtiting mips 1...end-1).
#define PULLPUSH_MODE_PUSH 1
// Constructs a mip chain using fast box filtering. If you just need
// mipmapping, take a look at the nvpro_pyramid library here:
// https://github.com/nvpro-samples/vk_compute_mipmaps
#define PULLPUSH_MODE_AVG 2

// Specialization constants
#define PULLPUSH_SPC_VARIANT 0
#define PULLPUSH_SPC_MODE 1
#define PULLPUSH_SPC_COUNT 2

// Bindings
#define PULLPUSH_BINDING_RGBA_TEXTURE 0
#define PULLPUSH_BINDING_RGBA_LEVELS 1
#define PULLPUSH_BINDING_DEPTHWEIGHT_TEXTURE 2
#define PULLPUSH_BINDING_DEPTHWEIGHT_LEVELS 3
#define PULLPUSH_BINDING_COUNT 4

struct PullPushConstants
{
  // When pulling and averaging, a nonzero value for levelActive[i] means
  // that we can write level `srcLevel + i`. This int[4] matches a bvec4.
  int levelActive[4];
  // width and height of the level we're reading from (`srcLevel` when
  // pulling and averaging, and `srcLevel + 1` when pushing).
  int srcSizeW;
  int srcSizeH;
  // Designates the mip level we're reading from when pulling and averaging,
  // and the level we're writing when pushing.
  int srcLevel;
};
static_assert(sizeof(PullPushConstants) == 28, "PullPushConstants size mismatch");

void checkVkSuccess(VkResult result)
{
  assert(result == VK_SUCCESS);
}

void PullPushFilter::init(VkDevice device)
{
  m_device = device;

  {
    VkSamplerCreateInfo info     = {VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.borderColor             = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    info.anisotropyEnable        = VK_FALSE;
    info.maxAnisotropy           = 0;
    info.flags                   = 0;
    info.compareEnable           = VK_FALSE;
    info.compareOp               = VK_COMPARE_OP_ALWAYS;
    info.unnormalizedCoordinates = VK_FALSE;
    info.mipLodBias              = 0;
    info.minLod                  = 0;
    info.maxLod                  = float(PULLPUSH_MAX_MIP_LEVELS);
    info.minFilter               = VK_FILTER_NEAREST;
    info.magFilter               = VK_FILTER_NEAREST;
    info.mipmapMode              = VK_SAMPLER_MIPMAP_MODE_NEAREST;

    checkVkSuccess(vkCreateSampler(device, &info, nullptr, &m_readSampler));
  }

  {
    std::array<VkDescriptorSetLayoutBinding, PULLPUSH_BINDING_COUNT> bindings{};
    bindings[PULLPUSH_BINDING_RGBA_TEXTURE].binding            = PULLPUSH_BINDING_RGBA_TEXTURE;
    bindings[PULLPUSH_BINDING_RGBA_TEXTURE].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[PULLPUSH_BINDING_RGBA_TEXTURE].descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[PULLPUSH_BINDING_RGBA_TEXTURE].descriptorCount    = 1;
    bindings[PULLPUSH_BINDING_RGBA_TEXTURE].pImmutableSamplers = &m_readSampler;

    bindings[PULLPUSH_BINDING_RGBA_LEVELS].binding            = PULLPUSH_BINDING_RGBA_LEVELS;
    bindings[PULLPUSH_BINDING_RGBA_LEVELS].stageFlags         = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[PULLPUSH_BINDING_RGBA_LEVELS].descriptorType     = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[PULLPUSH_BINDING_RGBA_LEVELS].descriptorCount    = MAX_MIP_LEVELS;
    bindings[PULLPUSH_BINDING_RGBA_LEVELS].pImmutableSamplers = nullptr;

    // The depth-weight bindings are the same as the RGBA bindings:
    bindings[PULLPUSH_BINDING_DEPTHWEIGHT_TEXTURE]         = bindings[PULLPUSH_BINDING_RGBA_TEXTURE];
    bindings[PULLPUSH_BINDING_DEPTHWEIGHT_TEXTURE].binding = PULLPUSH_BINDING_DEPTHWEIGHT_TEXTURE;

    bindings[PULLPUSH_BINDING_DEPTHWEIGHT_LEVELS]         = bindings[PULLPUSH_BINDING_RGBA_LEVELS];
    bindings[PULLPUSH_BINDING_DEPTHWEIGHT_LEVELS].binding = PULLPUSH_BINDING_DEPTHWEIGHT_LEVELS;

    VkDescriptorSetLayoutCreateInfo info = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount                    = uint32_t(bindings.size());
    info.pBindings                       = bindings.data();
    info.flags                           = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;

    checkVkSuccess(vkCreateDescriptorSetLayout(device, &info, nullptr, &m_descrLayout));
  }

  {
    VkPushConstantRange range;
    range.offset     = 0;
    range.size       = sizeof(PullPushConstants);
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipelineLayoutCreateInfo info = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    info.pSetLayouts                = &m_descrLayout;
    info.setLayoutCount             = 1;
    info.pPushConstantRanges        = &range;
    info.pushConstantRangeCount     = 1;

    checkVkSuccess(vkCreatePipelineLayout(device, &info, nullptr, &m_pipeLayout));
  }
}

void PullPushFilter::initPipes(Pipes& pipes, Variant variant, VkShaderModule shaderModule, bool recomputeAveragedMips) const
{
  deinitPipes(pipes);

  {
    std::array<VkSpecializationMapEntry, PULLPUSH_SPC_COUNT> spcEntries{};
    std::array<uint32_t, PULLPUSH_SPC_COUNT>                 spcData{};
    VkSpecializationInfo                                     spcInfo{};
    spcInfo.mapEntryCount = uint32_t(spcEntries.size());
    spcInfo.pData         = spcData.data();
    spcInfo.pMapEntries   = spcEntries.data();
    spcInfo.dataSize      = spcData.size() * sizeof(spcData[0]);

    spcEntries[PULLPUSH_SPC_VARIANT].constantID = PULLPUSH_SPC_VARIANT;
    spcEntries[PULLPUSH_SPC_VARIANT].offset     = sizeof(uint32_t) * PULLPUSH_SPC_VARIANT;
    spcEntries[PULLPUSH_SPC_VARIANT].size       = sizeof(uint32_t);
    spcEntries[PULLPUSH_SPC_MODE].constantID    = PULLPUSH_SPC_MODE;
    spcEntries[PULLPUSH_SPC_MODE].offset        = sizeof(uint32_t) * PULLPUSH_SPC_MODE;
    spcEntries[PULLPUSH_SPC_MODE].size          = sizeof(uint32_t);

    spcData[PULLPUSH_SPC_VARIANT] = static_cast<uint32_t>(variant);


    VkComputePipelineCreateInfo info = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage.sType                 = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage                 = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.pName                 = "main";
    info.stage.module                = shaderModule;
    info.stage.pSpecializationInfo   = &spcInfo;
    info.layout                      = m_pipeLayout;

    spcData[PULLPUSH_SPC_MODE] = PULLPUSH_MODE_PULL;
    checkVkSuccess(vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &pipes.pull));

    spcData[PULLPUSH_SPC_MODE] = PULLPUSH_MODE_PUSH;
    checkVkSuccess(vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &pipes.push));

    if(recomputeAveragedMips)
    {
      spcData[PULLPUSH_SPC_MODE] = PULLPUSH_MODE_AVG;
      checkVkSuccess(vkCreateComputePipelines(m_device, nullptr, 1, &info, nullptr, &pipes.mips));
    }
    else
    {
      pipes.mips = nullptr;
    }
  }
}

void PullPushFilter::deinitPipes(Pipes& pipes) const
{
  if(!m_device)
    return;

  vkDestroyPipeline(m_device, pipes.push, nullptr);
  vkDestroyPipeline(m_device, pipes.pull, nullptr);
  vkDestroyPipeline(m_device, pipes.mips, nullptr);

  pipes = Pipes();
}

void PullPushFilter::deinit()
{
  if(!m_device)
    return;

  vkDestroyPipelineLayout(m_device, m_pipeLayout, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descrLayout, nullptr);
  vkDestroySampler(m_device, m_readSampler, nullptr);

  *this = PullPushFilter();
}

bool PullPushFilter::process(VkCommandBuffer  cmd,
                             const Pipes&     pipes,
                             const ImageInfo& imageRGBA,
                             const ImageInfo& imageDepthWeight,
                             const Views&     views) const
{
  if(imageRGBA.levelCount < 1 || imageRGBA.levelCount > MAX_MIP_LEVELS)
  {
    assert(0 && "imageRGBA.levelCount out of bounds!");
    return true;
  }
  if(imageRGBA.width != imageDepthWeight.width || imageRGBA.height != imageDepthWeight.height)
  {
    assert(0 && "imageRGBA and imageDepthWeight had different widths or heights!");
    return true;
  }
  if(imageRGBA.levelCount != imageDepthWeight.levelCount)
  {
    assert(0 && "imageRGBA and imageDepthWeight had different numbers of mips!");
    return true;
  }
  const uint32_t levelCount = imageRGBA.levelCount;
  assert(pipes.push);
  assert(pipes.pull);

  std::array<uint32_t, MAX_MIP_LEVELS> mipWidths;
  std::array<uint32_t, MAX_MIP_LEVELS> mipHeights;

  // Write descriptor sets.
  VkDescriptorImageInfo                             descriptorRGBATexture{};
  std::array<VkDescriptorImageInfo, MAX_MIP_LEVELS> descriptorRGBALevels{};
  VkDescriptorImageInfo                             descriptorDepthWeightTexture{};
  std::array<VkDescriptorImageInfo, MAX_MIP_LEVELS> descriptorDepthWeightLevels{};

  descriptorRGBATexture.sampler     = nullptr;
  descriptorRGBATexture.imageView   = views.rgbaRead;
  descriptorRGBATexture.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

  descriptorDepthWeightTexture           = descriptorRGBATexture;
  descriptorDepthWeightTexture.imageView = views.depthWeightRead;

  for(uint32_t i = 0; i < MAX_MIP_LEVELS; i++)
  {
    mipWidths[i]  = std::max(1U, imageRGBA.width >> i);
    mipHeights[i] = std::max(1U, imageRGBA.height >> i);

    descriptorRGBALevels[i].sampler     = nullptr;
    descriptorRGBALevels[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    descriptorDepthWeightLevels[i]      = descriptorRGBALevels[i];

    descriptorRGBALevels[i].imageView        = views.rgbaReadWriteLevels[std::min(i, levelCount - 1)];
    descriptorDepthWeightLevels[i].imageView = views.depthWeightReadWriteLevels[std::min(i, levelCount - 1)];
  }

  std::array<VkWriteDescriptorSet, PULLPUSH_BINDING_COUNT> writeSets{};
  writeSets[PULLPUSH_BINDING_RGBA_TEXTURE].descriptorType    = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  writeSets[PULLPUSH_BINDING_RGBA_TEXTURE].descriptorCount   = 1;
  writeSets[PULLPUSH_BINDING_RGBA_TEXTURE].pImageInfo        = &descriptorRGBATexture;
  writeSets[PULLPUSH_BINDING_DEPTHWEIGHT_TEXTURE]            = writeSets[PULLPUSH_BINDING_RGBA_TEXTURE];
  writeSets[PULLPUSH_BINDING_DEPTHWEIGHT_TEXTURE].pImageInfo = &descriptorDepthWeightTexture;

  writeSets[PULLPUSH_BINDING_RGBA_LEVELS].descriptorType    = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  writeSets[PULLPUSH_BINDING_RGBA_LEVELS].descriptorCount   = uint32_t(descriptorRGBALevels.size());
  writeSets[PULLPUSH_BINDING_RGBA_LEVELS].pImageInfo        = descriptorRGBALevels.data();
  writeSets[PULLPUSH_BINDING_DEPTHWEIGHT_LEVELS]            = writeSets[PULLPUSH_BINDING_RGBA_LEVELS];
  writeSets[PULLPUSH_BINDING_DEPTHWEIGHT_LEVELS].pImageInfo = descriptorDepthWeightLevels.data();

  for(uint32_t i = 0; i < PULLPUSH_BINDING_COUNT; i++)
  {
    writeSets[i].sType      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeSets[i].dstBinding = i;
  }

  vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeLayout, 0, uint32_t(writeSets.size()), writeSets.data());

  // We use these image barriers between compute passes; they ensure that the
  // RGBA and depth-weight textures are ready for the next pass.
  std::array<VkImageMemoryBarrier, 2> imageBarriers{};
  imageBarriers[0].sType                       = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  imageBarriers[0].image                       = imageRGBA.image;
  imageBarriers[0].dstAccessMask               = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  imageBarriers[0].srcAccessMask               = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
  imageBarriers[0].newLayout                   = VK_IMAGE_LAYOUT_GENERAL;
  imageBarriers[0].oldLayout                   = VK_IMAGE_LAYOUT_GENERAL;
  imageBarriers[0].subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  imageBarriers[0].subresourceRange.layerCount = 1;
  imageBarriers[0].subresourceRange.levelCount = levelCount;
  imageBarriers[1]                             = imageBarriers[0];
  imageBarriers[1].image                       = imageDepthWeight.image;

  PullPushConstants constants;
  const uint32_t    tileSize = 8;  // Width and height of texels each workgroup processes; see pullpush.comp.

  if(pipes.pull)
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes.pull);

    // This shader processes three levels at a time:
    const uint32_t passLevels = 3;
    for(uint32_t i = 0; i < levelCount; i += passLevels)
    {
      if(i != 0)
      {
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, uint32_t(imageBarriers.size()), imageBarriers.data());
      }

      for(uint32_t level = 0; level < passLevels + 1; level++)
      {
        const int32_t active         = level + i < levelCount;
        constants.levelActive[level] = active;
      }

      const uint32_t inputW = mipWidths[i];
      const uint32_t inputH = mipHeights[i];

      constants.srcSizeW = inputW;
      constants.srcSizeH = inputH;

      // Each thread reads four values from mip i and produces a texel in mip i+1.
      const uint32_t subW = mipWidths[i + 1];
      const uint32_t subH = mipHeights[i + 1];

      constants.srcLevel = i;

      vkCmdPushConstants(cmd, m_pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
      vkCmdDispatch(cmd, (subW + tileSize - 1) / tileSize, (subH + tileSize - 1) / tileSize, 1);
    }
  }

  if(pipes.push)
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes.push);

    // 1 level at a time: push from mip levelCount - 1 to levelCount - 2 to ... to 0.
    for(uint32_t i = 1; i < levelCount; i++)
    {
      // Level we're writing to (not reading from)
      const uint32_t level = levelCount - 1 - i;

      const uint32_t inputW = mipWidths[level];
      const uint32_t inputH = mipHeights[level];

      constants.srcSizeW = inputW;
      constants.srcSizeH = inputH;

      constants.srcLevel = level;

      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                           nullptr, 0, nullptr, uint32_t(imageBarriers.size()), imageBarriers.data());

      vkCmdPushConstants(cmd, m_pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
      vkCmdDispatch(cmd, (inputW + tileSize - 1) / tileSize, (inputH + tileSize - 1) / tileSize, 1);
    }
  }

  if(pipes.mips)
  {
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, uint32_t(imageBarriers.size()), imageBarriers.data());

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipes.mips);

    // Same indexing as when pulling
    const uint32_t passLevels = 3;

    for(uint32_t i = 1; i < levelCount; i += passLevels)
    {
      if(i != 0)
      {
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, uint32_t(imageBarriers.size()), imageBarriers.data());
      }

      for(uint32_t level = 0; level < passLevels + 1; level++)
      {
        int32_t active               = level + i < levelCount;
        constants.levelActive[level] = active;
      }

      const uint32_t inputW = mipWidths[i];
      const uint32_t inputH = mipHeights[i];

      constants.srcSizeW = inputW;
      constants.srcSizeH = inputH;

      const uint32_t subW = mipWidths[i + 1];
      const uint32_t subH = mipHeights[i + 1];

      constants.srcLevel = i - 1;

      vkCmdPushConstants(cmd, m_pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
      vkCmdDispatch(cmd, (subW + tileSize - 1) / tileSize, (subH + tileSize - 1) / tileSize, 1);
    }
  }

  return false;
}

void PullPushFilter::initViews(Views& views, const ImageInfo& rgbaInfo, const ImageInfo& depthWeightInfo) const
{
  deinitViews(views);

  // Settings for all views
  VkImageViewCreateInfo viewInfo           = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  viewInfo.components.r                    = VK_COMPONENT_SWIZZLE_R;
  viewInfo.components.g                    = VK_COMPONENT_SWIZZLE_G;
  viewInfo.components.b                    = VK_COMPONENT_SWIZZLE_B;
  viewInfo.components.a                    = VK_COMPONENT_SWIZZLE_A;
  viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount     = 1;

  // rgbaRead
  viewInfo.image                         = rgbaInfo.image;
  viewInfo.format                        = rgbaInfo.imageFormat;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount   = rgbaInfo.levelCount;
  checkVkSuccess(vkCreateImageView(m_device, &viewInfo, nullptr, &views.rgbaRead));

  // rgbaReadWriteLevels
  for(uint32_t i = 0; i < rgbaInfo.levelCount; i++)
  {
    viewInfo.subresourceRange.baseMipLevel = i;
    viewInfo.subresourceRange.levelCount   = 1;
    checkVkSuccess(vkCreateImageView(m_device, &viewInfo, nullptr, &views.rgbaReadWriteLevels[i]));
  }

  // depthWeightRead
  viewInfo.image                         = depthWeightInfo.image;
  viewInfo.format                        = depthWeightInfo.imageFormat;
  viewInfo.subresourceRange.baseMipLevel = 0;
  viewInfo.subresourceRange.levelCount   = depthWeightInfo.levelCount;
  checkVkSuccess(vkCreateImageView(m_device, &viewInfo, nullptr, &views.depthWeightRead));

  // depthWeightReadWriteLevels
  for(uint32_t i = 0; i < depthWeightInfo.levelCount; i++)
  {
    viewInfo.subresourceRange.baseMipLevel = i;
    viewInfo.subresourceRange.levelCount   = 1;
    checkVkSuccess(vkCreateImageView(m_device, &viewInfo, nullptr, &views.depthWeightReadWriteLevels[i]));
  }
}

void PullPushFilter::deinitViews(Views& views) const
{
  if(!m_device)
    return;

  vkDestroyImageView(m_device, views.rgbaRead, nullptr);
  vkDestroyImageView(m_device, views.depthWeightRead, nullptr);
  for(uint32_t i = 0; i < MAX_MIP_LEVELS; i++)
  {
    vkDestroyImageView(m_device, views.rgbaReadWriteLevels[i], nullptr);
    vkDestroyImageView(m_device, views.depthWeightReadWriteLevels[i], nullptr);
  }

  views = Views();
}
