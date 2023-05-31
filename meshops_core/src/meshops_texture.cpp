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

#include "meshops/meshops_operations.h"
#include "micromesh/micromesh_types.h"
#include <meshops_internal/meshops_texture.h>
#include <micromesh/micromesh_utils.h>
#include <cstddef>
#include <nvvk/images_vk.hpp>

namespace meshops {

namespace {

void cmdTextureLayoutBarrier(VkCommandBuffer cmd, Texture tex, VkImageLayout src, VkImageLayout dst)
{
  VkImageSubresourceRange subresourceRange{};
  subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  subresourceRange.baseArrayLayer = 0;
  subresourceRange.baseMipLevel   = 0;
  subresourceRange.layerCount     = 1;
  subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;

  nvvk::cmdBarrierImageLayout(cmd, tex->m_vkData.image, src, dst, subresourceRange);
}

// Logs and returns an error if texture creation failed.
static micromesh::Result validateTextureUsage(Context context, TextureUsageFlags usageFlags, const TextureConfig& config)
{
  MESHOPS_CHECK_CTX_NONNULL(context);
  MESHOPS_CHECK_NONNULL(context, usageFlags);

  micromesh::Result result = micromesh::Result::eSuccess;
  if(config.baseFormat == micromesh::Format::eUndefined)
  {
    MESHOPS_LOGE(context, "`config.baseFormat` must not be micromesh::Format::eUndefined.");
    result = micromesh::Result::eInvalidFormat;
  }
  if(usageFlags & eTextureUsageBakerResamplingDistance)
  {
    if(config.baseFormat != micromesh::Format::eR32_sfloat)
    {
      MESHOPS_LOGE(context, "TextureConfig::baseFormat must be eR32_sfloat for eTextureUsageBakerResamplingDistance.");
      result = micromesh::Result::eInvalidFormat;
    }
    if(config.internalFormatVk != VK_FORMAT_R32_SFLOAT)
    {
      MESHOPS_LOGE(context,
                   "TextureConfig::internalFormatVk must be VK_FORMAT_R32_SFLOAT for "
                   "eTextureUsageBakerResamplingDistance.");
      result = micromesh::Result::eInvalidFormat;
    }
  }
  return result;
}

micromesh::Result createTexture(Context context, TextureUsageFlags usageFlags, const TextureConfig& config, Texture* pTexture)
{
  {
    micromesh::Result validUsageResult = validateTextureUsage(context, usageFlags, config);
    if(validUsageResult != micromesh::Result::eSuccess)
    {
      return validUsageResult;
    }
  }

  Texture tex = new Texture_c{config, usageFlags};
  tex->initMipSizes();

  if(tex->needsDevice())
  {
    tex->m_vk.imageCreateInfo           = nvvk::makeImage2DCreateInfo({config.width, config.height},
                                                                      VkFormat(config.internalFormatVk), VK_IMAGE_USAGE_SAMPLED_BIT);
    tex->m_vk.imageCreateInfo.mipLevels = config.mips;
    if(textureHasReadWriteAccess(usageFlags))
    {
      tex->m_vk.imageCreateInfo.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
    }

    tex->m_vk.imageLayout = textureHasReadWriteAccess(usageFlags) ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    tex->m_vkData   = context->m_vk->m_resourceAllocator.createImage(tex->m_vk.imageCreateInfo);
    tex->m_vk.image = tex->m_vkData.image;
    if(!tex->m_vk.image)
    {
      MESHOPS_LOGE(context, "The call to ResourceAllocator::createImage failed.");
      meshopsTextureDestroy(context, tex);
      return micromesh::Result::eFailure;
    }

    VkImageViewCreateInfo viewInfo = nvvk::makeImageViewCreateInfo(tex->m_vkData.image, tex->m_vk.imageCreateInfo, false);
    VkResult vkResult = vkCreateImageView(context->m_vkDevice, &viewInfo, nullptr, &tex->m_vk.imageView);
    if(vkResult != VK_SUCCESS)
    {
      MESHOPS_LOGE(context, "`vkCreateImageView()` failed, returning VkResult %d.", int32_t(vkResult));
      meshopsTextureDestroy(context, tex);
      return micromesh::Result::eFailure;
    }
  }
  if(tex->needsHost())
  {
    tex->m_mipData.resize(tex->m_mipSizes.size());
  }

  *pTexture = tex;

  return micromesh::Result::eSuccess;
}

}  // namespace

// requires device context
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsTextureCreate(Context                         context,
                                                                TextureUsageFlags               usageFlags,
                                                                const TextureConfig&            config,
                                                                const micromesh::MicromapValue* clearColor,
                                                                Texture*                        pTexture)
{
  MESHOPS_CHECK_CTX_NONNULL(context);
  MESHOPS_CHECK_NONNULL(context, pTexture);

  Texture           tex;
  micromesh::Result result = createTexture(context, usageFlags, config, &tex);
  if(result != micromesh::Result::eSuccess)
  {
    // createTexture() has already logged an error; no need to duplicate it.
    return result;
  }

  micromesh::MicromapValue distanceTextureClearColor{};
  distanceTextureClearColor.value_float[0] = std::numeric_limits<float>::max();
  if(usageFlags & eTextureUsageBakerResamplingDistance)
  {
    // Error out if the clear value is zero. Overriding the default is OK but
    // zero makes no sense and would be an understandably common mistake.
    micromesh::MicromapValue zeroes{};
    if(clearColor && memcmp(clearColor, &zeroes, sizeof(*clearColor)) == 0)
    {
      MESHOPS_LOGE(context,
                   "Textures with eTextureUsageBakerResamplingDistance must not be cleared to zero. Best to use float "
                   "max.");
      meshopsTextureDestroy(context, tex);
      return micromesh::Result::eInvalidValue;
    }

    // Provide a default for distance textures
    if(!clearColor)
    {
      clearColor = &distanceTextureClearColor;
    }
  }

  *pTexture = tex;

  if(tex->needsDevice())
  {
    VkCommandBuffer cmd = context->m_vk->m_cmdPoolGCT.createCommandBuffer();
    if(clearColor)
    {
      VkClearColorValue cv;
      cv.uint32[0] = clearColor->value_uint32[0];
      cv.uint32[1] = clearColor->value_uint32[1];
      cv.uint32[2] = clearColor->value_uint32[2];
      cv.uint32[3] = clearColor->value_uint32[3];

      VkImageSubresourceRange subresourceRange{};
      subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      subresourceRange.baseArrayLayer = 0;
      subresourceRange.baseMipLevel   = 0;
      subresourceRange.layerCount     = 1;
      subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
      cmdTextureLayoutBarrier(cmd, tex, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      vkCmdClearColorImage(cmd, tex->m_vk.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cv, 1, &subresourceRange);
      cmdTextureLayoutBarrier(cmd, tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, tex->m_vk.imageLayout);
    }
    else
    {
      cmdTextureLayoutBarrier(cmd, tex, VK_IMAGE_LAYOUT_UNDEFINED, tex->m_vk.imageLayout);
    }

    context->m_vk->m_cmdPoolGCT.submitAndWait(cmd);
  }

  return result;
}

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsTextureCreateVK(Context              context,
                                                                  TextureUsageFlags    usageFlags,
                                                                  const TextureConfig& config,
                                                                  const TextureVK&     source,
                                                                  Texture*             pTexture)
{
  MESHOPS_CHECK_CTX_NONNULL(context);
  MESHOPS_CHECK_NONNULL(context, pTexture);
  MESHOPS_CHECK_NONNULL(context, usageFlags);

  if(config.internalFormatVk == VK_FORMAT_UNDEFINED)
  {
    MESHOPS_LOGE(context, "`config.internalFormatVk` must not be VK_FORMAT_UNDEFINED.");
    return micromesh::Result::eInvalidFormat;
  }

  {
    micromesh::Result validUsageResult = validateTextureUsage(context, usageFlags, config);
    if(validUsageResult != micromesh::Result::eSuccess)
    {
      return validUsageResult;
    }
  }

  Texture tex = new Texture_c{config, usageFlags, source};
  tex->initMipSizes();
  *pTexture = tex;

  return micromesh::Result::eSuccess;
}

MESHOPS_API void MESHOPS_CALL meshopsTextureDestroy(Context context, Texture texture)
{
  if(!context || !texture)
  {
    // Nothing to do
    return;
  }

  if(texture->m_vkData.memHandle)
  {
    vkDestroyImageView(context->m_vkDevice, texture->m_vk.imageView, nullptr);
    context->m_vk->m_resourceAllocator.destroy(texture->m_vkData);
  }

  delete texture;
}

// get vk details, can be nullptr if context was created without vk support
MESHOPS_API TextureVK* MESHOPS_CALL meshopsTextureGetVK(Texture texture)
{
  if(!texture)
    return nullptr;
  assert(texture);
  return &texture->m_vk;
}

MESHOPS_API TextureConfig MESHOPS_CALL meshopsTextureGetConfig(Texture texture)
{
  assert(texture);
  return texture->m_config;
}

MESHOPS_API size_t MESHOPS_CALL meshopsTextureGetMipDataSize(Texture texture, uint32_t mipLevel)
{
  assert(texture);

  micromesh::FormatInfo info;
  if(micromesh::micromeshFormatGetInfo(texture->m_config.baseFormat, &info) != micromesh::Result::eSuccess)
  {
    assert(!"micromesh::micromeshFormatGetInfo failed.");
    return 0;
  }

  if(mipLevel > texture->m_mipSizes.size())
  {
    assert(!"mipLevel was too large.");
    return 0;
  }

  return size_t(texture->m_mipSizes[mipLevel].x) * size_t(texture->m_mipSizes[mipLevel].y) * size_t(info.byteSize);
}

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsTextureCreateFromLoader(Context                  context,
                                                                          const TextureDataLoader* loader,
                                                                          size_t                   count,
                                                                          micromesh::Result*       results,
                                                                          Texture*                 textures,
                                                                          const TextureUsageFlags* usageFlags,
                                                                          const void**             textureInputs)
{
  MESHOPS_CHECK_CTX_NONNULL(context);
  MESHOPS_CHECK_NONNULL(context, loader);
  MESHOPS_CHECK_NONNULL(context, results);
  MESHOPS_CHECK_NONNULL(context, textures);
  MESHOPS_CHECK_NONNULL(context, usageFlags);
  MESHOPS_CHECK_NONNULL(context, textureInputs);

  bool needsCmd = false;

  for(size_t i = 0; i < count; i++)
  {
    results[i]  = micromesh::Result::eFailure;
    textures[i] = nullptr;

    if(textureNeedsDevice(usageFlags[i]))
    {
      needsCmd = true;
    }
  }

  VkCommandBuffer             cmd     = needsCmd ? context->m_vk->m_cmdPoolGCT.createCommandBuffer() : VK_NULL_HANDLE;
  nvvk::StagingMemoryManager* staging = context->m_vk->m_resourceAllocator.getStaging();

  micromesh::Result result = micromesh::Result::eFailure;

  for(size_t i = 0; i < count; i++)
  {
    TextureConfig config;

    void* handle = nullptr;
    result       = loader->fnOpen(textureInputs[i], config, &handle, loader->fnUserData);
    if(result != micromesh::Result::eSuccess)
    {
      MESHOPS_LOGE(context, "Call to loader->fnOpen for texture %zu failed, returning code %u (%s).", i,
                   unsigned(result), micromesh::micromeshResultGetName(result));
      results[i] = result;
      break;
    }

    Texture tex;
    result = createTexture(context, usageFlags[i], config, &tex);
    if(result != micromesh::Result::eSuccess)
    {
      // createTexture has already sent an error; no need to duplicate it.
      results[i] = result;
      break;
    }

    if(tex->needsDevice())
    {
      cmdTextureLayoutBarrier(cmd, tex, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    }

    for(uint32_t m = 0; m < config.mips; m++)
    {
      size_t outputSize = 0;
      result            = loader->fnReadGetSize(handle, m, outputSize, loader->fnUserData);
      if(result != micromesh::Result::eSuccess)
      {
        MESHOPS_LOGE(context, "Could not successfully upload texture data.");
        break;
      }

      void* destination   = nullptr;
      void* deviceStaging = nullptr;
      if(tex->needsDevice())
      {
        VkOffset3D               offset = {0, 0, 0};
        VkExtent3D               extent = {tex->m_mipSizes[m].x, tex->m_mipSizes[m].y, 1};
        VkImageSubresourceLayers subResource;
        subResource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        subResource.baseArrayLayer = 0;
        subResource.mipLevel       = m;
        subResource.layerCount     = 1;

        deviceStaging = staging->cmdToImage(cmd, tex->m_vk.image, offset, extent, subResource, outputSize, nullptr);
        destination   = deviceStaging;
      }

      // if host required, prefer that as destination
      void* hostCopy = nullptr;
      if(tex->needsHost())
      {
        tex->m_mipData[m].resize(outputSize);
        hostCopy    = tex->m_mipData[m].data();
        destination = hostCopy;
      }

      result = loader->fnReadData(handle, m, outputSize, destination, loader->fnUserData);
      if(result != micromesh::Result::eSuccess)
      {
        MESHOPS_LOGE(context,
                     "Call to loader->fnReadData to read %zu bytes from mip %u of image %zu failed, returning code %u "
                     "(%s).",
                     outputSize, m, i, unsigned(result), micromesh::micromeshResultGetName(result));
        break;
      }

      // both exist, in that case also copy from host to staging
      if(hostCopy && deviceStaging)
      {
        memcpy(deviceStaging, hostCopy, outputSize);
      }
    }

    loader->fnClose(handle, loader->fnUserData);

    if(result != micromesh::Result::eSuccess)
    {
      // problem during upload
      // we stop here
      // Message has already been printed above.
      results[i] = result;

      // need to complete in-flight cmd buffer as it
      // may contain copy instruction from previous textures and
      // current texture
      staging->finalizeResources();
      context->m_vk->m_cmdPoolGCT.submitAndWait(cmd);
      staging->releaseResources();

      meshopsTextureDestroy(context, tex);

      return result;
    }

    if(tex->needsDevice())
    {
      cmdTextureLayoutBarrier(cmd, tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, tex->m_vk.imageLayout);
    }

    loader->fnClose(handle, loader->fnUserData);
    results[i]  = result;
    textures[i] = tex;
  }

  if(cmd)
  {
    staging->finalizeResources();
    context->m_vk->m_cmdPoolGCT.submitAndWait(cmd);
    staging->releaseResources();
  }

  return micromesh::Result::eSuccess;
}

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsTextureCreateFromData(Context              context,
                                                                        TextureUsageFlags    usageFlags,
                                                                        const TextureConfig& config,
                                                                        size_t               dataSize,
                                                                        const void*          data,
                                                                        Texture*             pTexture)
{
  MESHOPS_CHECK_CTX_NONNULL(context);
  MESHOPS_CHECK_NONNULL(context, dataSize);
  MESHOPS_CHECK_NONNULL(context, data);
  MESHOPS_CHECK_NONNULL(context, pTexture);

  if(config.mips != 1)
  {
    MESHOPS_LOGE(context, "`config.mips` (%u) must be 1.", config.mips);
    return micromesh::Result::eInvalidValue;
  }

  if(config.internalFormatVk == VK_FORMAT_UNDEFINED)
  {
    MESHOPS_LOGE(context, "`config.internalFormatVk` must not be VK_FORMAT_UNDEFINED.");
    return micromesh::Result::eInvalidFormat;
  }

  const bool needsCmd = textureNeedsDevice(usageFlags);

  VkCommandBuffer             cmd     = VK_NULL_HANDLE;
  nvvk::StagingMemoryManager* staging = nullptr;
  if(needsCmd)
  {
    cmd = context->m_vk->m_cmdPoolGCT.createCommandBuffer();
    if(cmd == VK_NULL_HANDLE)
    {
      MESHOPS_LOGE(context, "Failed to create a Vulkan command buffer.");
      return micromesh::Result::eFailure;
    }
    staging = context->m_vk->m_resourceAllocator.getStaging();
    assert(staging != nullptr);
  }

  {
    Texture           tex;
    micromesh::Result result = createTexture(context, usageFlags, config, &tex);
    if(result != micromesh::Result::eSuccess)
    {
      // createTexture() has already logged an error; no need to duplicate it.
      return result;
    }
    assert(tex->needsDevice() == needsCmd);

    if(tex->needsDevice())
    {
      cmdTextureLayoutBarrier(cmd, tex, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    }

    uint32_t m = 0;
    {
      void* destination   = nullptr;
      void* deviceStaging = nullptr;

      if(tex->needsDevice())
      {
        VkOffset3D               offset = {0, 0, 0};
        VkExtent3D               extent = {tex->m_mipSizes[m].x, tex->m_mipSizes[m].y, 1};
        VkImageSubresourceLayers subResource;
        subResource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        subResource.baseArrayLayer = 0;
        subResource.mipLevel       = m;
        subResource.layerCount     = 1;

        deviceStaging = staging->cmdToImage(cmd, tex->m_vk.image, offset, extent, subResource, dataSize, nullptr);
        destination   = deviceStaging;
      }

      // if host required, prefer that as destination
      void* hostCopy = nullptr;
      if(tex->needsHost())
      {
        tex->m_mipData[m].resize(dataSize);
        hostCopy    = tex->m_mipData[m].data();
        destination = hostCopy;
      }

      if(!destination)
      {
        MESHOPS_LOGE(context, "Attempted to create a meshops::Texture on neither the host nor the device.");
        (void)meshopsTextureDestroy(context, tex);
        return micromesh::Result::eInvalidValue;
      }

      memcpy(destination, data, dataSize);

      // both exist, in that case also to staging
      if(hostCopy && deviceStaging)
      {
        memcpy(deviceStaging, data, dataSize);
      }
    }

    if(tex->needsDevice())
    {
      cmdTextureLayoutBarrier(cmd, tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, tex->m_vk.imageLayout);
    }

    *pTexture = tex;
  }

  if(cmd)
  {
    staging->finalizeResources();
    context->m_vk->m_cmdPoolGCT.submitAndWait(cmd);
    staging->releaseResources();
  }

  return micromesh::Result::eSuccess;
}

// returns first non-success result
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsTextureToSaver(Context                 context,
                                                                 const TextureDataSaver* saver,
                                                                 size_t                  count,
                                                                 micromesh::Result*      results,
                                                                 const Texture*          textures,
                                                                 const void**            textureInputs)
{
  MESHOPS_CHECK_CTX_NONNULL(context);
  MESHOPS_CHECK_NONNULL(context, saver);
  MESHOPS_CHECK_NONNULL(context, results);
  MESHOPS_CHECK_NONNULL(context, textures);
  MESHOPS_CHECK_NONNULL(context, textureInputs);

  for(size_t i = 0; i < count; i++)
  {
    results[i] = micromesh::Result::eFailure;
  }

  micromesh::Result result = micromesh::Result::eFailure;

  for(size_t i = 0; i < count; i++)
  {
    Texture tex = textures[i];

    void*    handle   = nullptr;
    uint32_t mipCount = 0;
    result            = saver->fnOpen(textures[i], textureInputs[i], mipCount, &handle, saver->fnUserData);
    if(result != micromesh::Result::eSuccess)
    {
      MESHOPS_LOGE(context, "The call to TextureDataSaver's fnOpen failed for texture %zu.", i);
      results[i] = result;
      return result;
    }

    VkCommandBuffer cmd = (tex->needsDevice() && !tex->needsHost()) ? context->m_vk->m_cmdPoolGCT.createCommandBuffer() : VK_NULL_HANDLE;
    nvvk::StagingMemoryManager* staging = cmd ? context->m_vk->m_resourceAllocator.getStaging() : nullptr;

    std::vector<const void*> readDatas(mipCount);

    if(cmd)
    {
      cmdTextureLayoutBarrier(cmd, tex, tex->m_vk.imageLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    }

    for(uint32_t m = 0; m < mipCount; m++)
    {
      size_t mipSize = meshopsTextureGetMipDataSize(tex, 0);

      if(cmd)
      {
        VkOffset3D               offset = {0, 0, 0};
        VkExtent3D               extent = {tex->m_mipSizes[m].x, tex->m_mipSizes[m].y, 1};
        VkImageSubresourceLayers subResource;
        subResource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        subResource.baseArrayLayer = 0;
        subResource.mipLevel       = m;
        subResource.layerCount     = 1;

        readDatas[m] = staging->cmdFromImage(cmd, tex->m_vk.image, offset, extent, subResource, mipSize,
                                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      }
      else
      {
        readDatas[m] = tex->m_mipData[m].data();
      }
    }

    if(cmd)
    {
      staging->finalizeResources();
      cmdTextureLayoutBarrier(cmd, tex, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex->m_vk.imageLayout);
      context->m_vk->m_cmdPoolGCT.submitAndWait(cmd);
    }

    for(uint32_t m = 0; m < mipCount; m++)
    {
      size_t mipSize = meshopsTextureGetMipDataSize(tex, 0);
      result         = saver->fnWriteData(handle, m, mipSize, readDatas[m], saver->fnUserData);
      if(result != micromesh::Result::eSuccess)
      {
        MESHOPS_LOGE(context, "The call to TextureDataSaver's fnWriteData filed for texture %zu, mip %u.", i, m);
        break;
      }
    }

    saver->fnClose(handle, saver->fnUserData);

    if(cmd)
    {
      staging->releaseResources();
    }

    results[i] = result;

    if(result != micromesh::Result::eSuccess)
    {
      // Code above has already printed an error; no need to duplicate it.
      return result;
    }
  }

  return micromesh::Result::eSuccess;
}

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsTextureToData(Context context, Texture tex, size_t dataSize, void* data)
{
  MESHOPS_CHECK_CTX_NONNULL(context);
  MESHOPS_CHECK_NONNULL(context, tex);
  MESHOPS_CHECK_NONNULL(context, dataSize);
  MESHOPS_CHECK_NONNULL(context, data);

  if(dataSize != meshopsTextureGetMipDataSize(tex, 0))
  {
    MESHOPS_LOGE(context,
                 "dataSize (%zu) must be the same as the number of bytes in mip 0 of the texture (%zu), as reported by "
                 "meshopsTextureGetMipDataSize().",
                 dataSize, meshopsTextureGetMipDataSize(tex, 0));
    return micromesh::Result::eInvalidValue;
  }

  VkCommandBuffer cmd = (tex->needsDevice() && !tex->needsHost()) ? context->m_vk->m_cmdPoolGCT.createCommandBuffer() : VK_NULL_HANDLE;
  nvvk::StagingMemoryManager* staging = cmd ? context->m_vk->m_resourceAllocator.getStaging() : nullptr;

  const void* readData = nullptr;

  if(cmd)
  {
    cmdTextureLayoutBarrier(cmd, tex, tex->m_vk.imageLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VkOffset3D               offset = {0, 0, 0};
    VkExtent3D               extent = {tex->m_mipSizes[0].x, tex->m_mipSizes[0].y, 1};
    VkImageSubresourceLayers subResource;
    subResource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    subResource.baseArrayLayer = 0;
    subResource.mipLevel       = 0;
    subResource.layerCount     = 1;

    readData = staging->cmdFromImage(cmd, tex->m_vk.image, offset, extent, subResource, dataSize, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  }
  else
  {
    readData = tex->getImageData();
  }

  if(cmd)
  {
    staging->finalizeResources();
    cmdTextureLayoutBarrier(cmd, tex, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, tex->m_vk.imageLayout);
    context->m_vk->m_cmdPoolGCT.submitAndWait(cmd);
  }

  memcpy(data, readData, dataSize);

  if(cmd)
  {
    staging->releaseResources();
  }

  return micromesh::Result::eSuccess;
}

}  // namespace meshops
