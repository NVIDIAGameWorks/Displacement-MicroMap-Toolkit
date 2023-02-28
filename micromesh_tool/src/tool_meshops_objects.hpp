
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
// Provides RAII-style thin wrappers for meshops_core objects and vulkan object allocation

#pragma once

#include <meshops/meshops_operations.h>
#include <meshops/meshops_vk.h>
#include <nvvk/context_vk.hpp>
#include <nvvk/memallocator_vma_vk.hpp>
#include <nvvk/memorymanagement_vk.hpp>

namespace micromesh_tool {

template <class Operator, micromesh::Result (*FuncCreate)(meshops::Context, Operator*), void (*FuncDestroy)(meshops::Context, Operator)>
class MeshopsOperator
{
public:
  MeshopsOperator(const MeshopsOperator& other) = delete;
  MeshopsOperator(meshops::Context context)
      : m_context(context)
  {
    m_createResult = FuncCreate(m_context, &m_operator);
    if(m_createResult != micromesh::Result::eSuccess)
      assert(0 && "meshops operator");
  }
  ~MeshopsOperator() { FuncDestroy(m_context, m_operator); }
       operator Operator() { return m_operator; }
  bool valid() const { return m_createResult == micromesh::Result::eSuccess; }

private:
  meshops::Context  m_context      = nullptr;
  Operator          m_operator     = nullptr;
  micromesh::Result m_createResult = micromesh::Result::eFailure;
};

using BakeOperator =
    MeshopsOperator<meshops::BakerOperator, meshops::meshopsBakeOperatorCreate, meshops::meshopsBakeOperatorDestroy>;
using RemeshingOperator =
    MeshopsOperator<meshops::RemeshingOperator, meshops::meshopsRemeshingOperatorCreate, meshops::meshopsRemeshingOperatorDestroy>;
using GenerateImportanceOperator =
    MeshopsOperator<meshops::GenerateImportanceOperator, meshops::meshopsGenerateImportanceOperatorCreate, meshops::meshopsGenerateImportanceOperatorDestroy>;

// Combines a Vulkan image with information about the image.
struct GPUTextureContainer
{
  nvvk::Texture     texture{};
  VkImageCreateInfo info{};
  uint8_t           texcoordIndex = 0;

  // The BakerManager moves textures between disk and GPU to save VRAM, so
  // this keeps track of where the texture's data currently is.
  enum class Storage
  {
    // Default; we haven't loaded it yet, we never use this (e.g. it's a
    // placeholder in a vector), or it's no longer used.
    eUnknownOrUnused,
    // Create this image in VRAM when it's first used. "All-default, but not
    // stored anywhere." Resolution stored in `info`.
    eCreateOnFirstUse,
    // On disk, e.g. as a .png or .jpg.
    eImageFile,
    // From the scene (disk, embedded, generated at runtime)
    eToolImage,
    // On disk, but in a raw format that's generally faster to read and write.
    eCachedFile,
    // In the GPU's memory.
    eVRAM
  };
  Storage storageLocation = Storage::eUnknownOrUnused;

  // Final input or output location for this image, absolute, or relative to the current working directory. Unused for eImageFormatInBuffer.
  std::string filePath;

  uint64_t bytesPerComponent() const
  {
    switch(info.format)
    {
      case VK_FORMAT_R8G8B8A8_UNORM:
        return 1;
      case VK_FORMAT_R16_UNORM:
      case VK_FORMAT_R16G16B16A16_UNORM:
        return 2;
      case VK_FORMAT_R32_SFLOAT:
        return 4;
      default:
        assert(!"Unhandled case in bytesPerComponent()!");
        return 0;
    }
  }

  uint64_t bytesPerPixel() const
  {
    switch(info.format)
    {
      case VK_FORMAT_R16_UNORM:
        return 2;
      case VK_FORMAT_R8G8B8A8_UNORM:
      case VK_FORMAT_R32_SFLOAT:
        return 4;
      case VK_FORMAT_R16G16B16A16_UNORM:
        return 8;
      default:
        assert(!"Unhandled case in bytesPerPixel()!");
        return 0;
    }
  }

  uint64_t mipSizeInBytes(uint32_t mip) const
  {
    const uint64_t mipWidth  = std::max(1u, info.extent.width >> mip);
    const uint64_t mipHeight = std::max(1u, info.extent.height >> mip);
    return bytesPerPixel() * uint64_t(mipWidth) * uint64_t(mipHeight);
  }

  uint64_t fullSizeInBytes() const
  {
    uint64_t totalSize = 0;
    for(uint32_t i = 0; i < info.mipLevels; i++)
    {
      totalSize += mipSizeInBytes(i);
    }
    assert(totalSize > 0);  // Did you set info.mipLevels?
    return totalSize;
  }
};

class MeshopsTexture
{
public:
  MeshopsTexture(const MeshopsTexture& other) = delete;
  MeshopsTexture(meshops::Context context, meshops::TextureUsageFlags texUsage, const GPUTextureContainer& source)
      : m_context(context)
  {
    meshops::TextureVK     inputTexture{source.texture.image, source.texture.descriptor.imageView, source.info,
                                    source.texture.descriptor.imageLayout};
    meshops::TextureConfig inputConfig;
    // FIXME strictly speaking we would need to lower a BC compression format into an
    // uncompressed format here, but given noone is using this yet ;)
    inputConfig.baseFormat       = (micromesh::Format)source.info.format;
    inputConfig.internalFormatVk = source.info.format;
    inputConfig.width            = source.info.extent.width;
    inputConfig.height           = source.info.extent.height;
    inputConfig.mips             = source.info.mipLevels;

    m_createResult = meshops::meshopsTextureCreateVK(context, texUsage, inputConfig, inputTexture, &m_texture);
    if(m_createResult != micromesh::Result::eSuccess)
      assert(0 && "meshopsTextureCreateVK() failed.");
  }
  MeshopsTexture(meshops::Context context, meshops::TextureUsageFlags texUsage, const meshops::TextureConfig& config, size_t dataSize, const void* data)
      : m_context(context)
  {
    m_createResult = meshops::meshopsTextureCreateFromData(context, texUsage, config, dataSize, data, &m_texture);
    if(m_createResult != micromesh::Result::eSuccess)
      assert(0 && "meshopsTextureCreateFromData() failed.");
  }
  MeshopsTexture(meshops::Context                context,
                 meshops::TextureUsageFlags      texUsage,
                 const meshops::TextureConfig&   config,
                 const micromesh::MicromapValue* fillValue)
      : m_context(context)
  {
    m_createResult = meshops::meshopsTextureCreate(context, texUsage, config, fillValue, &m_texture);
    if(m_createResult != micromesh::Result::eSuccess)
      assert(0 && "meshopsTextureCreate() failed.");
  }
  ~MeshopsTexture() { meshops::meshopsTextureDestroy(m_context, m_texture); }
       operator meshops::Texture() { return m_texture; }
  bool valid() const { return m_createResult == micromesh::Result::eSuccess; }

private:
  meshops::Context  m_context      = nullptr;
  meshops::Texture  m_texture      = nullptr;
  micromesh::Result m_createResult = micromesh::Result::eFailure;
};

inline micromesh::Result buildTopologyData(meshops::Context context, const meshops::MeshView& meshView, meshops::MeshTopologyData& topologyData)
{
  meshops::OpBuildTopology_input input;
  input.meshView = meshView;
  meshops::OpBuildTopology_output output;
  output.meshTopology      = &topologyData;
  micromesh::Result result = meshops::meshopsOpBuildTopology(context, 1, &input, &output);
  assert(result == micromesh::Result::eSuccess && "buildTopologyData()");

  return result;
}

}  // namespace micromesh_tool
