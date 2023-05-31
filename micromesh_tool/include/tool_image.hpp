/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "micromesh/micromesh_types.h"
#include <cstring>
#include <memory>
#include <filesystem>
#include <imageio/imageio.hpp>
#include <meshops/meshops_array_view.h>
#include <meshops/meshops_operations.h>
#include <meshops_internal/heightmap.hpp>
#include <nvvk/resourceallocator_vk.hpp>
#include <nvh/nvprint.hpp>
#include <tiny_gltf.h>

namespace micromesh_tool {

namespace fs = std::filesystem;

// Single image container to abstract image data source (disk or generated)
// and shuffle it on-demand between disk, system and gpu (vulkan, using meshops)
// memory. Being lazy, any getter can fail when trying to load the file on disk.
class ToolImage
{
public:
  // Struct holding values from imageio::info() and the bit depth loaded.
  // Currently only supports the following formats:
  // - VK_FORMAT_R8G8B8A8_UNORM
  // - VK_FORMAT_R16G16B16A16_UNORM
  // - VK_FORMAT_R16_UNORM
  // TODO: there is no component type, so e.g. floats would not be supported.
  // Maybe just store the vulkan image type and have getters for the other
  // attributes. See GPUTextureContainer.
  struct Info
  {
    size_t width             = 0;
    size_t height            = 0;
    size_t components        = 0;
    size_t componentBitDepth = 0;

    size_t componentBytes() const { return components * (componentBitDepth / 8U); }
    size_t totalPixels() const { return width * height; }
    size_t totalBytes() const { return totalPixels() * componentBytes(); }
    bool   valid() const { return width > 0 && height > 0 && components > 0 && componentBitDepth > 0; }

    VkFormat vkFormat() const
    {
      if(components == 4 && componentBitDepth == 8)
      {
        return VK_FORMAT_R8G8B8A8_UNORM;
      }
      else if(components == 4 && componentBitDepth == 16)
      {
        return VK_FORMAT_R16G16B16A16_UNORM;
      }
      else if(components == 1 && componentBitDepth == 16)
      {
        return VK_FORMAT_R16_UNORM;
      }
      else if(components == 1 && componentBitDepth == 8)
      {
        return VK_FORMAT_R8_UNORM;
      }
      else
      {
        LOGE("Error: image with %zu %zu-bit components unsupported\n", components, componentBitDepth);
        assert(false);
        return VK_FORMAT_UNDEFINED;
      }
    }

    Info() = default;

    // Convenience converter
    Info(const tinygltf::Image& gltfImage)
        : width(gltfImage.width)
        , height(gltfImage.height)
        , components(gltfImage.component)
        , componentBitDepth(gltfImage.bits)
    {
    }

    bool operator==(const Info& other) const
    {
      return width == other.width &&            //
             height == other.height &&          //
             components == other.components &&  //
             componentBitDepth == other.componentBitDepth;
    }
    bool operator!=(const Info& other) const { return !(*this == other); }
  };

  struct ImageioDeleter
  {
    void operator()(imageio::ImageIOData ptr) { imageio::freeData(&ptr); }
  };

  // TODO: remove. Only needed by ToolScene::createImage().
  [[nodiscard]] static std::unique_ptr<ToolImage> createInvalid()
  {
    return std::unique_ptr<ToolImage>(new ToolImage());
  }

  // Create a ToolTexture with source data on disk
  [[nodiscard]] static std::unique_ptr<ToolImage> create(const fs::path& basePath, const fs::path& relativePath)
  {
    // Do not use the empty path for the current working directory. When
    // m_basePath is not empty, we may need to copy the texture to the new
    // relative location, even if it was never loaded into memory.
    assert(basePath.is_absolute());
    if(!basePath.is_absolute())
    {
      return {};
    }

    auto result            = std::unique_ptr<ToolImage>(new ToolImage());
    result->m_basePath     = basePath;
    result->m_relativePath = relativePath;
    fs::path filename      = result->m_basePath / result->m_relativePath;
    if(!imageio::info(filename.string().c_str(), &result->m_info.width, &result->m_info.height, &result->m_info.components))
    {
      LOGE("Error: failed to read %s\n", filename.string().c_str());
      // Return the result. We won't be able to render or resample it, but it
      // may be able to be copied to the new directory when saved.
      return result;
    }

    // The resampler currently always expects 4-component images
    // TODO: support more image formats
    if(result->m_info.components == 3)
    {
      LOGI("Image %s will be converted from rgb to rgba\n", result->m_relativePath.string().c_str());
      // This will be passed to imageio::loadGeneral()'s required_components
      result->m_info.components = 4;
    }

    result->m_info.componentBitDepth = imageio::is16Bit(filename.string().c_str()) ? 16 : 8;
    return result;
  }

  // Create a ToolTexture and allocate source data.
  [[nodiscard]] static std::unique_ptr<ToolImage> create(const Info& info, const fs::path& relativePath)
  {
    if(relativePath.empty())
    {
      LOGE("Error: Creating embedded images is not supported yet\n");
      assert(false);
      return {};
    }
    auto result            = std::unique_ptr<ToolImage>(new ToolImage());
    result->m_relativePath = relativePath;
    result->m_rawData = std::unique_ptr<void, ImageioDeleter>(imageio::allocateData(info.totalBytes()), ImageioDeleter());
    // Note: info may not be valid if copying from a source ToolImage that could not be read
    result->m_info = info;
    return result;
  }

  // Create a ToolTexture, taking ownership of the provided raw data.
  [[nodiscard]] static std::unique_ptr<ToolImage> create(const Info& info, const fs::path& relativePath, imageio::ImageIOData rawData)
  {
    assert(info.valid());
    assert(!relativePath.empty());  // Embedding images not supported yet
    if(!rawData)
    {
      return {};
    }
    auto result            = std::unique_ptr<ToolImage>(new ToolImage());
    result->m_relativePath = relativePath;
    result->m_rawData      = std::unique_ptr<void, ImageioDeleter>(rawData, ImageioDeleter());
    result->m_info         = info;
    return result;
  }

  // Copy constructor
  [[nodiscard]] static std::unique_ptr<ToolImage> create(const ToolImage& other)
  {
    auto result            = std::unique_ptr<ToolImage>(new ToolImage());
    result->m_relativePath = other.m_relativePath;
    // Note: info may not be valid if copying from a source ToolImage that could not be read
    result->m_info = other.m_info;

    // If this is an image from disk, keep it as such so it will be copied later
    result->m_basePath = other.m_basePath;

    // If the image has already been read and decompressed from disk, copy the in-memory data
    if(other.m_rawData)
    {
      result->m_rawData =
          std::unique_ptr<void, ImageioDeleter>(imageio::allocateData(other.m_info.totalBytes()), ImageioDeleter());
      memcpy(result->m_rawData.get(), other.raw(), result->m_info.totalBytes());
    }

    return result;
  }

  [[nodiscard]] bool save(const fs::path& basePath, const fs::path& relativePath);

#if 0
  // TODO: Either of these can be called first depending on usage. It is
  // possible to extract an nvvk::Texture from meshops and also create a meshops
  // texture from an nvvk one.
  meshops::Texture meshopsTexture(meshops::Context context, meshops::TextureUsageFlags texUsage) const;
  nvvk::Texture    nvvkTexture();
#endif

  std::unique_ptr<HeightMap>& heigtmap() const
  {
    // The HeightMap structure expects 32 bit float data and does not have a way
    // to sample anything else. For now, create a second copy of the image data
    // in float format for it to use.
    if(!m_heightmapData)
    {
      if(!raw())
        return m_heightmap;

      // TODO: add an imageio API to convert without this copy
      imageio::ImageIOData heightmapData = imageio::allocateData(m_info.totalBytes());
      memcpy(heightmapData, raw(), m_info.totalBytes());
      bool ok         = imageio::convertFormat(&heightmapData, m_info.width, m_info.height, m_info.components,
                                               m_info.componentBitDepth, 1, 32);
      m_heightmapData = std::unique_ptr<void, ImageioDeleter>(heightmapData, ImageioDeleter());
      if(ok)
      {
        m_heightmap = std::make_unique<HeightMap>(static_cast<int>(m_info.width), static_cast<int>(m_info.height),
                                                  reinterpret_cast<float*>(m_heightmapData.get()));
      }
      else
      {
        LOGE("Failed to convert heightmap data in %s\n", relativePath().string().c_str());
      }
    }
    return m_heightmap;
  }

  imageio::ImageIOData raw() const
  {
    // It is possible the file is an unknown format or simply doesn't exist, in
    // which case m_info will not be valid. In this case, ToolImage serves just
    // as a reference to the original path.
    if(m_info.valid() && !m_rawData && !m_loadAttempted)
    {
      m_loadAttempted = true;
      m_rawData       = load(m_basePath / m_relativePath);
    }
    return m_rawData.get();
  }
  const Info&                  info() const { return m_info; }
  fs::path&                    relativePath() { return m_relativePath; }
  const fs::path&              relativePath() const { return m_relativePath; }

  template <class T>
  meshops::ArrayView<T> array()
  {
    if(sizeof(T) != m_info.componentBytes())
    {
      assert(!"unexpected texture bit depth");
      return {};
    }
    imageio::ImageIOData ptr = raw();
    if(!ptr)
    {
      return {};
    }
    return {reinterpret_cast<T*>(ptr), m_info.totalPixels(), static_cast<ptrdiff_t>(m_info.componentBytes())};
  }

  // Returns true if the data is unmodified from what was loaded
  [[nodiscard]] bool isOriginalData() const
  {
    // Images are never modified in-place. If a base path exists, it is still
    // the same as what was loaded from disk. Otherwise it is probably
    // generated.
    return !m_basePath.empty();
  }

private:
  // Private default constructor. Use ::createInvalid() to be more explicit.
  ToolImage() = default;

  std::unique_ptr<void, ImageioDeleter> load(const fs::path& path) const;

#if 0
  // Lazily created gpu texture for meshops
  mutable meshops::Texture           m_meshopsTexture;
  mutable meshops::TextureUsageFlags m_meshopsTextureUsage;

  // Lazily created gpu texture for rendering
  // If this exists, m_meshopsTexture will be created from it on first use.
  mutable nvvk::Texture m_nvvkTexture;
#endif

  // Lazily loaded heightmap object. Holds m_rawData converted to floats
  mutable std::unique_ptr<HeightMap>            m_heightmap;
  mutable std::unique_ptr<void, ImageioDeleter> m_heightmapData;

  // Lazily loaded raw data
  mutable std::unique_ptr<void, ImageioDeleter> m_rawData;

  // Absolute texture path, if the source is from disk. Otherwise empty.
  fs::path m_basePath;

  // Relative path that is kept if the texture is saved.
  fs::path m_relativePath;

  // Populated on creation
  Info m_info;

  // Avoid spamming the console with error messages on every lazy access.
  mutable bool m_loadAttempted{false};
};

}  // namespace micromesh_tool
