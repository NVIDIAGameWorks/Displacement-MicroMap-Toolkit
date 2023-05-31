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

#include "vulkan/vulkan_core.h"
#include <filesystem>
#include <tool_image.hpp>

namespace micromesh_tool {

bool ToolImage::save(const fs::path& basePath, const fs::path& relativePath)
{
  // Do not use the empty path for the current working directory. When
  // m_basePath is not empty, we may need to copy the texture to the new
  // relative location, even if it was never loaded into memory.
  assert(basePath.is_absolute());
  if(!basePath.is_absolute())
  {
    LOGE("Error: ToolImage must be given an absolute path\n");
    return false;
  }
  if(relativePath.empty())
  {
    LOGI("Skipping writing image with no relative path (%zu x %zu)\n", m_info.width, m_info.height);
    return false;
  }

  fs::path oldFilename = m_basePath / m_relativePath;
  fs::path filename    = basePath / relativePath;

  // If the image came from disk it might need to be copied to the new relative
  // location. We never modify source images, so the in-memory data does not
  // need saving.
  if(!m_basePath.empty())
  {
    // If we're saving the same image, no copy is needed.
    if(oldFilename == filename)
    {
      return true;
    }

    LOGI("Copying %s to %s\n", oldFilename.string().c_str(), filename.string().c_str());

    std::error_code ec;
    fs::copy(oldFilename, filename, fs::copy_options::update_existing, ec);
    if(ec)
    {
      LOGE("Error: failed to copy %s to %s: %s\n", oldFilename.string().c_str(), filename.string().c_str(), ec.message().c_str());
      return false;
    }
  }
  else
  {
    VkFormat vkFormat = m_info.vkFormat();
    if(vkFormat == VK_FORMAT_UNDEFINED)
    {
      // An error has been printed already
      return false;
    }

    // The only way to get here without data is if the image was created at
    // runtime (not from a file) but not immediately populated. This is a bug.
    if(!m_rawData)
    {
      LOGE("Error: Generated image %s has no data to save\n", filename.string().c_str());
      assert(false);
      return false;
    }

    // Rewritten images are only saved as .png
    if(m_relativePath.extension() != ".png")
    {
      m_relativePath.replace_extension(".png");
      filename = basePath / m_relativePath;
    }

    LOGI("Writing %s (%zu x %zu)\n", relativePath.string().c_str(), m_info.width, m_info.height);
    if(!imageio::writePNG(filename.string().c_str(), m_info.width, m_info.height, raw(), vkFormat))
    {
      LOGE("Writing %s failed!\n", filename.string().c_str());
      return false;
    }
  }

  // Store the new path for future reference since saving was successful.
  m_basePath     = basePath;
  m_relativePath = relativePath;
  return true;
}

std::unique_ptr<void, ToolImage::ImageioDeleter> ToolImage::load(const fs::path& path) const
{
  Info                                  verify;
  std::unique_ptr<void, ImageioDeleter> result;
  assert(m_info.componentBitDepth);  // should be set in ::create()
  imageio::ImageIOData ptr = imageio::loadGeneral(path.string().c_str(), &verify.width, &verify.height,
                                                  &verify.components, m_info.components, m_info.componentBitDepth);
  if(ptr)
  {
    result = std::unique_ptr<void, ImageioDeleter>(ptr, ImageioDeleter());

    // Loaded bit depth will always match
    verify.componentBitDepth = m_info.componentBitDepth;
    if(verify != m_info)
    {
      LOGE("Error: inconsistent attributes after loading %s\n", path.string().c_str());
      result.reset();
    }
  }
  else
  {
    LOGE("Error: failed to load %s\n", path.string().c_str());
  }
  return result;
}

}  // namespace micromesh_tool
