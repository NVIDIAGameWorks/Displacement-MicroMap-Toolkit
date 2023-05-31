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

#include <bary/bary_types.h>
#include <baryutils/baryutils.h>
#include <memory>
#include <string>
#include <vector>
#include <filesystem>
#include <micromesh/micromesh_types.h>

namespace micromesh_tool {

namespace fs = std::filesystem;

[[nodiscard]] bool loadBaryFile(const fs::path& filename, baryutils::BaryFile& baryFile);

// Micromap data container that provides a bary::BasicView abstraction for data
// coming either from a .bary file or a BaryContent array.
class ToolBary
{
public:
  ToolBary() = default;

  // Create a ToolBary from a file on disk. The filename is split into base and
  // relative so that the relative portion can be reused when saving the file
  // for a different scene in a new location.
  [[nodiscard]] static std::unique_ptr<ToolBary> create(const fs::path& basePath, const fs::path& relativePath);

  // Create a ToolBary from in-memory BaryContents, taking ownership. The
  // relative portion of the filename is optional. If given, it will be used
  // when saving the scene. An empty relativePath indicates one should be
  // generated when saving.
  [[nodiscard]] static std::unique_ptr<ToolBary> create(std::vector<baryutils::BaryContentData>&& baryContents,
                                                        const fs::path&                           relativePath = {});

  // Copy constructor. Used to deep copy a scene.
  [[nodiscard]] static std::unique_ptr<ToolBary> create(const ToolBary& other);

  // Saves the bary data to disk. The filename is split into base and relative
  // so that the scene can reference the saved location with
  // ToolBary::relativePath().
  [[nodiscard]] bool save(const fs::path& basePath, const fs::path& relativePath);

  [[nodiscard]] bool isOriginalData() const { return static_cast<bool>(m_baryFile); }

  const std::vector<bary::ContentView>& groups() const { return m_views; }
  fs::path&                             relativePath() { return m_relativePath; }
  const fs::path&                       relativePath() const { return m_relativePath; }

private:
  ToolBary& operator=(ToolBary&& other) = default;
  ToolBary(std::unique_ptr<baryutils::BaryFile> baryFile, const fs::path& relativePath);
  ToolBary(std::vector<baryutils::BaryContentData>&& baryContents, const fs::path& relativePath);

  std::vector<bary::ContentView> m_views;

  // Mutually exclusive bary data sources
  std::unique_ptr<baryutils::BaryFile>    m_baryFile;
  std::vector<baryutils::BaryContentData> m_baryContents;

  // Last saved location. Maybe be temporarily empty if created from
  // baryContents and before calling save().
  fs::path m_relativePath;
};

}  // namespace micromesh_tool
