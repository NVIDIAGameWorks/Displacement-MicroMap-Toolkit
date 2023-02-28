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

#include <tool_bary.hpp>
#include <nvh/nvprint.hpp>

namespace micromesh_tool {

micromesh::Result ToolBary::create(const fs::path& basePath, const fs::path& relativePath)
{
  fs::path filename = basePath / relativePath;

  auto baryFile = std::make_unique<baryutils::BaryFile>();
  if(!loadBaryFile(filename, *baryFile))
  {
    LOGE("Error: failed to load '%s'\n", filename.string().c_str());
    return micromesh::Result::eFailure;
  }
  *this = ToolBary(std::move(baryFile), relativePath);
  return micromesh::Result::eSuccess;
}

micromesh::Result ToolBary::create(std::vector<baryutils::BaryContentData>&& baryContents, const fs::path& relativePath)
{
  *this = ToolBary(std::move(baryContents), relativePath);
  return micromesh::Result::eSuccess;
}

micromesh::Result ToolBary::create(const ToolBary& other)
{
  std::vector<baryutils::BaryContentData> baryContents;
  for(auto& group : other.groups())
  {
    baryContents.emplace_back(group);
  }
  *this = ToolBary(std::move(baryContents), other.relativePath());
  return micromesh::Result::eSuccess;
}

void ToolBary::destroy()
{
  m_baryFile.reset();
  m_baryContents.clear();
}

// Returns a new ContentView with just one bary::Group. Note that the basic
// triangle count will still have the original count, which is required as the
// group's first triangle offset must still be applied and the group itself
// cannot be modified.
bary::ContentView sliceBaryContentView(const bary::ContentView& view, uint32_t groupIndex, uint32_t groupCount)
{
  assert(groupIndex + groupCount <= view.basic.groupsCount);

  bary::ContentView groupView{view};
  groupView.basic.groupsCount = groupCount;
  groupView.basic.groups += groupIndex;
  if(view.basic.groupHistogramRangesCount)
  {
    groupView.basic.groupHistogramRangesCount = groupCount;
    groupView.basic.groupHistogramRanges += groupIndex;
  }
  if(view.mesh.meshGroupsCount)
  {
    groupView.mesh.meshGroupsCount = groupCount;
    groupView.mesh.meshGroups += groupIndex;
  }
  if(view.mesh.meshGroupHistogramRangesCount)
  {
    groupView.mesh.meshGroupHistogramRangesCount = groupCount;
    groupView.mesh.meshGroupHistogramRanges += groupIndex;
  }
  if(view.misc.groupUncompressedMipsCount)
  {
    groupView.misc.groupUncompressedMipsCount = groupCount;
    groupView.misc.groupUncompressedMips += groupIndex;
  }
  return groupView;
}

ToolBary::ToolBary(std::unique_ptr<baryutils::BaryFile> baryFile, const fs::path& relativePath)
    : m_baryFile(std::move(baryFile))
    , m_relativePath(relativePath)
{
  // Create a BasicView for each group. This makes a consistent abstraction for
  // bary content from .bary files and bary content from separate
  // BaryContentData structures.
  // TODO: move this to micromesh_util?
  const bary::ContentView& view = m_baryFile->getContent();
  assert(!view.mesh.meshGroupsCount || view.basic.groupsCount == view.mesh.meshGroupsCount);
  assert(!view.mesh.meshGroupHistogramRangesCount || view.basic.groupsCount == view.mesh.meshGroupHistogramRangesCount);
  assert(!view.misc.groupUncompressedMipsCount || view.basic.groupsCount == view.misc.groupUncompressedMipsCount);
  assert(!view.misc.triangleUncompressedMipsCount || view.basic.trianglesCount == view.misc.triangleUncompressedMipsCount);
  for(size_t i = 0; i < view.basic.groupsCount; ++i)
  {
    m_views.push_back(sliceBaryContentView(view, static_cast<uint32_t>(i), 1));
  }
}

ToolBary::ToolBary(std::vector<baryutils::BaryContentData>&& baryContents, const fs::path& relativePath)
    : m_baryContents(baryContents)
    , m_relativePath(relativePath)
{
  for(auto& content : m_baryContents)
  {
    m_views.push_back(content.getView());
  }
}

bool loadBaryFile(const fs::path& filename, baryutils::BaryFile& baryFile)
{
  baryutils::BaryFileOpenOptions baryOpenOptions = {};
  bary::Result                   baryResult      = baryFile.open(filename.string().c_str(), &baryOpenOptions);
  if(baryResult != bary::Result::eSuccess)
  {
    LOGE("Error: Failed to load .bary file %s with code %s\n", filename.string().c_str(), bary::baryResultGetName(baryResult));
    return false;
  }

  baryResult = baryFile.validate(bary::ValueSemanticType::eDisplacement);
  if(baryResult != bary::Result::eSuccess)
  {
    LOGE("Error: Failed to validate .bary file %s with code %s\n", filename.string().c_str(), bary::baryResultGetName(baryResult));
    return false;
  }
  return true;
}

bool ToolBary::save(const fs::path& basePath, const fs::path& relativePath)
{
  fs::path filename = basePath / relativePath;

  // No tool should be saving anything with a reference to the original bary
  // file and if so it's likely a bug.
  if(m_baryContents.empty())
  {
    assert(m_baryFile);  // no m_baryContents implies m_baryFile exists
    LOGE("Error: re-saving '%s' (loaded from disk) is not implemented\n", filename.string().c_str());
    return false;
  }

  // The saver operates on view pointers, which must remain valid
  std::vector<bary::ContentView> baryContentViews;
  for(auto& baryContent : m_baryContents)
  {
    baryContentViews.push_back(baryContent.getView());
  }

  LOGI("Writing %s\n", filename.string().c_str());
  assert(m_baryContents.size());
  bary::StandardPropertyType errorProp;
  baryutils::BarySaver       saver;
  bary::Result               result = saver.initContent(&baryContentViews[0], &errorProp);
  if(result != bary::Result::eSuccess)
  {
    LOGE("Error: Failure writing %s group 0\n", filename.string().c_str());
    return false;
  }
  for(size_t i = 1; i < baryContentViews.size(); ++i)
  {
    result = saver.appendContent(&baryContentViews[i], &errorProp);
    if(result != bary::Result::eSuccess)
    {
      LOGE("Error: Failure writing %s group %zu\n", filename.string().c_str(), i);
      return false;
    }
  }
  std::error_code       ec;
  fs::path              directory = filename.parent_path();
  if(!directory.empty() && !fs::exists(directory) && !fs::create_directories(directory, ec))
  {
    LOGE("Error: Failure creating %s\n", directory.string().c_str());
    return false;
  }
  result = saver.save(filename.string());
  if(result != bary::Result::eSuccess)
  {
    LOGE("Error: Failure writing %s\n", filename.string().c_str());
    return false;
  }

  m_relativePath = relativePath;
  return true;
}

}  // namespace micromesh_tool
