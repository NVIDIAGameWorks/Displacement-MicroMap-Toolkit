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

#include "meshops_bake_batch.hpp"
#include "meshops/meshops_types.h"
#include "meshops_internal/meshops_context.h"
#include "meshops_bake_vk.hpp"
#include "micromesh/micromesh_operations.h"
#include "micromesh/micromesh_utils.h"
#include <nvh/timesampler.hpp>

namespace meshops {

bool getMemoryUsageVk(VkPhysicalDevice physicalDevice, VkDeviceSize* budget, VkDeviceSize* usage)
{
  VkPhysicalDeviceMemoryBudgetPropertiesEXT memoryBudgetProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 memoryProperties2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &memoryBudgetProperties};
  vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memoryProperties2);
  uint32_t heapIndex = (uint32_t)-1;
  for(uint32_t memoryType = 0; memoryType < memoryProperties2.memoryProperties.memoryTypeCount; memoryType++)
  {
    if(memoryProperties2.memoryProperties.memoryTypes[memoryType].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    {
      heapIndex = memoryProperties2.memoryProperties.memoryTypes[memoryType].heapIndex;
      break;
    }
  }
  if(heapIndex == (uint32_t)-1)
  {
    return false;
  }
  *budget = memoryBudgetProperties.heapBudget[heapIndex];
  *usage  = memoryBudgetProperties.heapUsage[heapIndex];
  return true;
}

// Grow the selected triangles by one. This is needed because raytracing does
// not guarantee watertightness for geometry split over multiple BLAS. That is,
// a ray may miss both batches if it hits exactly on a shared edges. A
// MeshTopology structure is needed to generate indices for the new subset of
// mesh triangles.
static std::vector<uint32_t> makeTriangleSelection(micromesh::OpContext           micromeshContext,
                                                   const micromesh::MeshTopology* topology,
                                                   uint32_t                       batchFirst,
                                                   uint32_t                       batchCount)
{
  std::vector<uint32_t> selectedTriangles;

  micromesh::OpGrowTriangleSelection_input input;
  input.topology      = topology;
  input.triangleFirst = batchFirst;
  input.triangleCount = batchCount;

  selectedTriangles.resize(topology->triangleVertices.count);

  micromesh::OpGrowTriangleSelection_output output;
  micromesh::arraySetDataVec(output.triangleSelection, selectedTriangles);

  micromesh::Result result = micromesh::micromeshOpGrowTriangleSelection(micromeshContext, &input, &output);
  assert(result == micromesh::Result::eSuccess);
  if(result == micromesh::Result::eSuccess)
  {
    selectedTriangles.resize(output.triangleSelectionCount);
  }
  else
  {
    selectedTriangles.clear();
  }

  return selectedTriangles;
}

// Binary search to find an upper-bound. Similar to std::upper_bound but
// operating on a function rather than iterators. Returns the first value in
// [low, high) such that func(value) > target, or high if no such element is
// found. func must be non-decreasing within [low, high).
template <class Value, class Target, class Func>
Value findUpperBound(Value low, Value high, const Target& target, Func func)
{
  while(low < high)
  {
    // Midpoint without signed int overflow from low + (high - low) / 2
    Value mid = (low & high) + ((low ^ high) >> 1);
    if(func(mid) > target)
    {
      high = mid;
    }
    else
    {
      low = mid + 1;
    }
  }
  return low;
}

std::vector<GeometryBatch> computeBatches(Context                        context,
                                          uint64_t                       memLimitBytes,
                                          const micromesh::MeshTopology* topology,
                                          const meshops::MeshView&       referenceMeshView)
{
  nvh::ScopedTimer           t("Computing batches");
  std::vector<GeometryBatch> batches;
  uint32_t                   origTriangleCount = static_cast<uint32_t>(referenceMeshView.triangleCount());

  // If no limit is set, return a single batch for everything at once. Batching
  // also requires a topology which is only available when tessellating for
  // heightmaps.
  if(memLimitBytes == 0 || topology == nullptr || referenceMeshView.triangleSubdivisionLevels.empty())
  {
    GeometryBatch all{0, origTriangleCount, 0, 1, true, {}};
    batches.emplace_back(std::move(all));
    return batches;
  }

  // Function to compute the gpu memory required to process one batch of
  // reference mesh geometry. E.g. the generated vertex data and raytracing
  // acceleration structures.
  uint32_t batchStart      = 0;
  auto estimateBatchMemory = [&context, &topology, &referenceMeshView, &batchStart](uint32_t batchLast) -> uint64_t {
    uint32_t              batchEnd = batchLast + 1;
    std::vector<uint32_t> triangles =
        makeTriangleSelection(context->m_micromeshContext, topology, batchStart, batchEnd - batchStart);
    assert(!triangles.empty());
    uint32_t tessellatedTrianglCount = 0;
    uint32_t tessellatedVertexCount  = 0;
    for(auto& t : triangles)
    {
      tessellatedTrianglCount += micromesh::subdivLevelGetCount(referenceMeshView.triangleSubdivisionLevels[t],
                                                                micromesh::Frequency::ePerMicroTriangle);

      // Conservatively over-estimate no shared vertices
      tessellatedVertexCount += micromesh::subdivLevelGetCount(referenceMeshView.triangleSubdivisionLevels[t],
                                                               micromesh::Frequency::ePerMicroVertex);
    }
    uint64_t memoryEstimate = BakerVK::estimateBatchGpuMemory(context->m_vk->m_ptrs.context->m_device,
                                                              tessellatedTrianglCount, tessellatedVertexCount);
    return memoryEstimate;
  };

  GeometryBatch currentBatch{};
  while(batchStart < origTriangleCount)
  {
    // Binary search to find the biggest triangle selection that fits in the remaining memory
    uint32_t batchEnd = findUpperBound(batchStart, origTriangleCount, memLimitBytes, estimateBatchMemory);

    // Must always include at least one triangle
    if(batchEnd == batchStart)
    {
      batchEnd++;
      MESHOPS_LOGW(context,
                   "Single-triangle batch %zu may exceed remaining memory: %.2f / %.2f MiB. Consider pre-tessellating "
                   "the "
                   "reference mesh or reducing its subdivision levels.",
                   batches.size(), static_cast<double>(estimateBatchMemory(batchEnd - 1)) / 1024.0 / 1024.0,
                   static_cast<double>(memLimitBytes) / 1024.0 / 1024.0);
    }

    //MESHOPS_LOGI(context, "Expecting batch to take %zu bytes", (size_t)estimateBatchMemory(high));

    GeometryBatch batch{
        batchStart, batchEnd - batchStart,
        0,          0,
        false,      makeTriangleSelection(context->m_micromeshContext, topology, batchStart, batchEnd - batchStart)};
    if(batch.triangles.empty())
    {
      MESHOPS_LOGE(context, "Failed to compute triangle selection for batched baking");
      return {};
    }
    batches.emplace_back(std::move(batch));
    batchStart = batchEnd;
  }

  // Store index and total for logging
  for(size_t i = 0; i < batches.size(); ++i)
  {
    batches[i].batchIndex   = (uint32_t)i;
    batches[i].totalBatches = (uint32_t)batches.size();
  }

  return batches;
}

}  // namespace meshops
