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

#include "vulkan/vulkan_core.h"
#include <micromesh/micromesh_types.h>
#include <meshops/meshops_mesh_view.h>
#include <meshops/meshops_operations.h>
#include <vector>

namespace meshops {

bool getMemoryUsageVk(VkPhysicalDevice physicalDevice, VkDeviceSize* budget, VkDeviceSize* usage);

struct GeometryBatch
{
  // Range of triangles in the batch before growing
  uint32_t triangleOffset;
  uint32_t triangleCount;

  // Batch index for logging
  uint32_t batchIndex;
  uint32_t totalBatches;

  // The subset of triangle indices for the batch after growing to neighbours
  bool                  allTriangles;
  std::vector<uint32_t> triangles;
  size_t                size() const { return allTriangles ? triangleCount : triangles.size(); }
  uint32_t              triangle(uint32_t i) const { return allTriangles ? i : triangles[i]; }
};

std::vector<GeometryBatch> computeBatches(Context                        context,
                                          uint64_t                       memLimitBytes,
                                          const micromesh::MeshTopology* topology,
                                          const meshops::MeshView&       referenceMeshView);

}  // namespace meshops
