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

#include "nvvk/context_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/pipeline_container.hpp"

#include "toolbox_scene_vk.hpp"
#include "heightmap_rtx.h"

struct ViewerSettings;

class ToolboxSceneRtx
{
public:
  ToolboxSceneRtx(nvvk::Context* ctx, nvvkhl::AllocVma* alloc, uint32_t queueFamilyIndex = 0U);
  ~ToolboxSceneRtx();

  void create(const ViewerSettings&                             settings,
              const std::unique_ptr<micromesh_tool::ToolScene>& scene,
              const std::unique_ptr<ToolboxSceneVk>&            sceneVK,
              bool                                              hasMicroMesh,
              VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                           | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);

  VkAccelerationStructureKHR tlas() { return m_rtBuilder.getAccelerationStructure(); }


  void destroy();

private:
  nvvk::RaytracingBuilderKHR::BlasInput primitiveToGeometry(const micromesh_tool::ToolMesh& mesh,
                                                            VkDeviceAddress                 vertexAddress,
                                                            VkDeviceAddress                 indexAddress);

  //--------------------------------------------------------------------------------------------------
  // Create all bottom level acceleration structures (BLAS)
  //
  virtual void createBottomLevelAS(const ViewerSettings&                             settings,
                                   const std::unique_ptr<micromesh_tool::ToolScene>& scene,
                                   const std::unique_ptr<ToolboxSceneVk>&            sceneVK,
                                   VkBuildAccelerationStructureFlagsKHR              flags,
                                   bool                                              hasMicroMesh,
                                   bool                                              useMicroMesh);

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  virtual void createTopLevelAS(const std::unique_ptr<micromesh_tool::ToolScene>& scene, VkBuildAccelerationStructureFlagsKHR flags);

  nvvk::Context*    m_ctx;
  nvvkhl::AllocVma* m_alloc;

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::RaytracingBuilderKHR m_rtBuilder;

  HrtxPipeline              m_heightmapPipeline{};
  std::vector<HrtxMap>      m_heightmaps;
  std::vector<nvvk::Buffer> m_heightmapDirections;
};
