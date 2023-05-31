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
#include "vulkan/vulkan_core.h"
#include <tool_bary.hpp>
#include <tool_mesh.hpp>
#include <optional>
#include <nvvk/buffers_vk.hpp>
#include <nvvk/context_vk.hpp>
#include <nvvk/raytraceKHR_vk.hpp>
#include <vulkan_nv/vk_nv_micromesh.h>
#include <meshops/meshops_operations.h>
#include <shaders/common_micromesh_compressed.h>
#include <micromap/micromesh_decoder_subtri_vk.hpp>

enum DeviceMicromeshUsageFlagBits : uint64_t
{
  eDeviceMicromeshUsageRaytracingBit  = 1u << 0,
  eDeviceMicromeshUsageRasterizingBit = 1u << 1,
};

class DeviceMicromap
{
public:
  void     init(meshops::Context                        meshopsContext,
                nvvk::ResourceAllocator&                alloc,
                VkQueue                                 queue,
                uint32_t                                queueFamily,
                VkCommandBuffer                         cmd,
                uint64_t                                usageFlags,
                const microdisp::MicromeshSplitPartsVk& splitParts,
                const bary::ContentView&                bary,
                const uint8_t*                          decimateEdgeFlags);
  void     deinit(nvvk::ResourceAllocator& alloc);
  uint64_t valuesAddress() const { return nvvk::getBufferDeviceAddress(m_device, m_baryValues.buffer); }
  uint64_t trianglesAddress() const { return nvvk::getBufferDeviceAddress(m_device, m_baryTriangles.buffer); }

  // Vulkan object referencing m_baryValues, used for raytracing
  struct Raytrace
  {
    VkMicromapEXT                   micromap = nullptr;
    nvvk::Buffer                    micromapData;
    std::vector<VkMicromapUsageEXT> usages;

    // TODO: delete this after init() commands finish
    nvvk::Buffer scratchData;
  };

  //
  struct Raster
  {
    microdisp::MicromeshSetCompressedVK micromeshSet;
  };

  const std::optional<Raytrace>& raytrace() const { return m_raytrace; }
  const std::optional<Raster>&   raster() const { return m_raster; }

private:
  // The device buffers were allocated with
  VkDevice m_device = nullptr;

  // Compressed micromesh displacement values
  // TODO: Not needed after constructing Raytrace
  nvvk::Buffer m_baryValues;

  // Bary triangle data, e.g. subdiv level
  // TODO: Not needed after constructing Raytrace
  // TODO: Not needed for rasterization
  nvvk::Buffer m_baryTriangles;

  std::optional<Raytrace> m_raytrace;
  std::optional<Raster>   m_raster;
};

class DeviceBary
{
public:
  DeviceBary() = default;
  void addMicromap(meshops::Context                        meshopsContext,
                   nvvk::ResourceAllocator&                alloc,
                   VkQueue                                 queue,
                   uint32_t                                queueFamily,
                   VkCommandBuffer                         cmd,
                   uint64_t                                usageFlags,
                   const microdisp::MicromeshSplitPartsVk& splitParts,
                   const bary::ContentView&                bary,
                   const micromesh_tool::ToolMesh&         mesh);
  void addEmpty();
  void deinit(nvvk::ResourceAllocator& alloc);

  const std::vector<DeviceMicromap>& micromaps() { return m_micromaps; }

  // Disable copying
  DeviceBary(const DeviceBary& other)            = delete;
  DeviceBary& operator=(const DeviceBary& other) = delete;

private:
  std::vector<DeviceMicromap> m_micromaps;
};
