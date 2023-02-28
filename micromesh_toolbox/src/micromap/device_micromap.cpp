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

#include "device_micromap.hpp"
#include "bary/bary_types.h"
#include "micromap/microdisp_shim.hpp"
#include <micromesh/micromesh_operations.h>
#include <nvh/alignment.hpp>
#include <nvvk/buffers_vk.hpp>
#include <nvvk/error_vk.hpp>
#include <meshops_internal/meshops_context.h>

static_assert(sizeof(VkMicromapTriangleEXT) == sizeof(bary::Triangle),
              "sizeof mismatch VkMicromapTriangleEXT and bary::Triangle");
static_assert(offsetof(VkMicromapTriangleEXT, dataOffset) == offsetof(bary::Triangle, valuesOffset),
              "dataOffset mismatch VkMicromapTriangleEXT and bary::Triangle");
static_assert(offsetof(VkMicromapTriangleEXT, subdivisionLevel) == offsetof(bary::Triangle, subdivLevel),
              "subdivLevel mismatch VkMicromapTriangleEXT and bary::Triangle");
static_assert(offsetof(VkMicromapTriangleEXT, format) == offsetof(bary::Triangle, blockFormat),
              "format mismatch VkMicromapTriangleEXT and bary::Triangle");

void DeviceMicromap::init(meshops::Context         meshopsContext,
                          nvvk::ResourceAllocator& alloc,
                          VkQueue                  queue,
                          uint32_t                 queueFamily,
                          VkCommandBuffer          cmd,
                          uint64_t                 usageFlags,
                          const bary::ContentView& bary,
                          uint8_t*                 decimateEdgeFlags)
{
  m_device                     = alloc.getDevice();
  const bary::BasicView& basic = bary.basic;

  // For each element of pInfos, its scratchData.deviceAddress member must: be a
  // multiple of
  // VkPhysicalDeviceAccelerationStructurePropertiesKHR::minAccelerationStructureScratchOffsetAlignment
  VkPhysicalDeviceProperties2 props2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  props2.pNext = &asProps;
  vkGetPhysicalDeviceProperties2(alloc.getPhysicalDevice(), &props2);
  VkDeviceSize scratchAlignment = asProps.minAccelerationStructureScratchOffsetAlignment;

  // For simplicity, BasicView is expected to be split up into views of single
  // groups by this point. This is done largely because tool_bake produces a
  // separate BaryContentData for each mesh and these are not concatenated until
  // the final .bary file is written.
  if(basic.groupsCount != 1)
  {
    LOGE("Error: DeviceMicromap does not support multiple groups\n");
    assert(false);
    abort();
    return;
  }

  if(basic.valuesInfo->valueFormat != bary::Format::eDispC1_r11_unorm_block)
  {
    LOGE("Error: DeviceMicromap does not support uncompressed bary values\n");
    assert(false);
    abort();
    return;
  }

  size_t             groupIndex = 0;
  const bary::Group& baryGroup  = basic.groups[groupIndex];

  uint32_t baryBufferFlags = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  if(usageFlags & eDeviceMicromeshUsageRaytracingBit)
  {
    baryBufferFlags |= VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT;
  }
  if(usageFlags & eDeviceMicromeshUsageRasterizingBit)
  {
    baryBufferFlags |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  }

  // Upload the compressed bary values Note that buffers passed to
  // vkCmdBuildMicromapsEXT must be 256 byte aligned. This code does not pack
  // multiple micromeshes into the one buffer, so no manual alignment is needed
  // beyond that already done by alloc.createBuffer().
  const uint8_t* values      = basic.values + baryGroup.valueFirst * basic.valuesInfo->valueByteSize;
  size_t         valuesBytes = basic.valuesInfo->valueByteSize * baryGroup.valueCount;
  m_baryValues               = alloc.createBuffer(cmd, valuesBytes, values, baryBufferFlags);

  // Upload per-base triangle attributes
  const bary::Triangle* triangles      = basic.triangles + baryGroup.triangleFirst;
  size_t                trianglesBytes = sizeof(*triangles) * baryGroup.triangleCount;
  m_baryTriangles                      = alloc.createBuffer(cmd, trianglesBytes, triangles, baryBufferFlags);

  if(usageFlags & eDeviceMicromeshUsageRaytracingBit)
  {
    m_raytrace.emplace();
    m_raytrace->usages.resize(basic.groupHistogramRanges[groupIndex].entryCount);
    const bary::HistogramEntry* histoEntries = basic.histogramEntries + basic.groupHistogramRanges[groupIndex].entryFirst;
    for(size_t i = 0; i < m_raytrace->usages.size(); i++)
    {
      m_raytrace->usages[i].count            = histoEntries[i].count;
      m_raytrace->usages[i].format           = histoEntries[i].blockFormat;
      m_raytrace->usages[i].subdivisionLevel = histoEntries[i].subdivLevel;
    }

    // Compute required buffer sizes
    VkMicromapBuildInfoEXT buildInfo = {VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT};
    buildInfo.type                   = VK_MICROMAP_TYPE_DISPLACEMENT_MICROMAP_NV;
    buildInfo.flags                  = 0;
    buildInfo.mode                   = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
    buildInfo.dstMicromap            = VK_NULL_HANDLE;
    buildInfo.usageCountsCount       = uint32_t(m_raytrace->usages.size());
    buildInfo.pUsageCounts           = m_raytrace->usages.data();

    VkMicromapBuildSizesInfoEXT sizeInfo = {VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT};
    vkGetMicromapBuildSizesEXT(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &sizeInfo);
    assert(sizeInfo.micromapSize && "sizeInfo.micromeshSize was zero");

    // Black box buffer populated during vkCmdBuildMicromapsEXT
    m_raytrace->micromapData = alloc.createBuffer(sizeInfo.micromapSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                                             | VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT);

    // Create micromap object
    VkMicromapCreateInfoEXT mmCreateInfo = {VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT};
    mmCreateInfo.createFlags             = 0;
    mmCreateInfo.buffer                  = m_raytrace->micromapData.buffer;
    mmCreateInfo.offset                  = 0;
    mmCreateInfo.size                    = sizeInfo.micromapSize;
    mmCreateInfo.type                    = VK_MICROMAP_TYPE_DISPLACEMENT_MICROMAP_NV;
    mmCreateInfo.deviceAddress           = 0ull;
    NVVK_CHECK(vkCreateMicromapEXT(m_device, &mmCreateInfo, nullptr, &m_raytrace->micromap));

    // Barrier for bary value and triangle data upload
    VkMemoryBarrier2 memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memBarrier.srcStageMask     = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    memBarrier.srcAccessMask    = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    memBarrier.dstStageMask     = VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT;
    memBarrier.dstAccessMask    = VK_ACCESS_2_MICROMAP_READ_BIT_EXT;
    VkDependencyInfo depInfo    = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depInfo.memoryBarrierCount  = 1;
    depInfo.pMemoryBarriers     = &memBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);

    // The driver may use this
    VkDeviceSize scratchSize = nvh::align_up(std::max(sizeInfo.buildScratchSize, VkDeviceSize(4)), scratchAlignment);
    m_raytrace->scratchData  = alloc.createBuffer(scratchSize, VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT
                                                                   | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    // Build the micromap structure
    buildInfo.dstMicromap                 = m_raytrace->micromap;
    buildInfo.scratchData.deviceAddress   = nvvk::getBufferDeviceAddress(m_device, m_raytrace->scratchData.buffer);
    buildInfo.data.deviceAddress          = nvvk::getBufferDeviceAddress(m_device, m_baryValues.buffer);
    buildInfo.triangleArray.deviceAddress = nvvk::getBufferDeviceAddress(m_device, m_baryTriangles.buffer);
    buildInfo.triangleArrayStride         = sizeof(VkMicromapTriangleEXT);
    vkCmdBuildMicromapsEXT(cmd, 1, &buildInfo);
  }

  if(usageFlags & eDeviceMicromeshUsageRasterizingBit)
  {
    m_raster.emplace();
    microdisp::MicromeshSubTriangleDecoderVK decoder(m_raster->micromeshSet);
    microdisp::ResourcesVK                   res(alloc, cmd);
    uint32_t numThreads = micromesh::micromeshOpContextGetConfig(meshopsContext->m_micromeshContext).threadCount;
    decoder.init(res, bary, decimateEdgeFlags, bary.basic.groups[0].maxSubdivLevel, true, false, numThreads);
  }
}

void DeviceMicromap::deinit(nvvk::ResourceAllocator& alloc)
{
  if(m_raytrace)
  {
    alloc.destroy(m_raytrace->micromapData);
    alloc.destroy(m_raytrace->scratchData);
    vkDestroyMicromapEXT(m_device, m_raytrace->micromap, nullptr);
  }

  if(m_raster)
  {
    microdisp::ResourcesVK res(alloc, VK_NULL_HANDLE);
    m_raster->micromeshSet.deinit(res);
  }

  alloc.destroy(m_baryTriangles);
  alloc.destroy(m_baryValues);
}

void DeviceBary::addMicromap(meshops::Context                meshopsContext,
                             nvvk::ResourceAllocator&        alloc,
                             VkQueue                         queue,
                             uint32_t                        queueFamily,
                             VkCommandBuffer                 cmd,
                             uint64_t                        usageFlags,
                             const bary::ContentView&        bary,
                             const micromesh_tool::ToolMesh& mesh)
{
  m_micromaps.emplace_back();
  m_micromaps.back().init(meshopsContext, alloc, queue, queueFamily, cmd, usageFlags, bary,
                          mesh.view().trianglePrimitiveFlags.data());
}

void DeviceBary::addEmpty()
{
  m_micromaps.emplace_back();
}

void DeviceBary::deinit(nvvk::ResourceAllocator& alloc)
{
  for(auto& micromap : m_micromaps)
    micromap.deinit(alloc);
}
