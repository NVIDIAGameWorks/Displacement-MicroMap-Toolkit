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

#include <meshops/meshops_operations.h>

#include <nvvk/context_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>
#include <nvvk/commands_vk.hpp>
#include "nvvk/raytraceKHR_vk.hpp"

typedef struct VmaAllocator_T* VmaAllocator;

namespace meshops {

struct ContextVK
{
  nvvk::Context* context = nullptr;
  VmaAllocator   vma     = nullptr;

  // Optional
  nvvk::ResourceAllocator* resAllocator = nullptr;

  // Optional. Override context queues if the queue is not VK_NULL_HANDLE
  nvvk::Context::Queue queueGCT;
  nvvk::Context::Queue queueT;
  nvvk::Context::Queue queueC;
};

struct TextureVK
{
  VkImage           image;
  VkImageView       imageView;
  VkImageCreateInfo imageCreateInfo;
  VkImageLayout     imageLayout;
};

struct DeviceMeshTriangleAttributesVK
{
  uint16_t subdLevel      = 0;
  uint8_t  primitiveFlags = 0;
  uint8_t  openEdgeFlags  = 0;
  // (uint8 openEdgeFlags not yet implemented, filled from meshtopology,
  //  we can visualize via barycentrics in frag shader)
};

struct DeviceMeshVK
{
  DeviceMeshUsageFlags usageFlags = 0;

  // Indicates which attributes are real or generated/default-initialized.
  meshops::MeshAttributeFlags sourceAttribFlags = 0;

  // Indicates which buffers are requested by meshopsDeviceMeshCreate().
  meshops::MeshAttributeFlags deviceAttribFlags = 0;

  // uint32 x 3
  VkDescriptorBufferInfo triangleVertexIndexBuffer = {0};

  // uint32:
  // see DeviceMeshTriangleAttributesVK
  VkDescriptorBufferInfo triangleAttributesBuffer = {0};

  // fp32 x 3 + octant normal (snorm16x2)
  VkDescriptorBufferInfo vertexPositionNormalBuffer = {0};

  // 2 x octant normal (snorm16x2)
  VkDescriptorBufferInfo vertexTangentSpaceBuffer = {0};

  // n x fp32 x 2
  VkDescriptorBufferInfo vertexTexcoordBuffer = {0};
  uint32_t               vertexTexcoordCount  = 0;

  // fp16 x 4
  VkDescriptorBufferInfo vertexDirectionsBuffer = {0};
  // fp32 x 2
  VkDescriptorBufferInfo vertexDirectionBoundsBuffer = {0};

  // 1 x fp16
  // used by remesher
  VkDescriptorBufferInfo vertexImportanceBuffer = {0};

  VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
};


//////////////////////////////////////////////////////////////////////////

// create from existing context
// `vma` is optional
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsContextCreateVK(const ContextConfig& config,
                                                                  const ContextVK&     sharedContextVK,
                                                                  Context*             pContext);

// Logs and returns an error if texture creation failed.
MESHOPS_API micromesh::Result MESHOPS_CALL meshopsTextureCreateVK(Context              context,
                                                                  TextureUsageFlags    usageFlags,
                                                                  const TextureConfig& config,
                                                                  const TextureVK&     source,
                                                                  Texture*             pTexture);

// must account for all possible operations
MESHOPS_API void MESHOPS_CALL meshopsGetContextRequirements(const ContextConfig&     config,
                                                            nvvk::ContextCreateInfo& createInfo,
                                                            std::vector<uint8_t>&    createInfoData);

// get vk details, can be nullptr if context was created without vk support
MESHOPS_API ContextVK* MESHOPS_CALL meshopsContextGetVK(Context context);

// get vk details, can be nullptr if context was created without vk support
MESHOPS_API TextureVK* MESHOPS_CALL meshopsTextureGetVK(Texture texture);

// get vk details, can be nullptr if context was created without vk support
MESHOPS_API DeviceMeshVK* MESHOPS_CALL meshopsDeviceMeshGetVK(DeviceMesh mesh);

}  // namespace meshops
