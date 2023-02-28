//
// SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//

#pragma once


#include "meshops_context.h"
#include <meshops/meshops_types.h>
#include <meshops/meshops_mesh_view.h>
#include "nvvk/buffers_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvmath/nvmath.h"
#include "octant_encoding.h"
#include "meshops/meshops_operations.h"

namespace meshops {


class DeviceMeshVKData
{
public:
  micromesh::Result createDeviceData(Context context, const MeshView& meshView, DeviceMeshSettings& settings);
  micromesh::Result destroyDeviceData(Context context);
  micromesh::Result createAccelerationStructure(Context context, const MeshView& meshView);
  micromesh::Result allocateVertexImportance(Context context, const MeshView& meshView);
  micromesh::Result uploadBuffers(Context context, const MeshView& meshView, DeviceMeshSettings& settings);

  template <class T>
  T*                map(Context context, nvvk::Buffer b);
  void              unmapAndDestroy(Context context, nvvk::Buffer b);
  micromesh::Result readbackBuffers(Context context, MutableMeshView& meshView, const DeviceMeshSettings& settings);

  nvvk::Buffer readback(Context context, VkCommandBuffer cmd, nvvk::Buffer buffer, size_t sizeInBytes);

  // gets the current device mesh state showing which attributes and usages are currently available on the device
  DeviceMeshSettings getSettings() const;


  VkBuffer getTriangleVertexIndexBuffer();
  VkBuffer getTriangleAttributesBuffer();

  VkBuffer                   getVertexPositionNormalBuffer();
  VkBuffer                   getVertexTangentSpaceBuffer();
  VkBuffer                   getVertexTexcoordBuffer();
  VkBuffer                   getVertexDirectionsBuffer();
  VkBuffer                   getVertexDirectionBoundsBuffer();
  VkBuffer                   getVertexImportanceBuffer();
  VkAccelerationStructureKHR getAccelerationStructure();

private:
  nvvk::Buffer               m_triangleVertexIndex   = {0};
  nvvk::Buffer               m_triangleAttributes    = {0};
  nvvk::Buffer               m_vertexPositionNormal  = {0};
  nvvk::Buffer               m_vertexTangentSpace    = {0};
  nvvk::Buffer               m_vertexTexcoord        = {0};
  nvvk::Buffer               m_vertexDirections      = {0};
  nvvk::Buffer               m_vertexDirectionBounds = {0};
  nvvk::Buffer               m_vertexImportance      = {0};
  nvvk::RaytracingBuilderKHR m_raytracingBuilder;

private:
  template <class T>
  void createBuffer(Context context, VkCommandBuffer cmd, const std::vector<T> hostValues, nvvk::Buffer& buffer, VkBufferUsageFlags additionalFlags = {})
  {
    buffer = context->m_vk->m_resourceAllocator.createBuffer(cmd, hostValues.size() * sizeof(T), hostValues.data(),
                                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                                                 | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                                 | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | additionalFlags,
                                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  }
  template <class T>
  void createBuffer(Context context, VkCommandBuffer cmd, const ArrayView<T> hostValues, nvvk::Buffer& buffer, VkBufferUsageFlags additionalFlags = {})
  {
    //FIXME this only is legal for contiguous ArrayViews
    buffer = context->m_vk->m_resourceAllocator.createBuffer(cmd, hostValues.size() * sizeof(T), hostValues.data(),
                                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                                                 | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                                 | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | additionalFlags,
                                                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  }
};

class DeviceMesh_c
{
public:
  micromesh::Result create(Context context, const MeshView& meshView, DeviceMeshSettings& settings);

  micromesh::Result readback(Context context, MutableMeshView& meshView);
  micromesh::Result readback(Context context, MutableMeshView& meshView, DeviceMeshSettings settings);


  micromesh::Result destroy(Context context);

  micromesh::Result initializeMeshVk(Context context, meshops::MeshAttributeFlags sourceAttribFlags);

  DeviceMeshVK*             getDeviceMeshVk();
  const DeviceMeshSettings& getSettings() const;

private:
  DeviceMeshSettings m_settings;
  DeviceMeshVK       m_vk;
  DeviceMeshVKData   m_vkData;
};


}  // namespace meshops