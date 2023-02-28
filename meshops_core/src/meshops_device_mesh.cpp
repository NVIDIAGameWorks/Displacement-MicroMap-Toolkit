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

#include <meshops_internal/meshops_context.h>
#include <meshops/meshops_operations.h>
#include "meshops_internal/meshops_device_mesh.h"
#include <thread>
#include <vector>
#include "nvh/timesampler.hpp"
#include "nvh/parallel_work.hpp"

#define MIN_ITEMS_PER_THREAD (512U * 1024U)

using namespace meshops;

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsDeviceMeshCreate(Context                  context,
                                                                   const meshops::MeshView& meshView,
                                                                   DeviceMeshSettings&      settings,
                                                                   DeviceMesh*              pDeviceMesh)
{
  (*pDeviceMesh)           = new DeviceMesh_c;
  micromesh::Result result = (*pDeviceMesh)->create(context, meshView, settings);
  // Clean up resources if this wasn't constructed correctly, so that users of
  // meshopsDeviceMeshCreate can assume that if this fails, they don't need to
  // call meshopsDeviceMeshDestroy on an object that was not created.
  if(result != micromesh::Result::eSuccess)
  {
    meshopsDeviceMeshDestroy(context, *pDeviceMesh);
  }
  return result;
}
MESHOPS_API void MESHOPS_CALL meshopsDeviceMeshUpdate(Context                  context,
                                                      DeviceMesh               deviceMesh,
                                                      const meshops::MeshView& meshView,
                                                      DeviceMeshSettings&      settings)
{
  if(deviceMesh)
  {
    deviceMesh->create(context, meshView, settings);
  }
}

MESHOPS_API void MESHOPS_CALL meshopsDeviceMeshReadback(Context context, DeviceMesh deviceMesh, meshops::MutableMeshView& meshView)
{
  if(deviceMesh)
  {
    deviceMesh->readback(context, meshView);
  }
}

MESHOPS_API void MESHOPS_CALL meshopsDeviceMeshReadbackSpecific(Context                   context,
                                                                DeviceMesh                deviceMesh,
                                                                meshops::MutableMeshView& meshView,
                                                                DeviceMeshSettings        attributes)
{
  if(deviceMesh)
  {
    deviceMesh->readback(context, meshView, attributes);
  }
}


MESHOPS_API void MESHOPS_CALL meshopsDeviceMeshDestroy(Context context, DeviceMesh deviceMesh)
{
  if(deviceMesh)
  {
    deviceMesh->destroy(context);
    delete deviceMesh;
  }
}

// get vk details, can be nullptr if context was created without vk support
MESHOPS_API DeviceMeshVK* MESHOPS_CALL meshopsDeviceMeshGetVK(DeviceMesh deviceMesh)
{
  if(deviceMesh)
  {
    return deviceMesh->getDeviceMeshVk();
  }
  return nullptr;
}

static void copyVertexAttributes(size_t                           vertexIndex,
                                 const MeshView&                  meshView,
                                 DeviceMeshSettings&              settings,
                                 std::vector<nvmath::vec4f>&      positionNormal,
                                 std::vector<nvmath::vec2f>&      texCoord,
                                 std::vector<nvmath::vec2ui>&     tangentSpace,
                                 std::vector<glm::detail::hdata>& directions,
                                 std::vector<nvmath::vec2f>&      directionBounds,
                                 std::vector<uint16_t>&           importance)
{

  if(settings.attribFlags & eMeshAttributeVertexPositionBit)
  {
    if(vertexIndex < meshView.vertexPositions.size())
    {
      nvmath::vec3f inputPosition   = meshView.vertexPositions[vertexIndex];
      positionNormal[vertexIndex].x = inputPosition.x;
      positionNormal[vertexIndex].y = inputPosition.y;
      positionNormal[vertexIndex].z = inputPosition.z;
    }
    else
    {
      positionNormal[vertexIndex].x = 0.f;
      positionNormal[vertexIndex].y = 0.f;
      positionNormal[vertexIndex].z = 0.f;
    }
  }
  if(settings.attribFlags & eMeshAttributeVertexNormalBit)
  {
    if(vertexIndex < meshView.vertexNormals.size())
    {
      positionNormal[vertexIndex].w = glm::uintBitsToFloat(nvmath::vec_to_oct32(meshView.vertexNormals[vertexIndex]));
    }
    else
    {
      positionNormal[vertexIndex].w = glm::uintBitsToFloat(nvmath::vec_to_oct32(nvmath::vec3f(0.f, 0.f, 1.f)));
    }
  }
  if(settings.attribFlags & eMeshAttributeVertexTexcoordBit)
  {
    if(vertexIndex < meshView.vertexTexcoords0.size())
    {
      texCoord[vertexIndex] = meshView.vertexTexcoords0[vertexIndex];
    }
    else
    {
      texCoord[vertexIndex] = nvmath::vec2f(0.f);
    }
  }
  if(settings.attribFlags & eMeshAttributeVertexTangentBit)
  {
    if(vertexIndex < meshView.vertexTangents.size())
    {
      tangentSpace[vertexIndex].x = nvmath::vec_to_oct32(nvmath::vec3f(meshView.vertexTangents[vertexIndex]));
      tangentSpace[vertexIndex].y = glm::floatBitsToUint(meshView.vertexTangents[vertexIndex].w);
    }
    else
    {
      tangentSpace[vertexIndex].x = nvmath::vec_to_oct32(nvmath::vec3f(1.f, 0.f, 0.f));
      tangentSpace[vertexIndex].y = glm::floatBitsToUint(1.f);
    }
  }
  if(settings.attribFlags & eMeshAttributeVertexDirectionBit)
  {
    if(vertexIndex < meshView.vertexDirections.size())
    {
      directions[4 * vertexIndex + 0] = glm::detail::toFloat16(meshView.vertexDirections[vertexIndex].x);
      directions[4 * vertexIndex + 1] = glm::detail::toFloat16(meshView.vertexDirections[vertexIndex].y);
      directions[4 * vertexIndex + 2] = glm::detail::toFloat16(meshView.vertexDirections[vertexIndex].z);
    }
    else
    {
      if(vertexIndex < meshView.vertexNormals.size())
      {
        directions[4 * vertexIndex + 0] = glm::detail::toFloat16(meshView.vertexNormals[vertexIndex].x);
        directions[4 * vertexIndex + 1] = glm::detail::toFloat16(meshView.vertexNormals[vertexIndex].y);
        directions[4 * vertexIndex + 2] = glm::detail::toFloat16(meshView.vertexNormals[vertexIndex].z);
      }
      else
      {
        directions[4 * vertexIndex + 0] = glm::detail::toFloat16(0.f);
        directions[4 * vertexIndex + 1] = glm::detail::toFloat16(0.f);
        directions[4 * vertexIndex + 2] = glm::detail::toFloat16(1.f);
      }
    }
  }
  if(settings.attribFlags & eMeshAttributeVertexDirectionBoundsBit)
  {
    if(vertexIndex < meshView.vertexDirectionBounds.size())
    {
      directionBounds[vertexIndex] = meshView.vertexDirectionBounds[vertexIndex];

      // Combine the global transform and the per-vertex transform
      directionBounds[vertexIndex].x += settings.directionBoundsBias * directionBounds[vertexIndex].y;
      directionBounds[vertexIndex].y *= settings.directionBoundsScale;
    }
    else
    {
      directionBounds[vertexIndex] = nvmath::vec2f(settings.directionBoundsBias, settings.directionBoundsScale);
    }
  }
  if(settings.attribFlags & eMeshAttributeVertexImportanceBit)
  {
    if(vertexIndex < meshView.vertexImportance.size())
    {
      importance[vertexIndex] = glm::detail::toFloat16(meshView.vertexImportance[vertexIndex]);
    }
    else
    {
      importance[vertexIndex] = glm::detail::toFloat16(0.f);
    }
  }
}

static_assert(sizeof(DeviceMeshTriangleAttributesVK) == sizeof(uint32_t),
              "DeviceMeshTriangleAttributesVK must be sizeof(uint32_t)");

static void copyTriangleAttributes(size_t triangleIndex, const MeshView& meshView, DeviceMeshSettings& settings, std::vector<uint32_t>& triAttributes)
{
  meshops::DeviceMeshTriangleAttributesVK a;

  if(settings.attribFlags & eMeshAttributeTriangleSubdivLevelsBit)
  {
    if(triangleIndex < meshView.triangleSubdivisionLevels.size())
    {
      a.subdLevel = meshView.triangleSubdivisionLevels[triangleIndex];
    }
    else
    {
      a.subdLevel = 0u;
    }
  }
  if(settings.attribFlags & eMeshAttributeTrianglePrimitiveFlagsBit)
  {
    if(triangleIndex < meshView.trianglePrimitiveFlags.size())
    {
      a.primitiveFlags = meshView.trianglePrimitiveFlags[triangleIndex];
    }
    else
    {
      a.primitiveFlags = 0u;
    }
  }
  triAttributes[triangleIndex] = *reinterpret_cast<uint32_t*>(&a);
}

template <class T>
T* meshops::DeviceMeshVKData::map(Context context, nvvk::Buffer b)
{
  auto& alloc = context->m_vk->m_resourceAllocator;
  if(b.buffer != VK_NULL_HANDLE)
  {
    return static_cast<T*>(alloc.map(b));
  }
  return nullptr;
}


VkAccelerationStructureKHR DeviceMeshVKData::getAccelerationStructure()
{
  return m_raytracingBuilder.getAccelerationStructure();
}

VkBuffer DeviceMeshVKData::getVertexImportanceBuffer()
{
  return m_vertexImportance.buffer;
}

VkBuffer DeviceMeshVKData::getVertexDirectionBoundsBuffer()
{
  return m_vertexDirectionBounds.buffer;
}

VkBuffer DeviceMeshVKData::getVertexDirectionsBuffer()
{
  return m_vertexDirections.buffer;
}

VkBuffer DeviceMeshVKData::getVertexTexcoordBuffer()
{
  return m_vertexTexcoord.buffer;
}

VkBuffer DeviceMeshVKData::getVertexTangentSpaceBuffer()
{
  return m_vertexTangentSpace.buffer;
}

VkBuffer DeviceMeshVKData::getVertexPositionNormalBuffer()
{
  return m_vertexPositionNormal.buffer;
}

VkBuffer DeviceMeshVKData::getTriangleAttributesBuffer()
{
  return m_triangleAttributes.buffer;
}

VkBuffer DeviceMeshVKData::getTriangleVertexIndexBuffer()
{
  return m_triangleVertexIndex.buffer;
}

static void readbackTriangles(uint32_t                  threadId,
                              uint32_t                  triangleBatchSize,
                              MutableMeshView&          meshView,
                              const DeviceMeshSettings& settings,
                              const nvmath::vec3ui*     hostTriangleVertexIndex,
                              const uint32_t*           hostTriangleAttributes)
{
  uint32_t start = threadId * triangleBatchSize;
  for(uint32_t i = 0; i < triangleBatchSize; i++)
  {
    uint32_t index = start + i;
    if(index >= meshView.triangleCount())
      break;

    if(settings.attribFlags & eMeshAttributeTriangleVerticesBit)
    {
      meshView.triangleVertices[index] = hostTriangleVertexIndex[index];
    }


    if(settings.attribFlags & (eMeshAttributeTriangleSubdivLevelsBit | eMeshAttributeTrianglePrimitiveFlagsBit))
    {
      DeviceMeshTriangleAttributesVK a;
      *reinterpret_cast<uint32_t*>(&a) = hostTriangleAttributes[index];
      if(settings.attribFlags & eMeshAttributeTriangleSubdivLevelsBit)
      {
        if(meshView.triangleSubdivisionLevels.size() > index)
        {
          meshView.triangleSubdivisionLevels[index] = a.subdLevel;
        }
      }
      if(settings.attribFlags & eMeshAttributeTrianglePrimitiveFlagsBit)
      {
        if(meshView.trianglePrimitiveFlags.size() > index)
        {
          meshView.trianglePrimitiveFlags[index] = a.primitiveFlags;
        }
      }
    }
  }
}

static void readbackVertexAttributes(uint32_t                  threadId,
                                     uint32_t                  vertexBatchSize,
                                     MutableMeshView&          meshView,
                                     const DeviceMeshSettings& settings,
                                     const nvmath::vec4f*      hostVertexPositionNormal,
                                     const nvmath::vec2f*      hostVertexTexcoord,
                                     const nvmath::vec2ui*     hostVertexTangentSpace,
                                     const uint16_t*           hostVertexDirections,
                                     const nvmath::vec2f*      hostVertexDirectionBounds,
                                     const uint16_t*           hostVertexImportance)
{
  uint32_t start = threadId * vertexBatchSize;
  for(uint32_t i = 0; i < vertexBatchSize; i++)
  {
    uint32_t index = start + i;
    if(index >= meshView.vertexCount())
      break;
    if(settings.attribFlags & eMeshAttributeVertexPositionBit)
    {
      if(meshView.vertexPositions.size() > index)
      {
        nvmath::vec3f pos               = nvmath::vec3f(hostVertexPositionNormal[index]);
        meshView.vertexPositions[index] = pos;
      }
    }
    nvmath::vec3f normal{};  // Can be removed once MICROSDK-343 is fixed.
    if(settings.attribFlags & eMeshAttributeVertexNormalBit)
    {
      if(meshView.vertexNormals.size() > index)
      {
        normal                        = nvmath::oct32_to_vec(glm::floatBitsToUint(hostVertexPositionNormal[index].w));
        meshView.vertexNormals[index] = normal;
      }
    }
    if(settings.attribFlags & eMeshAttributeVertexTexcoordBit)
    {
      if(meshView.vertexTexcoords0.size() > index)
      {
        meshView.vertexTexcoords0[index] = hostVertexTexcoord[index];
      }
    }
    if(settings.attribFlags & eMeshAttributeVertexTangentBit)
    {
      if(meshView.vertexTangents.size() > index)
      {
        nvmath::vec3f tangent          = nvmath::oct32_to_vec(hostVertexTangentSpace[index].x);
        float         sign_bit         = glm::uintBitsToFloat(hostVertexTangentSpace[index].y);
        meshView.vertexTangents[index] = nvmath::vec4f(tangent.x, tangent.y, tangent.z, sign_bit);
      }
    }
    if(settings.attribFlags & eMeshAttributeVertexDirectionBit)
    {
      if(meshView.vertexDirections.size() > index)
      {
        meshView.vertexDirections[index].x = glm::detail::toFloat32(hostVertexDirections[4 * index + 0]);
        meshView.vertexDirections[index].y = glm::detail::toFloat32(hostVertexDirections[4 * index + 1]);
        meshView.vertexDirections[index].z = glm::detail::toFloat32(hostVertexDirections[4 * index + 2]);
      }
    }
    if(settings.attribFlags & eMeshAttributeVertexDirectionBoundsBit)
    {
      if(meshView.vertexDirectionBounds.size() > index)
      {
        meshView.vertexDirectionBounds[index] = hostVertexDirectionBounds[index];
      }
    }
    if(settings.attribFlags & eMeshAttributeVertexImportanceBit)
    {
      if(meshView.vertexImportance.size() > index)
      {
        meshView.vertexImportance[index] = glm::detail::toFloat32(hostVertexImportance[index]);
      }
    }
  }
}

meshops::DeviceMeshSettings DeviceMeshVKData::getSettings() const
{
  meshops::DeviceMeshSettings settings{};

  if(m_raytracingBuilder.getAccelerationStructure() != VK_NULL_HANDLE)
    settings.usageFlags |= DeviceMeshUsageBlasBit;

  if(m_triangleVertexIndex.buffer != VK_NULL_HANDLE)
  {
    settings.attribFlags |= eMeshAttributeTriangleVerticesBit;
  }
  if(m_triangleAttributes.buffer != VK_NULL_HANDLE)
  {  // FIXME: on upload there is no guarantee both go together
    settings.attribFlags |= eMeshAttributeTrianglePrimitiveFlagsBit | eMeshAttributeTriangleSubdivLevelsBit;
  }
  if(m_vertexPositionNormal.buffer != VK_NULL_HANDLE)
  {
    settings.attribFlags |= eMeshAttributeVertexPositionBit | eMeshAttributeVertexNormalBit;
  }
  if(m_vertexTangentSpace.buffer != VK_NULL_HANDLE)
  {
    settings.attribFlags |= eMeshAttributeVertexTangentBit;
  }
  if(m_vertexTexcoord.buffer != VK_NULL_HANDLE)
  {
    settings.attribFlags |= eMeshAttributeVertexTexcoordBit;
  }
  if(m_vertexDirections.buffer != VK_NULL_HANDLE)
  {
    settings.attribFlags |= eMeshAttributeVertexDirectionBit;
  }
  if(m_vertexDirectionBounds.buffer != VK_NULL_HANDLE)
  {
    settings.attribFlags |= eMeshAttributeVertexDirectionBoundsBit;
  }
  if(m_vertexImportance.buffer != VK_NULL_HANDLE)
  {
    settings.attribFlags |= eMeshAttributeVertexImportanceBit;
  }
  return settings;
}

nvvk::Buffer DeviceMeshVKData::readback(Context context, VkCommandBuffer cmd, nvvk::Buffer buffer, size_t sizeInBytes)
{
  auto hostVisibleBuffer = context->m_vk->m_resourceAllocator.createBuffer(sizeInBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                                               | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  VkBufferCopy region{0, 0, sizeInBytes};
  vkCmdCopyBuffer(cmd, buffer.buffer, hostVisibleBuffer.buffer, 1, &region);
  return hostVisibleBuffer;
}

micromesh::Result meshops::DeviceMeshVKData::readbackBuffers(Context context, MutableMeshView& meshView, const DeviceMeshSettings& settings)
{
  nvvk::CommandPool cmdPool(context->m_vk->m_ptrs.context->m_device, context->m_vk->m_ptrs.context->m_queueC);
  VkCommandBuffer   cmd = cmdPool.createCommandBuffer();

  auto* staging = context->m_vk->m_resourceAllocator.getStaging();


  const nvmath::vec3ui* hostTriangleVertexIndex   = nullptr;
  const uint32_t*       hostTriangleAttributes    = nullptr;
  const nvmath::vec4f*  hostVertexPositionNormal  = nullptr;
  const nvmath::vec2ui* hostVertexTangentSpace    = nullptr;
  const nvmath::vec2f*  hostVertexTexcoord        = nullptr;
  const uint16_t*       hostVertexDirections      = nullptr;
  const nvmath::vec2f*  hostVertexDirectionBounds = nullptr;
  const uint16_t*       hostVertexImportance      = nullptr;

  if(settings.attribFlags & eMeshAttributeTriangleVerticesBit)
  {
    hostTriangleVertexIndex = static_cast<const nvmath::vec3ui*>(
        staging->cmdFromBuffer(cmd, m_triangleVertexIndex.buffer, 0, meshView.triangleCount() * sizeof(nvmath::vec3ui)));
  }

  if(settings.attribFlags & (eMeshAttributeTriangleSubdivLevelsBit | eMeshAttributeTrianglePrimitiveFlagsBit))
  {
    hostTriangleAttributes = static_cast<const uint32_t*>(
        staging->cmdFromBuffer(cmd, m_triangleAttributes.buffer, 0, meshView.triangleCount() * sizeof(uint32_t)));
  }

  if(settings.attribFlags & (eMeshAttributeVertexPositionBit | eMeshAttributeVertexNormalBit))
  {
    hostVertexPositionNormal = static_cast<const nvmath::vec4f*>(
        staging->cmdFromBuffer(cmd, m_vertexPositionNormal.buffer, 0, meshView.vertexCount() * sizeof(nvmath::vec4ui)));
  }
  if(settings.attribFlags & eMeshAttributeVertexTangentBit)
  {
    hostVertexTangentSpace = static_cast<const nvmath::vec2ui*>(
        staging->cmdFromBuffer(cmd, m_vertexTangentSpace.buffer, 0, meshView.vertexCount() * sizeof(nvmath::vec2ui)));
  }
  if(settings.attribFlags & eMeshAttributeVertexTexcoordBit)
  {
    hostVertexTexcoord = static_cast<const nvmath::vec2f*>(
        staging->cmdFromBuffer(cmd, m_vertexTexcoord.buffer, 0, meshView.vertexCount() * sizeof(nvmath::vec2f)));
  }
  if(settings.attribFlags & eMeshAttributeVertexDirectionBit)
  {
    hostVertexDirections = static_cast<const uint16_t*>(
        staging->cmdFromBuffer(cmd, m_vertexDirections.buffer, 0, 4 * meshView.vertexCount() * sizeof(uint16_t)));
  }
  if(settings.attribFlags & eMeshAttributeVertexDirectionBoundsBit)
  {
    hostVertexDirectionBounds = static_cast<const nvmath::vec2f*>(
        staging->cmdFromBuffer(cmd, m_vertexDirectionBounds.buffer, 0, meshView.vertexCount() * sizeof(nvmath::vec2f)));
  }
  if(settings.attribFlags & eMeshAttributeVertexImportanceBit)
  {
    hostVertexImportance = static_cast<const uint16_t*>(
        staging->cmdFromBuffer(cmd, m_vertexImportance.buffer, 0, meshView.vertexCount() * sizeof(uint16_t)));
  }
  staging->finalizeResources();
  cmdPool.submitAndWait(cmd);

  uint32_t maxUsableVertexThreads = uint32_t((meshView.vertexCount() + MIN_ITEMS_PER_THREAD - 1) / MIN_ITEMS_PER_THREAD);
  uint32_t maxVertexThreads       = std::min(maxUsableVertexThreads, context->m_config.threadCount);
  uint32_t maxUsableTriangleThreads = uint32_t((meshView.triangleCount() + MIN_ITEMS_PER_THREAD - 1) / MIN_ITEMS_PER_THREAD);
  uint32_t maxTriangleThreads       = std::min(maxUsableTriangleThreads, context->m_config.threadCount);
  uint32_t vertexBatchSize          = uint32_t((meshView.vertexCount() + maxVertexThreads - 1) / maxVertexThreads);
  uint32_t triangleBatchSize = uint32_t((meshView.triangleCount() + maxTriangleThreads - 1) / maxTriangleThreads);


  if(maxVertexThreads > 1)
  {
    std::vector<std::thread> vertexThreads;
    vertexThreads.reserve(maxVertexThreads);
    for(uint32_t threadId = 0; threadId < maxVertexThreads; threadId++)
    {
      vertexThreads.emplace_back([&, threadId] {
        readbackVertexAttributes(threadId, vertexBatchSize, meshView, settings, hostVertexPositionNormal, hostVertexTexcoord,
                                 hostVertexTangentSpace, hostVertexDirections, hostVertexDirectionBounds, hostVertexImportance);
      });
    }
    for(auto& t : vertexThreads)
    {
      t.join();
    }
  }
  else
  {
    readbackVertexAttributes(0, uint32_t(meshView.vertexCount()), meshView, settings, hostVertexPositionNormal, hostVertexTexcoord,
                             hostVertexTangentSpace, hostVertexDirections, hostVertexDirectionBounds, hostVertexImportance);
  }


  if(maxTriangleThreads > 1)
  {
    std::vector<std::thread> triangleThreads;
    triangleThreads.reserve(maxTriangleThreads);
    for(uint32_t threadId = 0; threadId < maxTriangleThreads; threadId++)
    {
      triangleThreads.emplace_back([&, threadId] {
        readbackTriangles(threadId, triangleBatchSize, meshView, settings, hostTriangleVertexIndex, hostTriangleAttributes);
      });
    }
    for(auto& t : triangleThreads)
    {
      t.join();
    }
  }
  else
  {
    readbackTriangles(0, uint32_t(meshView.triangleCount()), meshView, settings, hostTriangleVertexIndex, hostTriangleAttributes);
  }
  staging->releaseResources();

  return micromesh::Result::eSuccess;
}

void DeviceMeshVKData::unmapAndDestroy(Context context, nvvk::Buffer b)
{
  auto& alloc = context->m_vk->m_resourceAllocator;
  if(b.buffer != VK_NULL_HANDLE)
  {
    alloc.unmap(b);
    alloc.destroy(b);
  }
}

micromesh::Result DeviceMeshVKData::uploadBuffers(Context context, const MeshView& meshView, DeviceMeshSettings& settings)
{
  nvvk::CommandPool cmdPool(context->m_vk->m_ptrs.context->m_device, context->m_vk->m_ptrs.context->m_queueT);
  VkCommandBuffer   cmd = cmdPool.createCommandBuffer();
  if(settings.attribFlags & eMeshAttributeTriangleVerticesBit)
  {
    createBuffer(context, cmd, meshView.triangleVertices, m_triangleVertexIndex,
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    if(m_triangleVertexIndex.buffer == VK_NULL_HANDLE)
    {
      return micromesh::Result::eFailure;
    }
  }

  std::vector<nvmath::vec4f>      positionNormal;
  std::vector<nvmath::vec2ui>     tangentSpace;
  std::vector<nvmath::vec2f>      texCoord;
  std::vector<glm::detail::hdata> directions;
  std::vector<nvmath::vec2f>      directionBounds;
  std::vector<uint16_t>           importance;
  if(settings.attribFlags & eMeshAttributeVertexPositionBit)
  {
    positionNormal.resize(meshView.vertexCount());
  }

  if(settings.attribFlags & eMeshAttributeVertexTangentBit)
  {
    tangentSpace.resize(meshView.vertexCount());
  }
  // n x fp32 x 2
  // FIXME: to change when/if meshview supports multiple UVs
  if(settings.attribFlags & eMeshAttributeVertexTexcoordBit)
  {
    texCoord.resize(meshView.vertexCount());
  }

  // fp16 x 4
  if(settings.attribFlags & eMeshAttributeVertexDirectionBit)
  {
    directions.resize(meshView.vertexCount() * 4);
  }

  // fp32 x 2
  if(settings.attribFlags & eMeshAttributeVertexDirectionBoundsBit)
  {
    directionBounds.resize(meshView.vertexCount());
  }

  // fp16
  if(settings.attribFlags & eMeshAttributeVertexImportanceBit)
  {
    importance.resize(meshView.vertexCount());
  }
  nvh::parallel_batches(
      meshView.vertexCount(),
      [&](uint64_t vertIdx) {
        copyVertexAttributes(vertIdx, meshView, settings, positionNormal, texCoord, tangentSpace, directions,
                             directionBounds, importance);
      },
      context->m_config.threadCount

  );
  if(settings.attribFlags & eMeshAttributeVertexPositionBit)
  {
    createBuffer(context, cmd, positionNormal, m_vertexPositionNormal,
                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    if(m_vertexPositionNormal.buffer == VK_NULL_HANDLE)
    {
      return micromesh::Result::eFailure;
    }
  }

  if(settings.attribFlags & eMeshAttributeVertexTangentBit)
  {
    createBuffer(context, cmd, tangentSpace, m_vertexTangentSpace, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    if(m_vertexTangentSpace.buffer == VK_NULL_HANDLE)
    {
      return micromesh::Result::eFailure;
    }
  }
  // n x fp32 x 2
  // FIXME: support multiple UVs
  if(settings.attribFlags & eMeshAttributeVertexTexcoordBit)
  {
    createBuffer(context, cmd, texCoord, m_vertexTexcoord, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    if(m_vertexTexcoord.buffer == VK_NULL_HANDLE)
    {
      return micromesh::Result::eFailure;
    }
  }

  // fp16 x 4
  if(settings.attribFlags & eMeshAttributeVertexDirectionBit)
  {
    createBuffer(context, cmd, directions, m_vertexDirections, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    if(m_vertexDirections.buffer == VK_NULL_HANDLE)
    {
      return micromesh::Result::eFailure;
    }
  }

  // fp32 x 2
  if(settings.attribFlags & eMeshAttributeVertexDirectionBoundsBit)
  {
    createBuffer(context, cmd, directionBounds, m_vertexDirectionBounds, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    if(m_vertexDirectionBounds.buffer == VK_NULL_HANDLE)
    {
      return micromesh::Result::eFailure;
    }
  }

  // fp16
  if(settings.attribFlags & eMeshAttributeVertexImportanceBit)
  {
    createBuffer(context, cmd, importance, m_vertexImportance, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    if(m_vertexImportance.buffer == VK_NULL_HANDLE)
    {
      return micromesh::Result::eFailure;
    }
  }
  if((settings.attribFlags & eMeshAttributeTriangleSubdivLevelsBit) || (settings.attribFlags & eMeshAttributeTrianglePrimitiveFlagsBit))
  {
    std::vector<uint32_t> triAttributes(meshView.triangleCount());

    nvh::parallel_batches(
        meshView.triangleCount(),
        [&](uint64_t triangleIdx) { copyTriangleAttributes(triangleIdx, meshView, settings, triAttributes); },
        context->m_config.threadCount);

    createBuffer(context, cmd, triAttributes, m_triangleAttributes);
    if(m_triangleAttributes.buffer == VK_NULL_HANDLE)
    {
      return micromesh::Result::eFailure;
    }
  }
  cmdPool.submitAndWait(cmd);
  return micromesh::Result::eSuccess;
}

micromesh::Result DeviceMeshVKData::allocateVertexImportance(Context context, const MeshView& meshView)
{
  m_vertexImportance = context->m_vk->m_resourceAllocator.createBuffer(meshView.vertexCount() * sizeof(uint16_t),
                                                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                                                           | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                                           | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  return (m_vertexImportance.buffer != VK_NULL_HANDLE) ? micromesh::Result::eSuccess : micromesh::Result::eFailure;
}

micromesh::Result DeviceMeshVKData::createAccelerationStructure(Context context, const MeshView& meshView)
{

  micromesh::Result r = micromesh::Result::eFailure;
  if(context->m_vk == nullptr)
    return r;

  nvvk::RaytracingBuilderKHR::BlasInput blasInput;


  auto vertexAddress = nvvk::getBufferDeviceAddress(context->m_vk->m_ptrs.context->m_device, m_vertexPositionNormal.buffer);
  auto indexAddress = nvvk::getBufferDeviceAddress(context->m_vk->m_ptrs.context->m_device, m_triangleVertexIndex.buffer);

  // Describe buffer as array of VertexObj.
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
  triangles.vertexFormat             = VK_FORMAT_R32G32B32A32_SFLOAT;  // vec3 vertex position data.
  triangles.vertexData.deviceAddress = vertexAddress;
  triangles.vertexStride             = sizeof(nvmath::vec4ui);
  triangles.indexType                = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress  = indexAddress;
  triangles.maxVertex                = uint32_t(meshView.vertexCount());
  //triangles.transformData; // Identity

  // Identify the above data as containing opaque triangles.
  VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  asGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  //asGeom.flags              = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
  asGeom.geometry.triangles = triangles;

  VkAccelerationStructureBuildRangeInfoKHR offset{};
  offset.firstVertex     = 0;
  offset.primitiveCount  = uint32_t(meshView.triangleCount());
  offset.primitiveOffset = 0;
  offset.transformOffset = 0;

  // Our blas is made from only one geometry, but could be made of many geometries
  blasInput.asGeometry.emplace_back(asGeom);
  blasInput.asBuildOffsetInfo.emplace_back(offset);

  m_raytracingBuilder.setup(context->m_vk->m_ptrs.context->m_device, context->m_vk->m_ptrs.resAllocator,
                            context->m_vk->m_ptrs.context->m_queueC);

  m_raytracingBuilder.buildBlas({blasInput}, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);


  VkAccelerationStructureInstanceKHR tlasInstance;


  VkGeometryInstanceFlagsKHR flags{};
  flags |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;                  // All opaque (faster)
  flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;  // double sided


  tlasInstance.transform           = nvvk::toTransformMatrixKHR(nvmath::mat4f_id);  // Position of the instance
  tlasInstance.instanceCustomIndex = 0;                                             // gl_InstanceCustomIndexEXT
  tlasInstance.accelerationStructureReference         = m_raytracingBuilder.getBlasDeviceAddress(0);
  tlasInstance.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
  tlasInstance.flags                                  = flags & 0xFF;
  tlasInstance.mask                                   = 0xFF;


  m_raytracingBuilder.buildTlas({tlasInstance}, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

  return micromesh::Result::eSuccess;
}

micromesh::Result DeviceMeshVKData::destroyDeviceData(Context context)
{
  micromesh::Result r = micromesh::Result::eFailure;
  if(context->m_vk == nullptr)
    return r;

  context->m_vk->m_resourceAllocator.destroy(m_triangleVertexIndex);
  context->m_vk->m_resourceAllocator.destroy(m_vertexPositionNormal);
  context->m_vk->m_resourceAllocator.destroy(m_vertexTexcoord);
  context->m_vk->m_resourceAllocator.destroy(m_vertexTangentSpace);
  context->m_vk->m_resourceAllocator.destroy(m_vertexDirections);
  context->m_vk->m_resourceAllocator.destroy(m_vertexDirectionBounds);
  context->m_vk->m_resourceAllocator.destroy(m_triangleAttributes);

  context->m_vk->m_resourceAllocator.destroy(m_vertexImportance);

  m_raytracingBuilder.destroy();

  return micromesh::Result::eSuccess;
}

micromesh::Result DeviceMeshVKData::createDeviceData(Context context, const MeshView& meshView, DeviceMeshSettings& settings)
{
  micromesh::Result r = micromesh::Result::eFailure;
  if(context->m_vk == nullptr)
    return r;

  r = destroyDeviceData(context);
  if(r != micromesh::Result::eSuccess)
  {
    return r;
  }
  r = uploadBuffers(context, meshView, settings);
  if(r != micromesh::Result::eSuccess)
  {
    return r;
  }

  if(settings.usageFlags & DeviceMeshUsageBlasBit)
  {
    r = createAccelerationStructure(context, meshView);
    if(r != micromesh::Result::eSuccess)
    {
      return r;
    }
  }

  return r;
}

micromesh::Result DeviceMesh_c::initializeMeshVk(Context context, meshops::MeshAttributeFlags sourceAttribFlags)
{
  if(context->m_vk != nullptr)
  {
    m_vk.usageFlags        = getSettings().usageFlags;
    m_vk.sourceAttribFlags = sourceAttribFlags;
    m_vk.deviceAttribFlags = getSettings().attribFlags;

    m_vk.triangleVertexIndexBuffer   = {m_vkData.getTriangleVertexIndexBuffer(), 0, VK_WHOLE_SIZE};
    m_vk.triangleAttributesBuffer    = {m_vkData.getTriangleAttributesBuffer(), 0, VK_WHOLE_SIZE};
    m_vk.vertexPositionNormalBuffer  = {m_vkData.getVertexPositionNormalBuffer(), 0, VK_WHOLE_SIZE};
    m_vk.vertexTangentSpaceBuffer    = {m_vkData.getVertexTangentSpaceBuffer(), 0, VK_WHOLE_SIZE};
    m_vk.vertexTexcoordBuffer        = {m_vkData.getVertexTexcoordBuffer(), 0, VK_WHOLE_SIZE};
    m_vk.vertexDirectionsBuffer      = {m_vkData.getVertexDirectionsBuffer(), 0, VK_WHOLE_SIZE};
    m_vk.vertexDirectionBoundsBuffer = {m_vkData.getVertexDirectionBoundsBuffer(), 0, VK_WHOLE_SIZE};
    m_vk.vertexImportanceBuffer      = {m_vkData.getVertexImportanceBuffer(), 0, VK_WHOLE_SIZE};

    m_vk.vertexTexcoordCount = ((m_vkData.getVertexTexcoordBuffer() == VK_NULL_HANDLE) ? 0 : 1);

    m_vk.blas = m_vkData.getAccelerationStructure();

    return micromesh::Result::eSuccess;
  }
  return micromesh::Result::eFailure;
}

micromesh::Result DeviceMesh_c::destroy(Context context)
{
  if(context->m_vk != nullptr)
  {
    return m_vkData.destroyDeviceData(context);
  }
  return micromesh::Result::eFailure;
}

micromesh::Result DeviceMesh_c::readback(Context context, MutableMeshView& meshView)
{
  return m_vkData.readbackBuffers(context, meshView, getSettings());
}
micromesh::Result DeviceMesh_c::readback(Context context, MutableMeshView& meshView, DeviceMeshSettings attributes)
{
  return m_vkData.readbackBuffers(context, meshView, attributes);
}

micromesh::Result DeviceMesh_c::create(Context context, const MeshView& meshView, DeviceMeshSettings& settings)
{
  m_settings          = settings;
  micromesh::Result r = micromesh::Result::eFailure;
  if(context->m_vk != nullptr)
  {

    r = m_vkData.createDeviceData(context, meshView, settings);

    if(r != micromesh::Result::eSuccess)
      return r;
    r = initializeMeshVk(context, meshView.getMeshAttributeFlags());
    if(r != micromesh::Result::eSuccess)
      return r;
  }
  return r;
}

meshops::DeviceMeshVK* DeviceMesh_c::getDeviceMeshVk()
{
  return &m_vk;
}

const meshops::DeviceMeshSettings& DeviceMesh_c::getSettings() const
{
  return m_settings;
}
