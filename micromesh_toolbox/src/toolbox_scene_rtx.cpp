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


#include "toolbox_scene_rtx.hpp"
#include "heightmap_rtx.h"
#include "meshops/meshops_vk.h"
#include "nvh/timesampler.hpp"
#include "vulkan/vulkan_core.h"
#include <nvvk/error_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>
#include <glm/detail/type_half.hpp>
#include <settings.hpp>

ToolboxSceneRtx::ToolboxSceneRtx(nvvk::Context* ctx, nvvkhl::AllocVma* alloc, uint32_t queueFamilyIndex)
    : m_ctx(ctx)
    , m_alloc(alloc)
{
  // Requesting ray tracing properties
  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  prop2.pNext = &m_rtProperties;
  vkGetPhysicalDeviceProperties2(m_ctx->m_physicalDevice, &prop2);

  // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
  m_rtBuilder.setup(m_ctx->m_device, m_alloc, queueFamilyIndex);
}

ToolboxSceneRtx::~ToolboxSceneRtx()
{
  // The destructor wasn't called
  assert(m_rtBuilder.getAccelerationStructure() == VK_NULL_HANDLE);
}

//--------------------------------------------------------------------------------------------------
// Create the acceleration structures for the `ToolScene`
//
void ToolboxSceneRtx::create(const ViewerSettings&                             settings,
                             const std::unique_ptr<micromesh_tool::ToolScene>& scene,
                             const std::unique_ptr<ToolboxSceneVk>&            sceneVK,
                             bool                                              hasMicroMesh,
                             VkBuildAccelerationStructureFlagsKHR flags /*= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR*/)
{
  destroy();  // Make sure not to leave allocated buffers

  createBottomLevelAS(settings, scene, sceneVK, flags, hasMicroMesh, settings.geometryView.baked);
  createTopLevelAS(scene, flags);
}

//--------------------------------------------------------------------------------------------------
//
//
void ToolboxSceneRtx::destroy()
{
  m_rtBuilder.destroy();

  // TODO: raii and stop calling destroy() on objects that aren't initialized
  for(auto& map : m_heightmaps)
  {
    hrtxDestroyMap(map);
  }
  m_heightmaps.clear();
  for(auto& buffer : m_heightmapDirections)
  {
    m_alloc->destroy(buffer);
  }
  m_heightmapDirections.clear();
  if(m_heightmapPipeline)
  {
    hrtxDestroyPipeline(m_heightmapPipeline);
    m_heightmapPipeline = nullptr;
  }
}

//--------------------------------------------------------------------------------------------------
// Converting a PrimitiveMesh as input for BLAS
//
nvvk::RaytracingBuilderKHR::BlasInput ToolboxSceneRtx::primitiveToGeometry(const micromesh_tool::ToolMesh& mesh,
                                                                           VkDeviceAddress vertexAddress,
                                                                           VkDeviceAddress indexAddress)
{
  uint32_t max_prim_count = static_cast<uint32_t>(mesh.view().triangleCount());

  // Describe buffer as array of VertexObj.
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
  triangles.vertexFormat             = VK_FORMAT_R32G32B32A32_SFLOAT;  // vec3 vertex position data.
  triangles.vertexData.deviceAddress = vertexAddress;
  triangles.vertexStride             = sizeof(nvmath::vec4f);
  triangles.indexType                = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress  = indexAddress;
  triangles.maxVertex                = static_cast<uint32_t>(mesh.view().vertexCount());
  //triangles.transformData; // Identity

  // Identify the above data as containing opaque triangles.
  VkAccelerationStructureGeometryKHR as_geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  as_geom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  as_geom.flags              = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
  as_geom.geometry.triangles = triangles;

  VkAccelerationStructureBuildRangeInfoKHR offset{};
  offset.firstVertex     = 0;
  offset.primitiveCount  = max_prim_count;
  offset.primitiveOffset = 0;
  offset.transformOffset = 0;

  // Our blas is made from only one geometry, but could be made of many geometries
  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(as_geom);
  input.asBuildOffsetInfo.emplace_back(offset);

  return input;
}

//--------------------------------------------------------------------------------------------------
// Creating the Bottom Level Acceleration Structure for all `ToolMesh` in the `ToolScene`
// -
void ToolboxSceneRtx::createBottomLevelAS(const ViewerSettings&                             settings,
                                          const std::unique_ptr<micromesh_tool::ToolScene>& scene,
                                          const std::unique_ptr<ToolboxSceneVk>&            sceneVK,
                                          VkBuildAccelerationStructureFlagsKHR              flags,
                                          bool                                              hasMicroMesh,
                                          bool                                              useMicroMesh)
{
  nvh::ScopedTimer _st("- Create BLAS");

  const std::vector<std::unique_ptr<micromesh_tool::ToolMesh>>& meshes = scene->meshes();

  // BLAS - Storing each primitive in a geometry
  std::vector<nvvk::RaytracingBuilderKHR::BlasInput> all_blas;
  all_blas.reserve(meshes.size());

  // #MICROMESH
  std::vector<VkAccelerationStructureTrianglesDisplacementMicromapNV> geometry_displacements;  // hold data until BLAS is created
  geometry_displacements.reserve(meshes.size());


  for(uint32_t p_idx = 0; p_idx < meshes.size(); p_idx++)
  {
    const meshops::DeviceMesh& device_mesh = sceneVK->deviceMesh(p_idx);
    meshops::DeviceMeshVK*     device_vk   = meshops::meshopsDeviceMeshGetVK(device_mesh);


    auto vertex_address = nvvk::getBufferDeviceAddress(m_ctx->m_device, device_vk->vertexPositionNormalBuffer.buffer);
    auto index_address  = nvvk::getBufferDeviceAddress(m_ctx->m_device, device_vk->triangleVertexIndexBuffer.buffer);

    const auto& mesh = meshes[p_idx];

    auto geo = primitiveToGeometry(*mesh, vertex_address, index_address);

    // Add micromap information to the BLAS if it exists
    float bias, scale;
    int   heightmapImageIndex;
    if(useMicroMesh && (mesh->relations().bary != -1 && mesh->relations().group != -1))
    {
      const std::unique_ptr<DeviceBary>& deviceBary = sceneVK->barys()[mesh->relations().bary];
      const DeviceMicromap&              micromap   = deviceBary->micromaps()[mesh->relations().group];
      if(micromap.raytrace())
      {
        // micromap for this mesh
        const VkDeviceAddress primitive_flags_addr =
            nvvk::getBufferDeviceAddress(m_ctx->m_device, device_vk->triangleAttributesBuffer.buffer);
        const VkDeviceAddress directions_addr =
            nvvk::getBufferDeviceAddress(m_ctx->m_device, device_vk->vertexDirectionsBuffer.buffer);
        const VkDeviceAddress direction_bounds_addr =
            nvvk::getBufferDeviceAddress(m_ctx->m_device, device_vk->vertexDirectionBoundsBuffer.buffer);

        // #MICROMESH
        VkAccelerationStructureTrianglesDisplacementMicromapNV displacement{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_DISPLACEMENT_MICROMAP_NV};
        displacement.micromap         = micromap.raytrace()->micromap;
        displacement.usageCountsCount = static_cast<uint32_t>(micromap.raytrace()->usages.size());
        displacement.pUsageCounts     = micromap.raytrace()->usages.data();

        assert(directions_addr);
        {
          displacement.displacementVectorBuffer.deviceAddress = directions_addr;
          displacement.displacementVectorStride               = sizeof(glm::detail::hdata) * 4;
          displacement.displacementVectorFormat               = VK_FORMAT_R16G16B16A16_SFLOAT;
        }

        if(direction_bounds_addr != 0U)  // optional
        {
          displacement.displacementBiasAndScaleBuffer.deviceAddress = direction_bounds_addr;
          displacement.displacementBiasAndScaleStride               = sizeof(nvmath::vec2f);
          displacement.displacementBiasAndScaleFormat               = VK_FORMAT_R32G32_SFLOAT;
        }

        if(primitive_flags_addr != 0U)  // optional
        {
          displacement.displacedMicromapPrimitiveFlags.deviceAddress =
              primitive_flags_addr + offsetof(meshops::DeviceMeshTriangleAttributesVK, primitiveFlags);
          displacement.displacedMicromapPrimitiveFlagsStride = sizeof(meshops::DeviceMeshTriangleAttributesVK);
        }

        // Adding micromap
        geometry_displacements.emplace_back(displacement);
        geo.asGeometry[0].geometry.triangles.pNext = &geometry_displacements.back();
      }
    }
    else if(hasMicroMesh && scene->getHeightmap(mesh->relations().material, bias, scale, heightmapImageIndex))
    {
      nvvk::CommandPool cmd_pool(m_ctx->m_device, m_ctx->m_queueGCT.familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                 m_ctx->m_queueGCT.queue);
      VkCommandBuffer   cmd = cmd_pool.createCommandBuffer();
      if(!m_heightmapPipeline)
      {
        HrtxAllocatorCallbacks allocatorCallbacks{
            [](const VkBufferCreateInfo bufferCreateInfo, const VkMemoryPropertyFlags memoryProperties, void* userPtr) {
              auto alloc  = reinterpret_cast<nvvkhl::AllocVma*>(userPtr);
              auto result = new nvvk::Buffer();
              *result     = alloc->createBuffer(bufferCreateInfo, memoryProperties);
              return &result->buffer;  // return pointer to member
            },
            [](VkBuffer* bufferPtr, void* userPtr) {
              auto alloc = reinterpret_cast<nvvkhl::AllocVma*>(userPtr);
              // reconstruct from pointer to member
              auto nvvkBuffer = reinterpret_cast<nvvk::Buffer*>(reinterpret_cast<char*>(bufferPtr)
                                                                - offsetof(nvvk::Buffer, nvvk::Buffer::buffer));
              alloc->destroy(*nvvkBuffer);
              delete nvvkBuffer;
            },
            m_alloc,
        };
        HrtxPipelineCreate hrtxPipelineCreate{
            m_ctx->m_physicalDevice, m_ctx->m_device, allocatorCallbacks, VK_NULL_HANDLE, nullptr, nullptr, VK_NULL_HANDLE, [](VkResult result) {
              nvvk::checkResult(result, "HRTX");
            }};
        if(hrtxCreatePipeline(cmd, &hrtxPipelineCreate, &m_heightmapPipeline) != VK_SUCCESS)
        {
          LOGW("Warning: Failed to create HrtxPipeline. Raytracing heightmaps will not work.\n");
        }
      }

      // Convert direction vectors to fp16
      using hvec4 = glm::vec<4, glm::detail::hdata>;
      std::vector<hvec4> directionsHalfVec4(mesh->view().vertexNormals.size());
      std::transform(mesh->view().vertexNormals.begin(), mesh->view().vertexNormals.end(), directionsHalfVec4.begin(),
                     [](nvmath::vec3f normal) {
                       return hvec4{glm::detail::toFloat16(normal.x), glm::detail::toFloat16(normal.y),
                                    glm::detail::toFloat16(normal.z), glm::detail::toFloat16(0.0f)};
                     });
      m_heightmapDirections.push_back(m_alloc->createBuffer(cmd, directionsHalfVec4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));

      // Barrier for displacement directions buffer. Texture coordinates and
      // heightmap are created synchronously so no barrier is needed.
      {
        VkMemoryBarrier2 memoryBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                          VK_ACCESS_2_TRANSFER_WRITE_BIT};
        hrtxBarrierFlags(nullptr, nullptr, &memoryBarrier.dstStageMask, &memoryBarrier.dstAccessMask, nullptr);
        VkDependencyInfo depencencyInfo = {VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 1, &memoryBarrier};
        vkCmdPipelineBarrier2(cmd, &depencencyInfo);
      }

      // Convert image index to sceneVK->textures() index
      size_t heightmapTextureIndex = 0;
      for(size_t i = 0; i < scene->textures().size(); ++i)
      {
        if(scene->textures()[i].source == heightmapImageIndex)
        {
          heightmapTextureIndex = i;
          break;
        }
      }

      const uint32_t subdivLevel = settings.heightmapRTXSubdivLevel;
      HrtxMapCreate  mapCreate{
          &geo.asGeometry[0].geometry.triangles,
          geo.asBuildOffsetInfo[0].primitiveCount,
          nvvk::getBufferDeviceAddress(m_ctx->m_device, device_vk->vertexTexcoordBuffer.buffer),
          VK_FORMAT_R32G32_SFLOAT,
          sizeof(float) * 2,
          nvvk::getBufferDeviceAddress(m_ctx->m_device, m_heightmapDirections.back().buffer),
          VK_FORMAT_R16G16B16A16_SFLOAT,  // possible issue with VK_FORMAT_R32G32B32_SFLOAT?
          sizeof(uint16_t) * 4,
          sceneVK->textures()[heightmapTextureIndex].descriptor,
          bias * settings.heightmapScale + settings.heightmapOffset,
          scale * settings.heightmapScale,
          subdivLevel,
      };

      HrtxMap hrtxMap;
      if(hrtxCmdCreateMap(cmd, m_heightmapPipeline, &mapCreate, &hrtxMap) == VK_SUCCESS)
      {
        m_heightmaps.push_back(hrtxMap);
        geometry_displacements.emplace_back(hrtxMapDesc(m_heightmaps.back()));
        geo.asGeometry[0].geometry.triangles.pNext = &geometry_displacements.back();
      }
      else
      {
        LOGW("Warning: Failed to create HrtxMap for mesh %u. Raytracing heightmaps will not work.\n", p_idx);
      }

      cmd_pool.submitAndWait(cmd);
      m_alloc->finalizeAndReleaseStaging();  // Make sure there are no pending staging buffers and clear them up
    }

    all_blas.push_back({geo});
  }

  m_rtBuilder.buildBlas(all_blas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                      | VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
// Creating the Top Level Acceleration Structure for `ToolScene`
//
void ToolboxSceneRtx::createTopLevelAS(const std::unique_ptr<micromesh_tool::ToolScene>& scene, VkBuildAccelerationStructureFlagsKHR flags)
{
  nvh::ScopedTimer _st("- Create TLAS");

  meshops::ArrayView<micromesh_tool::ToolScene::Instance> instances = scene->instances();

  std::vector<VkAccelerationStructureInstanceKHR> tlas;
  tlas.reserve(instances.size());
  for(const auto& instance : instances)
  {
    VkGeometryInstanceFlagsKHR flags{};
    int                        materialIndex = scene->meshes()[instance.mesh]->relations().material;
    const tinygltf::Material&  material      = scene->material(materialIndex);
    // Always opaque, no need to use anyhit (faster)
    if(material.alphaMode == "OPAQUE"
       || (material.pbrMetallicRoughness.baseColorFactor[3] == 1.0F && material.pbrMetallicRoughness.baseColorTexture.index == -1))
    {
      flags |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    }

    // Need to skip the cull flag in traceray_rtx for double sided materials
    if(material.doubleSided == 1)
    {
      flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    }

    VkAccelerationStructureInstanceKHR ray_inst{};
    ray_inst.transform           = nvvk::toTransformMatrixKHR(instance.worldMatrix);  // Position of the instance
    ray_inst.instanceCustomIndex = instance.mesh & 0x00FFFFFF;                        // gl_InstanceCustomIndexEXT
    ray_inst.accelerationStructureReference         = m_rtBuilder.getBlasDeviceAddress(instance.mesh);
    ray_inst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
    ray_inst.flags                                  = flags & 0xFF;
    ray_inst.mask                                   = 0xFF;
    tlas.emplace_back(ray_inst);
  }

  VkBuildAccelerationStructureFlagsKHR tlasFlags =
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  m_rtBuilder.buildTlas(tlas, tlasFlags);
}
