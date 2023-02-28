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
#include "meshops/meshops_vk.h"
#include "nvh/timesampler.hpp"

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
void ToolboxSceneRtx::create(const std::unique_ptr<micromesh_tool::ToolScene>& scene,
                             const std::unique_ptr<ToolboxSceneVk>&            sceneVK,
                             bool                                              useMicroMesh,
                             VkBuildAccelerationStructureFlagsKHR flags /*= VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR*/)
{
  destroy();  // Make sure not to leave allocated buffers

  createBottomLevelAS(scene, sceneVK, flags, useMicroMesh);
  createTopLevelAS(scene, flags, useMicroMesh);
}

//--------------------------------------------------------------------------------------------------
//
//
void ToolboxSceneRtx::destroy()
{
  m_rtBuilder.destroy();
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
void ToolboxSceneRtx::createBottomLevelAS(const std::unique_ptr<micromesh_tool::ToolScene>& scene,
                                          const std::unique_ptr<ToolboxSceneVk>&            sceneVK,
                                          VkBuildAccelerationStructureFlagsKHR              flags,
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

    all_blas.push_back({geo});
  }

  m_rtBuilder.buildBlas(all_blas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                      | VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
// Creating the Top Level Acceleration Structure for `ToolScene`
//
void ToolboxSceneRtx::createTopLevelAS(const std::unique_ptr<micromesh_tool::ToolScene>& scene,
                                       VkBuildAccelerationStructureFlagsKHR              flags,
                                       bool                                              useMicroMesh)
{
  nvh::ScopedTimer _st("- Create TLAS");

  const std::vector<micromesh_tool::ToolScene::PrimitiveInstance>& prim_instances = scene->getPrimitiveInstances();

  std::vector<VkAccelerationStructureInstanceKHR> tlas;
  tlas.reserve(prim_instances.size());
  for(const auto& prim_inst : prim_instances)
  {
    VkGeometryInstanceFlagsKHR flags{};

    tinygltf::Material mat = {};
    if(prim_inst.material >= 0 && static_cast<size_t>(prim_inst.material) < scene->model().materials.size())
      mat = scene->model().materials[prim_inst.material];
    // Always opaque, no need to use anyhit (faster)
    if(mat.alphaMode == "OPAQUE"
       || (mat.pbrMetallicRoughness.baseColorFactor[3] == 1.0F && mat.pbrMetallicRoughness.baseColorTexture.index == -1))
    {
      flags |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    }

    // Need to skip the cull flag in traceray_rtx for double sided materials
    if(mat.doubleSided == 1)
    {
      flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    }

    VkAccelerationStructureInstanceKHR ray_inst{};
    ray_inst.transform           = nvvk::toTransformMatrixKHR(prim_inst.worldMatrix);  // Position of the instance
    ray_inst.instanceCustomIndex = prim_inst.primMeshRef & 0x00FFFFFF;                 // gl_InstanceCustomIndexEXT
    ray_inst.accelerationStructureReference         = m_rtBuilder.getBlasDeviceAddress(prim_inst.primMeshRef);
    ray_inst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
    ray_inst.flags                                  = flags & 0xFF;
    ray_inst.mask                                   = 0xFF;
    tlas.emplace_back(ray_inst);
  }

  VkBuildAccelerationStructureFlagsKHR tlasFlags =
      VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  if(useMicroMesh)
    tlasFlags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DISPLACEMENT_MICROMAP_INSTANCE_NV;
  m_rtBuilder.buildTlas(tlas, tlasFlags);
}
