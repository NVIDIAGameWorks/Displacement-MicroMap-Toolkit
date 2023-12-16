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

#include <cinttypes>
#include "meshops/meshops_array_view.h"
#include "micromap/microdisp_shim.hpp"
#include "tiny_gltf.h"
#include "toolbox_scene_vk.hpp"
#include "meshops/meshops_vk.h"
#include "nvh/parallel_work.hpp"
#include "nvh/timesampler.hpp"
#include "nvh/gltfscene.hpp"
#include "nvvk/images_vk.hpp"
#include "shaders/dh_scn_desc.h"
#include "shaders/device_host.h"
#include "vulkan/vulkan_core.h"
#include "vulkan_mutex.h"
#include "micromesh/micromesh_utils.h"

ToolboxSceneVk::ToolboxSceneVk(nvvk::Context* ctx, nvvkhl::AllocVma* alloc, meshops::Context context, nvvk::Context::Queue extraQueue)
    : m_ctx(ctx)
    , m_alloc(alloc)
    , m_context(context)
    , m_qGCT1(extraQueue)
{
  m_dutil = std::make_unique<nvvk::DebugUtil>(ctx->m_device);  // Debug utility
}

ToolboxSceneVk::~ToolboxSceneVk()
{
  // The destructor wasn't called
  assert(m_deviceMeshes.empty());
}

//--------------------------------------------------------------------------------------------------
// Create all Vulkan resources to hold a nvvkhl::Scene
//
void ToolboxSceneVk::create(VkCommandBuffer cmd, micromesh_tool::ToolScene& scn)
{
  destroy();  // Make sure not to leave allocated buffers

  m_hasDisplacementMicromeshExt = m_ctx->hasDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME);

  // Create tables/meshlets of micro-vertex positions and topology for rasterizing meshes
  // with micromaps and heightmaps.
  microdisp::ResourcesVK res(*m_alloc, cmd);
  microdisp::initSplitPartsSubTri(res, m_micromeshSplitPartsVK);

  createMaterialBuffer(cmd, scn);
  createInstanceInfoBuffer(cmd, scn);
  if(!createDeviceMeshBuffer(cmd, scn))
  {
    destroy();  // Makes sure that external code does not need to call destroy() if it is not fully constructed
    return;
  }
  createTextureImages(cmd, scn.textures(), scn.images());
  if(!createDeviceBaryBuffer(cmd, m_qGCT1, scn))
  {
    destroy();  // Makes sure that external code does not need to call destroy() if it is not fully constructed
    return;
  }

  // Buffer references
  shaders::SceneDescription scene_desc{};
  scene_desc.materialAddress           = nvvk::getBufferDeviceAddress(m_ctx->m_device, m_bMaterial.buffer);
  scene_desc.deviceMeshInfoAddress     = nvvk::getBufferDeviceAddress(m_ctx->m_device, m_bDeviceMeshInfo.buffer);
  scene_desc.deviceBaryInfoAddress     = nvvk::getBufferDeviceAddress(m_ctx->m_device, m_bDeviceBaryInfo.buffer);
  scene_desc.instInfoAddress           = nvvk::getBufferDeviceAddress(m_ctx->m_device, m_bInstances.buffer);
  scene_desc.splitPartsVerticesAddress = m_micromeshSplitPartsVK.vertices.addr;
  scene_desc.splitPartsIndicesAddress  = m_micromeshSplitPartsVK.triangleIndices.addr;

  auto lock    = GetVkQueueOrAllocatorLock();
  m_bSceneDesc = m_alloc->createBuffer(cmd, sizeof(shaders::SceneDescription), &scene_desc,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  m_dutil->DBG_NAME(m_bSceneDesc.buffer);
}

//--------------------------------------------------------------------------------------------------
// Create a buffer of all materials, with only the elements we need
//
void ToolboxSceneVk::createMaterialBuffer(VkCommandBuffer cmd, const micromesh_tool::ToolScene& scn)
{
  nvh::ScopedTimer _st("- Create Material Buffer");

  // Importing the tinygltf::material in a struct with all value resolved
  nvh::GltfScene scene_materials;
  scene_materials.importMaterials(scn.model());

  // The material on the GPU is slightly different/smaller
  std::vector<shaders::GltfShadeMaterial> shade_materials;
  shade_materials.reserve(scene_materials.m_materials.size());

  // Lambda to convert from nvh::GltfMaterial to the GPU version
  auto convertMaterial = [](const nvh::GltfMaterial& m) {
    shaders::GltfShadeMaterial s{};
    s.emissiveFactor               = m.emissiveFactor;
    s.emissiveTexture              = m.emissiveTexture;
    s.khrDiffuseFactor             = m.specularGlossiness.diffuseFactor;
    s.khrDiffuseTexture            = m.specularGlossiness.diffuseTexture;
    s.khrSpecularFactor            = m.specularGlossiness.specularFactor;
    s.khrGlossinessFactor          = m.specularGlossiness.glossinessFactor;
    s.khrSpecularGlossinessTexture = m.specularGlossiness.specularGlossinessTexture;
    s.normalTexture                = m.normalTexture;
    s.normalTextureScale           = m.normalTextureScale;
    s.pbrBaseColorFactor           = m.baseColorFactor;
    s.pbrBaseColorTexture          = m.baseColorTexture;
    s.pbrMetallicFactor            = m.metallicFactor;
    s.pbrMetallicRoughnessTexture  = m.metallicRoughnessTexture;
    s.pbrRoughnessFactor           = m.roughnessFactor;
    s.shadingModel                 = m.shadingModel;
    s.alphaMode                    = m.alphaMode;
    s.alphaCutoff                  = m.alphaCutoff;
    s.khrDisplacementTexture       = m.displacement.displacementGeometryTexture;
    s.khrDisplacementFactor        = m.displacement.displacementGeometryFactor;
    s.khrDisplacementOffset        = m.displacement.displacementGeometryOffset;

    return s;
  };

  // Converting all materials
  for(const auto& m : scene_materials.m_materials)
  {
    shaders::GltfShadeMaterial s = convertMaterial(m);
    shade_materials.push_back(s);
  }

  // Add the scene's default material at the end
  {
    nvh::GltfScene  scene_material_default;
    tinygltf::Model tmp_model;
    tmp_model.materials.push_back(scn.material(-1));
    scene_material_default.importMaterials(tmp_model);
    assert(scene_material_default.m_materials.size() == 1);
    shade_materials.push_back(convertMaterial(scene_material_default.m_materials[0]));
  }

  // Create the buffer of all scene materials
  auto lock   = GetVkQueueOrAllocatorLock();
  m_bMaterial = m_alloc->createBuffer(cmd, shade_materials,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  m_dutil->DBG_NAME(m_bMaterial.buffer);
}

//--------------------------------------------------------------------------------------------------
// Array of instance information
// - Use by the vertex shader to retrieve the position of the instance
void ToolboxSceneVk::createInstanceInfoBuffer(VkCommandBuffer cmd, const micromesh_tool::ToolScene& scn)
{
  assert(scn.model().scenes.size() > 0);
  nvh::ScopedTimer _st("- Create Instance Buffer");

  const std::vector<micromesh_tool::ToolScene::Instance>& instances = scn.instances();

  std::vector<shaders::InstanceInfo> inst_info;
  inst_info.reserve(instances.size());
  for(const auto& instance : instances)
  {
    shaders::InstanceInfo info{};
    info.objectToWorld = instance.worldMatrix;
    info.worldToObject = nvmath::invert(instance.worldMatrix);
    info.materialID    = scn.meshes()[instance.mesh]->relations().material;
    inst_info.push_back(info);
  }

  auto lock = GetVkQueueOrAllocatorLock();
  m_bInstances = m_alloc->createBuffer(cmd, inst_info, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  m_dutil->DBG_NAME(m_bInstances.buffer);
}

//--------------------------------------------------------------------------------------------------
// Creating information per primitive
// - Create a buffer of Vertex and Index for each primitive
// - Each primInfo has a reference to the vertex and index buffer, and which material id it uses
//
bool ToolboxSceneVk::createDeviceMeshBuffer(VkCommandBuffer cmd, micromesh_tool::ToolScene& scn)
{
  nvh::ScopedTimer _st("- Create Vertex Buffer");

  const std::vector<std::unique_ptr<micromesh_tool::ToolMesh>>& meshes = scn.meshes();

  auto lock = GetVkQueueOrAllocatorLock();
  for(size_t meshIndex = 0; meshIndex < meshes.size(); meshIndex++)
  {
    micromesh_tool::ToolMesh&   mesh     = *meshes[meshIndex];
    meshops::ResizableMeshView& meshView = mesh.view();

    // Create the buffers of the attributes that exist
    meshops::DeviceMeshSettings settings{};
    settings.attribFlags = meshView.getMeshAttributeFlags();
    if(mesh.relations().bary != -1 && !scn.barys().empty())
    {
      // Forcing the direction vector to be present (will use normal if was not provided)
      // as direction vectors are mandatory for displacement
      settings.attribFlags |= meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBit;
      settings.attribFlags |= meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBoundsBit;

      // Provide defaults to initialize the device vertexDirectionBounds with
      // the bary group's bias and scale in case ToolMesh has no direction
      // bounds. They should be mutually exclusive.
      const bary::BasicView& basic  = scn.barys()[mesh.relations().bary]->groups()[mesh.relations().group].basic;
      settings.directionBoundsBias  = basic.groups[0].floatBias.r;
      settings.directionBoundsScale = basic.groups[0].floatScale.r;
    }

    meshops::DeviceMesh     d{nullptr};
    const micromesh::Result result = meshops::meshopsDeviceMeshCreate(m_context, meshView, settings, &d);
    if(micromesh::Result::eSuccess != result)
    {
      LOGE("Error: Could not create device mesh %zu\n", meshIndex);
      return false;
    }
    m_deviceMeshes.push_back(d);

    float bias;
    float scale;
    int   imageIndex;
    if(scn.getHeightmap(mesh.relations().material, bias, scale, imageIndex))
    {
      // Create the base mesh topology
      meshops::MeshTopologyData topology;
      {
        meshops::OpBuildTopology_input input;
        input.meshView = meshView;
        meshops::OpBuildTopology_output output;
        output.meshTopology = &topology;
        if(meshops::meshopsOpBuildTopology(m_context, 1, &input, &output) != micromesh::Result::eSuccess)
        {
          LOGE("Error: failed to build mesh topology\n");
          return false;
        }
      }
      m_meshWatertightIndices.push_back(createWatertightIndicesBuffer(cmd, meshView.triangleVertices, topology));
    }
    else
    {
      // Create a null element to keep indexing consistent with m_deviceMeshes.
      m_meshWatertightIndices.emplace_back();
    }
  }

  std::vector<shaders::DeviceMeshInfo> device_mesh_infos;
  device_mesh_infos.reserve(m_deviceMeshes.size());
  for(size_t i = 0; i < m_deviceMeshes.size(); ++i)
  {
    auto&                   mesh   = m_deviceMeshes[i];
    meshops::DeviceMeshVK*  meshVk = meshops::meshopsDeviceMeshGetVK(mesh);
    shaders::DeviceMeshInfo info{};
    info.triangleVertexIndexBuffer = nvvk::getBufferDeviceAddress(m_ctx->m_device, meshVk->triangleVertexIndexBuffer.buffer);
    info.triangleAttributesBuffer = nvvk::getBufferDeviceAddress(m_ctx->m_device, meshVk->triangleAttributesBuffer.buffer);
    info.vertexPositionNormalBuffer = nvvk::getBufferDeviceAddress(m_ctx->m_device, meshVk->vertexPositionNormalBuffer.buffer);
    info.vertexTangentSpaceBuffer = nvvk::getBufferDeviceAddress(m_ctx->m_device, meshVk->vertexTangentSpaceBuffer.buffer);
    info.vertexTexcoordBuffer   = nvvk::getBufferDeviceAddress(m_ctx->m_device, meshVk->vertexTexcoordBuffer.buffer);
    info.vertexDirectionsBuffer = nvvk::getBufferDeviceAddress(m_ctx->m_device, meshVk->vertexDirectionsBuffer.buffer);
    info.vertexDirectionBoundsBuffer = nvvk::getBufferDeviceAddress(m_ctx->m_device, meshVk->vertexDirectionBoundsBuffer.buffer);
    info.vertexImportanceBuffer = nvvk::getBufferDeviceAddress(m_ctx->m_device, meshVk->vertexImportanceBuffer.buffer);
    info.triangleWatertightIndicesBuffer = nvvk::getBufferDeviceAddress(m_ctx->m_device, m_meshWatertightIndices[i].buffer);
    info.deviceAttribFlags = meshVk->deviceAttribFlags;
    info.sourceAttribFlags = meshVk->sourceAttribFlags;
    device_mesh_infos.push_back(info);
  }

  auto usage_flag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                    | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

  // Creating the buffer of all device mesh information
  m_bDeviceMeshInfo = m_alloc->createBuffer(cmd, device_mesh_infos, usage_flag);
  m_dutil->DBG_NAME(m_bDeviceMeshInfo.buffer);
  return true;
}


//--------------------------------------------------------------------------------------------------
// Creating the resources holding the Barycentric data for micromeshes
//
bool ToolboxSceneVk::createDeviceBaryBuffer(VkCommandBuffer cmd, nvvk::Context::Queue extraQueue, micromesh_tool::ToolScene& scn)
{
  if(scn.barys().empty())
  {
    return true;
  }

  std::map<std::pair<int32_t, int32_t>, const micromesh_tool::ToolMesh*> baryToMeshMap;
  for(auto& mesh : scn.meshes())
  {
    baryToMeshMap[{mesh->relations().bary, mesh->relations().group}] = mesh.get();
  }

  uint64_t usageFlags = eDeviceMicromeshUsageRasterizingBit;

  if(m_hasDisplacementMicromeshExt)
  {
    usageFlags |= eDeviceMicromeshUsageRaytracingBit;
  }

  std::vector<shaders::DeviceBaryInfo> deviceBaryInfos;
  deviceBaryInfos.reserve(m_barys.size());
  for(int32_t baryIndex = 0; baryIndex < static_cast<int32_t>(scn.barys().size()); ++baryIndex)
  {
    // Create a DeviceBary from a ToolBary
    const std::unique_ptr<micromesh_tool::ToolBary>& toolBary = scn.barys()[baryIndex];
    m_barys.push_back(std::make_unique<DeviceBary>());
    std::unique_ptr<DeviceBary>& deviceBary = m_barys.back();
    for(int32_t baryGroup = 0; baryGroup < static_cast<int32_t>(toolBary->groups().size()); ++baryGroup)
    {
      // Build a structure of addresses to reference the ToolMicromap data in
      // shaders. These are linearized, so m_deviceBaryInfoMap is created to
      // refer back to them given a bary and group index.
      shaders::DeviceBaryInfo info{};

      // Add a DeviceMicromap to the DeviceBary for every ToolBary's group
      const bary::ContentView& groupView     = toolBary->groups()[baryGroup];
      auto                     displacedMesh = baryToMeshMap.find({baryIndex, baryGroup});
      if(displacedMesh == baryToMeshMap.end())
      {
        LOGI("Skipping unused micromap %i group %i\n", baryIndex, baryGroup);
        m_barys.back()->addEmpty();
      }
      else if(groupView.basic.valuesInfo->valueFormat != bary::Format::eDispC1_r11_unorm_block)
      {
        LOGW("Warning: cannot render uncompressed micromap %i group %i\n", baryIndex, baryGroup);
        m_barys.back()->addEmpty();
      }
      else
      {
        m_barys.back()->addMicromap(m_context, *m_alloc, extraQueue.queue, extraQueue.familyIndex, cmd, usageFlags,
                                    m_micromeshSplitPartsVK, groupView, *displacedMesh->second);
        const DeviceMicromap& micromap = deviceBary->micromaps().back();

        info.baryValuesBuffer    = micromap.valuesAddress();
        info.baryTrianglesBuffer = micromap.trianglesAddress();
        if(micromap.raster())
        {
          const microdisp::MicromeshSetCompressedVK& rasterData = micromap.raster()->micromeshSet;
          assert(rasterData.meshDatas.size() == 1);
          info.rasterMeshDataBindingBuffer =
              nvvk::getBufferDeviceAddress(m_ctx->m_device, rasterData.meshDatas[0].binding.buffer);
        }
      }

      // Add info even if it's empty so that baryInfoIndex() can always return a valid index
      // TODO: use same index as the mesh
      m_deviceBaryInfoMap[{baryIndex, baryGroup}] = static_cast<int32_t>(deviceBaryInfos.size());
      deviceBaryInfos.push_back(info);
    }
  }

  auto usage_flag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                    | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

  // Creating the buffer of all device bary information
  m_bDeviceBaryInfo = m_alloc->createBuffer(cmd, deviceBaryInfos, usage_flag);
  m_dutil->DBG_NAME(m_bDeviceBaryInfo.buffer);

  return true;
}


//--------------------------------------------------------------------------------------------------
// Create heightmap displacement seam welding information
nvvk::Buffer ToolboxSceneVk::createWatertightIndicesBuffer(VkCommandBuffer                          cmd,
                                                           meshops::ArrayView<const nvmath::vec3ui> indices,
                                                           const meshops::MeshTopologyData&         topology)
{
  // Default to no edge sanitization
  shaders::WatertightIndices ignored{/*.seamEdge =*/{nvmath::vec2i{WATERTIGHT_INDICES_INVALID_VERTEX},
                                                     nvmath::vec2i{WATERTIGHT_INDICES_INVALID_VERTEX},
                                                     nvmath::vec2i{WATERTIGHT_INDICES_INVALID_VERTEX}},
                                     /*.padding_ =*/{},
                                     /*.watertightCornerVertex =*/nvmath::vec3i{WATERTIGHT_INDICES_INVALID_VERTEX},
                                     /*.adjacentTriangles =*/nvmath::vec3i{WATERTIGHT_INDICES_INVALID_VERTEX}};

  std::vector<shaders::WatertightIndices> triInfos(indices.size(), ignored);

  meshops::ArrayView<const micromesh::Vector_uint32_3> triVertices(indices);
  meshops::ArrayView<const micromesh::Vector_uint32_3> triVerticesWt(topology.triangleVertices);

  // Resolve cracks when tessellating and displacing with a heightmap. See TriInfo.
  // UVs at base triangle corners are snapped to the watertight index.
  // UVs along shared edges, not includin the corners, are averaged.
  // We only do this if triangles have a been split due to different normals/UVs.
  for(uint32_t triIdx = 0; triIdx < static_cast<uint32_t>(triVertices.size()); ++triIdx)
  {
    shaders::WatertightIndices& triInfo = triInfos[triIdx];

    // If the watertight vertex ID is different to the regular one, get the shader to use that for corner vertices
    micromesh::Vector_uint32_3 tri   = triVertices[triIdx];
    micromesh::Vector_uint32_3 triWt = triVerticesWt[triIdx];

    // Skip degenerate triangles
    if(micromesh::meshIsTriangleDegenerate(triWt))
      continue;

    triInfo.watertightCornerVertex.x = tri.x == triWt.x ? WATERTIGHT_INDICES_INVALID_VERTEX : triWt.x;
    triInfo.watertightCornerVertex.y = tri.y == triWt.y ? WATERTIGHT_INDICES_INVALID_VERTEX : triWt.y;
    triInfo.watertightCornerVertex.z = tri.z == triWt.z ? WATERTIGHT_INDICES_INVALID_VERTEX : triWt.z;

    // Find adjacent split triangles.
    // edge ordering (vertices of each edge are unordered): {v0,v1}, {v1,v2}, {v2,v0}
    micromesh::Vector_uint32_3 triEdgesWt = topology.triangleEdges[triIdx];
    for(uint32_t edgeIdx = 0; edgeIdx < 3; ++edgeIdx)
    {
      uint32_t edgeWt = (&triEdgesWt.x)[edgeIdx];

      // Compute the indices of the edge vertices in the current triangle
      micromesh::Vector_uint32_2 edgeVerticesWt = topology.edgeVertices[edgeWt];
      uint32_t                   edgeVertex0Idx = micromesh::topoTriangleFindVertex(triWt, edgeVerticesWt.x);
      uint32_t                   edgeVertex1Idx = micromesh::topoTriangleFindVertex(triWt, edgeVerticesWt.y);
      uint32_t                   edgeVertex0    = (&tri.x)[edgeVertex0Idx];
      uint32_t                   edgeVertex1    = (&tri.x)[edgeVertex1Idx];
      nvmath::vec2i              edgeVertices{edgeVertex0, edgeVertex1};

      // Search adjacent triangles
      for(auto& otherTriIdx : topology.getEdgeTriangles(edgeWt))
      {
        if(triIdx == otherTriIdx)
          continue;

        // Store per-triangle watertight indices for dynamic heightmap LOD.
        triInfo.adjacentTriangles[edgeIdx] = otherTriIdx;

        // Compute the indices of the current triangle's edge's vertices in the other triangle, found by matching
        // the indices in the watertight triangle.
        micromesh::Vector_uint32_3 otherTri   = triVertices[otherTriIdx];
        micromesh::Vector_uint32_3 otherTriWt = triVerticesWt[otherTriIdx];
        uint32_t      otherEdgeVertex0Idx     = micromesh::topoTriangleFindVertex(otherTriWt, edgeVerticesWt.x);
        uint32_t      otherEdgeVertex1Idx     = micromesh::topoTriangleFindVertex(otherTriWt, edgeVerticesWt.y);
        uint32_t      otherEdgeVertex0        = (&otherTri.x)[otherEdgeVertex0Idx];
        uint32_t      otherEdgeVertex1        = (&otherTri.x)[otherEdgeVertex1Idx];
        nvmath::vec2i otherEdgeVertices{otherEdgeVertex0, otherEdgeVertex1};

        // If the adjacent watertight triangle has different vertices, we need to weld the edge
        if(edgeVertices != otherEdgeVertices)
        {
          // Make edge order consistent for triangle primTriIdx. If the order of the two is not one of the three
          // expexted, swap them. Needed because getEdgeVertices() doesn't/couldn't have consistent pair ordering.
          if(!((edgeVertex0Idx == 0 && edgeVertex1Idx == 1) || (edgeVertex0Idx == 1 && edgeVertex1Idx == 2)
               || (edgeVertex0Idx == 2 && edgeVertex1Idx == 0)))
            std::swap(otherEdgeVertices.x, otherEdgeVertices.y);

          // Write the other triangle's edge's vertex indices for reference by draw_bary_lod.mesh
          triInfo.seamEdge[edgeIdx] = otherEdgeVertices;
        }
      }
    }
  }

  return m_alloc->createBuffer(cmd, triInfos, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
}

void ToolboxSceneVk::createTextureImages(VkCommandBuffer cmd, const std::vector<tinygltf::Texture>& textures, const ToolImageVector& images)
{
  nvh::ScopedTimer _st("- Create Textures\n");

  VkSamplerCreateInfo sampler_create_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  sampler_create_info.minFilter  = VK_FILTER_LINEAR;
  sampler_create_info.magFilter  = VK_FILTER_LINEAR;
  sampler_create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sampler_create_info.maxLod     = FLT_MAX;

  // Make dummy image(1,1), needed as we cannot have an empty array
  auto addDefaultImage = [&](uint32_t idx, const std::array<uint8_t, 4>& color) {
    VkImageCreateInfo image_create_info = nvvk::makeImage2DCreateInfo(VkExtent2D{1, 1});
    nvvk::Image       image             = m_alloc->createImage(cmd, 4, color.data(), image_create_info);
    assert(idx < m_images.size());
    m_images[idx] = {image, image_create_info};
    m_dutil->setObjectName(m_images[idx].nvvkImage.image, "Dummy");
  };

  // Make dummy texture/image(1,1), needed as we cannot have an empty array
  auto addDefaultTexture = [&]() {
    assert(!m_images.empty());
    SceneImage&           scn_image = m_images[0];
    VkImageViewCreateInfo iv_info   = nvvk::makeImageViewCreateInfo(scn_image.nvvkImage.image, scn_image.createInfo);
    m_textures.emplace_back(m_alloc->createTexture(scn_image.nvvkImage, iv_info, sampler_create_info));
  };

  // Load images in parallel
  m_images.resize(images.size());
  uint32_t num_threads = std::min((uint32_t)images.size(), std::thread::hardware_concurrency());
  nvh::parallel_batches<1>(  // Not batching
      images.size(),
      [&](uint64_t i) {
        auto& image = *images[i];
        LOGI("  - (%" PRIu64 ") %s \n", i, image.relativePath().string().c_str());
        loadImage(image, m_images[i]);
      },
      num_threads);

  // Create Vulkan images
  auto lock = GetVkQueueOrAllocatorLock();
  for(size_t i = 0; i < m_images.size(); i++)
  {
    if(!createImage(cmd, m_images[i]))
    {
      addDefaultImage((uint32_t)i, {255, 0, 255, 255});  // Image not present or incorrectly loaded (image.empty)
    }
  }

  // Add default image if nothing was loaded
  if(images.empty())
  {
    m_images.resize(1);
    addDefaultImage(0, {255, 255, 255, 255});
  }

  // Creating the textures using the above images
  m_textures.reserve(textures.size());
  for(size_t i = 0; i < textures.size(); i++)
  {
    int source_image = textures[i].source;
    if(static_cast<size_t>(source_image) >= images.size() || source_image < 0)
    {
      addDefaultTexture();  // Incorrect source image
      continue;
    }

    SceneImage&           scn_image = m_images[source_image];
    VkImageViewCreateInfo iv_info   = nvvk::makeImageViewCreateInfo(scn_image.nvvkImage.image, scn_image.createInfo);
    m_textures.emplace_back(m_alloc->createTexture(scn_image.nvvkImage, iv_info, sampler_create_info));
  }

  // Add a default texture, cannot work with empty descriptor set
  if(textures.empty())
  {
    addDefaultTexture();
  }
}

//--------------------------------------------------------------------------------------------------
// Loading images from disk
//
void ToolboxSceneVk::loadImage(micromesh_tool::ToolImage& toolImage, SceneImage& image)
{
  if(!toolImage.info().valid())
  {
    // Image failed to load, e.g. file not found.
    return;
  }

  VkFormat vkFormat = toolImage.info().vkFormat();
  if(vkFormat == VK_FORMAT_UNDEFINED)
  {
    // Unsupported image format
    return;
  }

  auto rawData = static_cast<uint8_t*>(toolImage.raw());
  if(!rawData)
  {
    return;
  }

  image.size   = {static_cast<uint32_t>(toolImage.info().width), static_cast<uint32_t>(toolImage.info().height)};
  image.format = vkFormat;
  image.mipData.emplace_back(rawData, rawData + toolImage.info().totalBytes());
}

bool ToolboxSceneVk::createImage(const VkCommandBuffer& cmd, SceneImage& image)
{
  if(image.size.width == 0 || image.size.height == 0)
    return false;

  VkFormat          format            = image.format;
  VkExtent2D        img_size          = image.size;
  VkImageCreateInfo image_create_info = nvvk::makeImage2DCreateInfo(img_size, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);

  // Check if we can generate mipmap with the the incoming image
  bool               can_generate_mipmaps = false;
  VkFormatProperties format_properties;
  vkGetPhysicalDeviceFormatProperties(m_ctx->m_physicalDevice, format, &format_properties);
  if((format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) == VK_FORMAT_FEATURE_BLIT_DST_BIT)
    can_generate_mipmaps = true;
  if(image.mipData.size() > 1)  // Use only the number of levels defined
    image_create_info.mipLevels = (uint32_t)image.mipData.size();
  if(image.mipData.size() == 1 && can_generate_mipmaps == false)
    image_create_info.mipLevels = 1;  // Cannot use cmdGenerateMipmaps

  // Keep info for the creation of the texture
  image.createInfo = image_create_info;

  VkDeviceSize buffer_size  = image.mipData[0].size();
  nvvk::Image  result_image = m_alloc->createImage(cmd, buffer_size, image.mipData[0].data(), image_create_info);

  if(image.mipData.size() == 1 && can_generate_mipmaps)
  {
    nvvk::cmdGenerateMipmaps(cmd, result_image.image, format, img_size, image_create_info.mipLevels);
  }
  else
  {
    // Create all mip-levels
    nvvk::cmdBarrierImageLayout(cmd, result_image.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    auto staging = m_alloc->getStaging();
    for(uint32_t mip = 1; mip < (uint32_t)image.mipData.size(); mip++)
    {
      image_create_info.extent.width  = std::max(1u, image.size.width >> mip);
      image_create_info.extent.height = std::max(1u, image.size.height >> mip);

      VkOffset3D               offset{};
      VkImageSubresourceLayers subresource{};
      subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      subresource.layerCount = 1;
      subresource.mipLevel   = mip;

      std::vector<uint8_t>& mipresource = image.mipData[mip];
      VkDeviceSize          bufferSize  = mipresource.size();
      if(image_create_info.extent.width > 0 && image_create_info.extent.height > 0)
      {
        staging->cmdToImage(cmd, result_image.image, offset, image_create_info.extent, subresource, bufferSize,
                            mipresource.data());
      }
    }
    nvvk::cmdBarrierImageLayout(cmd, result_image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }

  if(!image.imgName.empty())
  {
    m_dutil->setObjectName(result_image.image, image.imgName);
  }
  else
  {
    m_dutil->DBG_NAME(result_image.image);
  }

  // Clear image.mipData as it is no longer needed
  image = {result_image, image_create_info, image.srgb, image.imgName};

  return true;
}

void ToolboxSceneVk::destroy()
{
  for(auto& mesh : m_deviceMeshes)
  {
    meshops::meshopsDeviceMeshDestroy(m_context, mesh);
  }
  m_deviceMeshes = {};

  for(auto& buffer : m_meshWatertightIndices)
  {
    m_alloc->destroy(buffer);
  }
  m_meshWatertightIndices.clear();

  microdisp::ResourcesVK res(*m_alloc, VK_NULL_HANDLE);
  m_micromeshSplitPartsVK.deinit(res);

  auto lock = GetVkQueueOrAllocatorLock();
  m_alloc->destroy(m_bMaterial);
  m_alloc->destroy(m_bDeviceMeshInfo);
  m_alloc->destroy(m_bDeviceBaryInfo);
  m_alloc->destroy(m_bInstances);
  m_alloc->destroy(m_bSceneDesc);

  for(auto& bary : m_barys)
  {
    bary->deinit(*m_alloc);
  }
  m_barys.clear();

  for(auto& i : m_images)
  {
    m_alloc->destroy(i.nvvkImage);
  }
  m_images.clear();

  for(auto& t : m_textures)
  {
    vkDestroyImageView(m_ctx->m_device, t.descriptor.imageView, nullptr);
  }
  m_textures.clear();
}
