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

#include "meshops_bake_vk.hpp"
#include <glm/gtx/dual_quaternion.hpp>
#include <meshops/meshops_operations.h>
#include <meshops_internal/meshops_context.h>
#include <meshops_internal/meshops_texture.h>
#include <meshops_internal/heightmap.hpp>
#include <meshops_internal/pn_triangles.hpp>
#include <meshops_internal/umesh_util.hpp>
#include <meshops/bias_scale.hpp>
#include "nvvk/specialization.hpp"
#include <glm/gtx/hash.hpp>
#include <nvmath/nvmath_types.h>
#include <nvh/parallel_work.hpp>
#include <scoped_timer.hpp>
#include <nvvk/buffers_vk.hpp>
#include <nvvk/shaders_vk.hpp>
#include <nvvk/pipeline_vk.hpp>
#include <vector>

#include "_autogen/bary_trace.comp.h"
#include "_autogen/resample.vert.h"
#include "_autogen/resample.frag.h"

namespace meshops {

using namespace glsl_shared;

template <class ArrayInfoType>
ArrayView<typename ArrayInfoType::value_type> makeArrayView(const ArrayInfoType& arrayInfo)
{
  return ArrayView(reinterpret_cast<typename ArrayInfoType::value_type*>(arrayInfo.data), arrayInfo.count, arrayInfo.byteStride);
}

static bool getMemoryUsageVk(VkPhysicalDevice physicalDevice, VkDeviceSize* budget, VkDeviceSize* usage)
{
  VkPhysicalDeviceMemoryBudgetPropertiesEXT memoryBudgetProperties = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
  VkPhysicalDeviceMemoryProperties2 memoryProperties2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, &memoryBudgetProperties};
  vkGetPhysicalDeviceMemoryProperties2(physicalDevice, &memoryProperties2);
  uint32_t heapIndex = (uint32_t)-1;
  for(uint32_t memoryType = 0; memoryType < memoryProperties2.memoryProperties.memoryTypeCount; memoryType++)
  {
    if(memoryProperties2.memoryProperties.memoryTypes[memoryType].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    {
      heapIndex = memoryProperties2.memoryProperties.memoryTypes[memoryType].heapIndex;
      break;
    }
  }
  if(heapIndex == (uint32_t)-1)
  {
    return false;
  }
  *budget = memoryBudgetProperties.heapBudget[heapIndex];
  *usage  = memoryBudgetProperties.heapUsage[heapIndex];
  return true;
}

BakerVK::BakerVK(micromesh::OpContext micromeshContext, meshops::ContextVK& vkContext, unsigned int threadCount)
    : m_micromeshContext(micromeshContext)
    , m_alloc(*vkContext.resAllocator)
    , m_threadCount(threadCount)
{
  nvvk::Context& ctx = *vkContext.context;
  m_device           = ctx.m_device;
  m_physicalDevice   = ctx.m_physicalDevice;
  m_queueT           = vkContext.queueT;
  m_queueC           = vkContext.queueC;
  m_queueGCT         = vkContext.queueGCT;
  m_rtBuilder.setup(m_device, &m_alloc, m_queueC.familyIndex);
}

BakerVK::~BakerVK()
{
  destroy();
}

bool BakerVK::bakeAndResample(const meshops::OpBake_input&              input,
                              const GeometryBatch&                      batch,
                              bool                                      resample,
                              const std::vector<VkDescriptorImageInfo>& inputTextures,
                              const std::vector<VkDescriptorImageInfo>& outputTextures,
                              const std::vector<VkDescriptorImageInfo>& distanceTextures,
                              ArrayView<meshops::Texture>               outputTextureInfo)
{
  LOGI("Batch %u/%u\n", batch.batchIndex + 1, batch.totalBatches);

  if(!createVkHighGeometry(input, input.referenceMeshView, batch))
  {
    LOGE("createVkHighGeometry() failed\n");
    return false;
  }

  // Create pipeline and descriptor set
  createBakerPipeline();

  if(resample && !outputTextureInfo.empty())
    createResamplerPipeline(inputTextures, outputTextures, distanceTextures);

  VkDeviceSize budget, usage;
  getMemoryUsageVk(m_physicalDevice, &budget, &usage);
  LOGI("Memory usage: %.2f/%.2f MB\n", (double)usage / (1024.0 * 1024.0), (double)budget / (1024.0 * 1024.0));

  // Run compute shader
  runComputePass(input, batch.batchIndex + 1 == batch.totalBatches);

  // Resample all textures, keeping the minimum hits for this batch
  if(resample && !outputTextureInfo.empty())
  {
    runResampler(input, outputTextureInfo);
  }

  destroyBakerPipeline();

  if(resample && !outputTextureInfo.empty())
    destroyResamplerPipeline();

  destroyVkHighGeometry();
  return true;
}

//--------------------------------------------------------------------------------------------------
// Creating Vulkan resources
//
void BakerVK::createVkResources(const meshops::OpBake_input& info,
                                const meshops::MeshView&     lowScene,
                                const meshops::MeshView&     highScene,
                                MutableArrayView<float>      distances)
{
  {
    ScopedTimer t("Create VK Resources\n");

    m_push.maxDistance             = info.settings.maxTraceLength;
    m_push.replaceDirectionLength  = info.settings.maxTraceLength != 0.0f;
    m_push.highMeshHasDisplacement = info.referenceMeshHeightmap.texture != nullptr;
    m_push.uniDirectional          = info.settings.uniDirectional ? 1 : 0;
    m_push.maxDistanceFactor       = info.settings.maxDistanceFactor;

    VkBufferUsageFlags bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    // Creating the Vulkan resources of the scene
    nvvk::CommandPool cmdPool(m_device, m_queueT.familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, m_queueT.queue);
    VkCommandBuffer   cmdBuf = cmdPool.createCommandBuffer();

    // Direction bounds are fitted to the displacements during baking.
    m_push.hasDirectionBounds = !lowScene.vertexDirectionBounds.empty();
    if(m_push.hasDirectionBounds)
    {
      VkDeviceSize boundsTotalBytes = lowScene.vertexDirectionBounds.size() * sizeof(lowScene.vertexDirectionBounds[0]);
      m_directionBoundsBuf = m_alloc.createBuffer(boundsTotalBytes, bufferUsage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
      {
        nvmath::vec2f* directionBounds = static_cast<nvmath::vec2f*>(m_alloc.map(m_directionBoundsBuf));
        std::copy(lowScene.vertexDirectionBounds.begin(), lowScene.vertexDirectionBounds.end(), directionBounds);
        m_alloc.unmap(m_directionBoundsBuf);
      }
      m_directionBoundsOrigBuf = m_alloc.createBuffer(boundsTotalBytes, bufferUsage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
      {
        nvmath::vec2f* directionBounds = static_cast<nvmath::vec2f*>(m_alloc.map(m_directionBoundsOrigBuf));
        std::copy(lowScene.vertexDirectionBounds.begin(), lowScene.vertexDirectionBounds.end(), directionBounds);
        m_alloc.unmap(m_directionBoundsOrigBuf);
      }
    }

    assert(!lowScene.vertexDirections.empty());
    createVertexBuffer(cmdBuf, lowScene, m_lowVk);

    // Initialize distances to the max float value as a "no hit" marker.
    // This way we can take the min() of multiple traces when baking geometry in batches.
    // This is undone later, converting any remaining values back to zero displacement.
    std::fill(distances.begin(), distances.end(), std::numeric_limits<float>::max());

    // Initialize min/max displacement values to float [max, min]. These are uploaded and used during tracing to compute
    // per-vertex direction bounds. m_microMesh.baryBaker() recomputes them offline so they are not copied back to this
    // array.
    std::vector<nvmath::vec2f> minMaxPairs(lowScene.triangleCount());
    std::fill(minMaxPairs.begin(), minMaxPairs.end(),
              nvmath::vec2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()));

    // Create a buffer holding all distances
    std::vector<Triangle> triangles(lowScene.triangleCount());
    for(size_t i = 0; i < triangles.size(); ++i)
    {
      triangles[i].meshTriangle = static_cast<uint32_t>(i);
      triangles[i].subdivLevel =
          lowScene.triangleSubdivisionLevels.empty() ? info.settings.level : lowScene.triangleSubdivisionLevels[i];
      triangles[i].valueCount = bary::baryValueFrequencyGetCount(bary::ValueFrequency::ePerVertex, triangles[i].subdivLevel);
      triangles[i].valueFirst = i == 0 ? 0 : triangles[i - 1].valueFirst + triangles[i - 1].valueCount;
    }
    m_distanceBuf = m_alloc.createBuffer(cmdBuf, distances.size() * sizeof(distances[0]), distances.data(), bufferUsage,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    m_trianglesBuf      = m_alloc.createBuffer(cmdBuf, triangles, bufferUsage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    m_triangleMinMaxBuf = m_alloc.createBuffer(cmdBuf, minMaxPairs, bufferUsage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    // Buffers holding the barycentric values for all levels
    // FIXME: bmap seems to have an extra unused level defined
    assert(info.settings.level <= MAX_SUBDIV_LEVELS);
    baryutils::BaryLevelsMap bmap;
    bmap.initialize(bary::ValueLayout::eTriangleBirdCurve, info.settings.level);
    int maxSubdivLevels = std::min(bmap.getNumLevels(), uint32_t(MAX_SUBDIV_LEVELS));
    m_baryCoordBuf.resize(maxSubdivLevels);
    for(int levelIdx = 0; levelIdx < maxSubdivLevels; levelIdx++)
    {
      const baryutils::BaryLevelsMap::Level& level         = bmap.getLevel(levelIdx);
      uint32_t                               numBaryCoords = static_cast<int>(level.coordinates.size());
      m_maxNumBaryCoord                                    = std::max(numBaryCoords, m_maxNumBaryCoord);
      std::vector<nvmath::vec3f> baryCoord(numBaryCoords);
      for(uint32_t i = 0; i < numBaryCoords; i++)
      {
        level.getFloatCoord(i, &baryCoord[i].x);
      }
      m_baryCoordBuf[levelIdx] = m_alloc.createBuffer(cmdBuf, baryCoord, bufferUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    }

    cmdPool.submitAndWait(cmdBuf);
    m_alloc.finalizeAndReleaseStaging();
  }
}


bool BakerVK::createVkHighGeometry(const meshops::OpBake_input& info, const meshops::MeshView& highScene, const GeometryBatch& batch)
{
  nvvk::CommandPool cmdPool(m_device, m_queueT.familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, m_queueT.queue);
  VkCommandBuffer   cmdBuf = cmdPool.createCommandBuffer();

  if(info.referenceMeshHeightmap.texture)
  {
    if(!createTessellatedVertexBuffer(info, cmdBuf, highScene, m_highVk, batch, info.referenceMeshHeightmap.maxSubdivLevel))
      return false;
  }
  else
  {
    createVertexBuffer(cmdBuf, highScene, m_highVk);
    LOGI("Batch high nb triangles: %zu\n", m_highVk.totalTriangles);
  }

  SceneDescription sceneDesc;
  sceneDesc.primLoInfoAddress      = nvvk::getBufferDeviceAddress(m_device, m_lowVk.primInfoBuf.buffer);
  sceneDesc.primHiInfoAddress      = nvvk::getBufferDeviceAddress(m_device, m_highVk.primInfoBuf.buffer);
  sceneDesc.distancesAddress       = nvvk::getBufferDeviceAddress(m_device, m_distanceBuf.buffer);
  sceneDesc.trianglesAddress       = nvvk::getBufferDeviceAddress(m_device, m_trianglesBuf.buffer);
  sceneDesc.triangleMinMaxsAddress = nvvk::getBufferDeviceAddress(m_device, m_triangleMinMaxBuf.buffer);
  for(size_t levelIdx = 0; levelIdx < m_baryCoordBuf.size(); levelIdx++)
    sceneDesc.baryCoordsAddress[levelIdx] = nvvk::getBufferDeviceAddress(m_device, m_baryCoordBuf[levelIdx].buffer);

  m_sceneDescBuf = m_alloc.createBuffer(cmdBuf, sizeof(SceneDescription), &sceneDesc,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

  cmdPool.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();

  // Create BVH of high mesh
  createBottomLevelAS(highScene, m_highVk, batch);
  createTopLevelAS(info, highScene, batch);

  return true;
}

void BakerVK::destroyVkHighGeometry()
{
  m_alloc.destroy(m_sceneDescBuf);
  m_rtBuilder.destroy();
  destroyVkScene(m_highVk);
}

//--------------------------------------run------------------------------------------------------------
// Creating information per primitive
// - Create a buffer of Vertex and Index for each primitive
// - Each primInfo has a reference to the vertex and index buffer, and which material id it uses
//
void BakerVK::createVertexBuffer(VkCommandBuffer cmdBuf, const meshops::MeshView& meshView, SceneVK& vkScene)
{
  nvh::Stopwatch sw;
  LOGI("- Create Vertex Buffer ");

  std::vector<PrimMeshInfo> primInfo;  // The array of all primitive information
  uint32_t                  primIdx{0};

  auto usageFlag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                   | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

  for(auto meshIdx = 0; meshIdx < 1; ++meshIdx)
  {
    nvvk::Buffer v_buffer;  // Vertex buffer result
    {
      // Buffer of Vertex per primitive
      std::vector<CompressedVertex> cvertices;
      for(size_t v_ctx = 0; v_ctx < meshView.vertexCount(); v_ctx++)
      {
        Vertex v{};
        size_t idx = v_ctx;
        v.position = meshView.vertexPositions[idx];
        if(!meshView.vertexNormals.empty())
          v.normal = meshView.vertexNormals[idx];
        if(!meshView.vertexTangents.empty())
          v.tangent = meshView.vertexTangents[idx];
        if(!meshView.vertexTexcoords0.empty())
          v.texCoord = meshView.vertexTexcoords0[idx];
        if(!meshView.vertexDirections.empty())
          v.displacementDirection = meshView.vertexDirections[idx];
        cvertices.emplace_back(compressVertex(v));
      }
      v_buffer = m_alloc.createBuffer(cmdBuf, cvertices, usageFlag | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
      vkScene.numVertices.push_back((uint32_t)cvertices.size());
    }
    vkScene.verticesBuf.emplace_back(v_buffer);

    // Buffer of indices
    auto i_buffer = m_alloc.createBuffer(cmdBuf, meshView.triangleVertices.size() * sizeof(meshView.triangleVertices[0]),
                                         meshView.triangleVertices.data(), usageFlag | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    vkScene.indicesBuf.emplace_back(i_buffer);

    // Primitive information, material Id and addresses of buffers
    PrimMeshInfo info{};
    info.materialIndex = 0;
    info.vertexAddress = nvvk::getBufferDeviceAddress(m_device, v_buffer.buffer);
    info.indexAddress  = nvvk::getBufferDeviceAddress(m_device, i_buffer.buffer);
    if(m_directionBoundsBuf.buffer)
    {
      info.vertexDirectionBoundsAddress = nvvk::getBufferDeviceAddress(m_device, m_directionBoundsBuf.buffer);
    }
    if(m_directionBoundsOrigBuf.buffer)
    {
      info.vertexDirectionBoundsOrigAddress = nvvk::getBufferDeviceAddress(m_device, m_directionBoundsOrigBuf.buffer);
    }
    info.numTriangles = static_cast<uint32_t>(meshView.triangleCount());
    primInfo.emplace_back(info);

    vkScene.numTriangles.push_back(info.numTriangles);
    vkScene.totalTriangles += meshView.triangleCount();

    primIdx++;
  }

  // Creating the buffer of all primitive information
  vkScene.primInfoBuf = m_alloc.createBuffer(cmdBuf, primInfo, usageFlag);
  LOGI("%7.2fms\n", sw.elapsed());
}

// Temporary struct passed to per-thread vertex generation functions below.
struct MakeVertexData
{
  const MeshView&                               meshView;
  const meshops::OpBake_heightmap&              dispInfo;
  const HeightMap&                              heightmap;
  const meshops::ConstArrayView<nvmath::vec3f>& directions;
  const micromesh::ArrayInfo_uint16&            inputTriangleSubdivLevels;
  const micromesh::MeshTopology&                topology;
  const size_t                                  maxAdjacentVertices;
  const std::vector<uint32_t>&                  selectedBaseTriangles;

  // Output compressed vertices
  std::vector<CompressedVertex>& cvertices;

  // Per-thread temporary buffer for sanitization, each containing maxAdjacentVertices entries
  std::vector<std::vector<micromesh::MicroVertexInfo>> sanitizeBuffers;

  void setupSanitizationBuffers(uint32_t threadCount)
  {
    sanitizeBuffers.resize(threadCount);
    for(auto& b : sanitizeBuffers)
      b.resize(maxAdjacentVertices);
  }
};

static inline Vertex makeVertex(const MakeVertexData* makeVertexData, uint32_t triIndex, nvmath::vec3f baryCoord)
{
  nvmath::vec3ui triVertices = makeVertexData->meshView.triangleVertices[triIndex];
  stabilizeTriangleVerticesOrder(triVertices, baryCoord);
  Vertex result{};
  if(!makeVertexData->meshView.vertexNormals.empty())
    result.normal = baryInterp(makeVertexData->meshView.vertexNormals, triVertices, baryCoord);
  if(!makeVertexData->meshView.vertexTangents.empty())
    result.tangent = baryInterp(makeVertexData->meshView.vertexTangents, triVertices, baryCoord);
  if(!makeVertexData->meshView.vertexTexcoords0.empty())
    result.texCoord = baryInterp(makeVertexData->meshView.vertexTexcoords0, triVertices, baryCoord);

  if(makeVertexData->dispInfo.pnTriangles)
  {
    const vec3f& v0 = makeVertexData->meshView.vertexPositions[triVertices.x];
    const vec3f& v1 = makeVertexData->meshView.vertexPositions[triVertices.y];
    const vec3f& v2 = makeVertexData->meshView.vertexPositions[triVertices.z];
    const vec3f& n0 = makeVertexData->directions[triVertices.x];
    const vec3f& n1 = makeVertexData->directions[triVertices.y];
    const vec3f& n2 = makeVertexData->directions[triVertices.z];
    PNTriangles  pnt(v0, v1, v2, n0, n1, n2);
    result.position              = pnt.position(baryCoord);
    result.displacementDirection = pnt.normal(baryCoord);
  }
  else
  {
    result.position              = baryInterp(makeVertexData->meshView.vertexPositions, triVertices, baryCoord);
    result.displacementDirection = baryInterp(makeVertexData->directions, triVertices, baryCoord);
  }

  if(makeVertexData->dispInfo.normalizeDirections)
    result.displacementDirection = glm::normalize(glm::vec3(result.displacementDirection));
  if(makeVertexData->dispInfo.texture)
  {
    float displacement = makeVertexData->heightmap.bilinearFetch(result.texCoord);
    displacement       = displacement * makeVertexData->dispInfo.scale + makeVertexData->dispInfo.bias;
    result.position += result.displacementDirection * displacement;
  }
  return result;
};

static inline Vertex makeSanitizedVertex(MakeVertexData* makeVertexData, const micromesh::VertexGenerateInfo* vertexInfo, uint32_t threadIndex)
{
  std::vector<micromesh::MicroVertexInfo>& sanitizeBuffer = makeVertexData->sanitizeBuffers[threadIndex];

  uint32_t meshTriangleIndex = makeVertexData->selectedBaseTriangles[vertexInfo->meshTriangleIndex];

  micromesh::MicroVertexInfo queryVertex{meshTriangleIndex, vertexInfo->vertexUV};

  uint32_t count = micromesh::micromeshMeshTopologyGetVertexSanitizationList(
      &makeVertexData->topology, &makeVertexData->inputTriangleSubdivLevels, nullptr, queryVertex,
      (uint32_t)sanitizeBuffer.size(), sanitizeBuffer.data());
  assert(count <= sanitizeBuffer.size());
  Vertex avgVertex;
  for(uint32_t i = 0; i < count; ++i)
  {
    uint32_t subdivLevel =
        micromesh::arrayGetV<uint16_t>(makeVertexData->inputTriangleSubdivLevels, sanitizeBuffer[i].triangleIndex);
    micromesh::BaryWUV_float otherWUVfloat = micromesh::baryUVtoWUV_float(sanitizeBuffer[i].vertexUV, subdivLevel);
    nvmath::vec3f            baryCoord(otherWUVfloat.w, otherWUVfloat.u, otherWUVfloat.v);
    Vertex                   otherVertex = makeVertex(makeVertexData, sanitizeBuffer[i].triangleIndex, baryCoord);
    if(i == 0)
      avgVertex = otherVertex;
    else
    {
      avgVertex.position += otherVertex.position;
      avgVertex.normal += otherVertex.normal;
    }
  }
  if(count == 0)
  {
    glm::vec3 baryCoord(vertexInfo->vertexWUVfloat.w, vertexInfo->vertexWUVfloat.u, vertexInfo->vertexWUVfloat.v);
    avgVertex = makeVertex(makeVertexData, meshTriangleIndex, baryCoord);
  }
  else
  {
    avgVertex.position /= (float)count;
    avgVertex.normal /= (float)count;
  }
  return avgVertex;
};


static inline uint32_t generateTessellatedVertex(const micromesh::VertexGenerateInfo* vertexInfo,
                                                 micromesh::VertexDedup               dedupState,
                                                 uint32_t                             threadIndex,
                                                 void*                                beginResult,
                                                 void*                                userData)
{
  assert(userData);
  auto     makeVertexData = reinterpret_cast<MakeVertexData*>(userData);
  Vertex   vertex         = makeSanitizedVertex(makeVertexData, vertexInfo, threadIndex);
  uint32_t index;
  if(dedupState)
  {
    micromeshVertexDedupAppendAttribute(dedupState, sizeof(vertex), &vertex);
    index = micromeshVertexDedupGetIndex(dedupState);
  }
  else
  {
    index = vertexInfo->nonDedupIndex;
  }
  makeVertexData->cvertices[index] = compressVertex(vertex);
  return index;
};
//--------------------------------------------------------------------------------------------------
// Creating information per primitive
// - Create a buffer of Vertex and Index for each primitive
// - Each primInfo has a reference to the vertex and index buffer, and which material id it uses
//
bool BakerVK::createTessellatedVertexBuffer(const meshops::OpBake_input& info,
                                            VkCommandBuffer              cmdBuf,
                                            const meshops::MeshView&     meshView,
                                            SceneVK&                     vkScene,
                                            const GeometryBatch&         batch,
                                            int                          maxSubdivLevel)
{
  nvh::Stopwatch sw;
  LOGI("- Create Vertex Buffer ");

  std::vector<PrimMeshInfo> primInfo;  // The array of all primitive information
  auto                      usageFlag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                   | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

  for(auto primIdx = 0; primIdx < 1; ++primIdx)
  {
    auto&     dispInfo   = info.referenceMeshHeightmap;
    auto      directions = dispInfo.usesVertexNormalsAsDirections ? meshView.vertexNormals : meshView.vertexDirections;
    HeightMap heightmap;
    if(dispInfo.texture)
    {
      heightmap = HeightMap(dispInfo.texture->m_config.width, dispInfo.texture->m_config.height,
                            reinterpret_cast<const float*>(dispInfo.texture->getImageData()));
    }

    micromesh::Result       result;
    micromesh::MeshTopology topology = *info.referenceMeshTopology;

    // Grow the selected triangles by one. A MeshTopology structure is needed to
    // generate indices for the new subset of mesh triangles.
    std::vector<uint32_t> selectedBaseTriangles;
    {
      micromesh::OpGrowTriangleSelection_input input;
      input.topology      = &topology;
      input.triangleFirst = batch.triangleOffset;
      input.triangleCount = batch.triangleCount;

      selectedBaseTriangles.resize(topology.triangleVertices.count);

      micromesh::OpGrowTriangleSelection_output output;
      micromesh::arraySetDataVec(output.triangleSelection, selectedBaseTriangles);

      result = micromesh::micromeshOpGrowTriangleSelection(m_micromeshContext, &input, &output);
      assert(result == micromesh::Result::eSuccess);

      selectedBaseTriangles.resize(output.triangleSelectionCount);
    }

    // Extract only the subdiv levels for the selected triangles
    std::vector<uint16_t> selectedSubdivLevels(selectedBaseTriangles.size());
    std::vector<uint8_t>  selectedEdgeFlags(selectedBaseTriangles.size());
    for(size_t i = 0; i < selectedBaseTriangles.size(); ++i)
    {
      selectedSubdivLevels[i] = meshView.triangleSubdivisionLevels[selectedBaseTriangles[i]];
      selectedEdgeFlags[i]    = meshView.trianglePrimitiveFlags[selectedBaseTriangles[i]];
    }

    // Tessellation output
    std::vector<CompressedVertex>           cvertices;
    std::vector<micromesh::Vector_uint32_3> triangleVertices;


    micromesh::ArrayInfo_uint16 inputTriangleSubdivLevels;
    // Need to cast because ArrayInfo does not have a const void pointer
    micromesh::arraySetDataVec(inputTriangleSubdivLevels, ArrayViewConstCast(meshView.triangleSubdivisionLevels));
    size_t maxAdjacentVertices(std::max(topology.maxEdgeTriangleValence, topology.maxVertexTriangleValence));

    MakeVertexData makeVertexData{
        meshView, dispInfo, heightmap, directions, inputTriangleSubdivLevels, topology, maxAdjacentVertices, selectedBaseTriangles, cvertices};
    makeVertexData.setupSanitizationBuffers(m_threadCount);

    // Tessellate the selected triangles
    {
      micromesh::OpTessellateMesh_input input;
      input.useVertexDeduplication = true;
      input.maxSubdivLevel         = maxSubdivLevel;
      input.userData               = &makeVertexData;
      input.pfnGenerateVertex      = generateTessellatedVertex;
      micromesh::arraySetDataVec(input.meshTriangleSubdivLevels, selectedSubdivLevels);
      micromesh::arraySetDataVec(input.meshTrianglePrimitiveFlags, selectedEdgeFlags);

      micromesh::OpTessellateMesh_output output;
      result = micromesh::micromeshOpTessellateMeshBegin(m_micromeshContext, &input, &output);
      assert(result == micromesh::Result::eSuccess);

      cvertices.resize(output.vertexCount);
      triangleVertices.resize(output.meshTriangleVertices.count);
      output.meshTriangleVertices.data = triangleVertices.data();

      result = micromesh::micromeshOpTessellateMeshEnd(m_micromeshContext, &input, &output);
      assert(result == micromesh::Result::eSuccess);

      // Some vertices may have been merged if useVertexDeduplication is set
      cvertices.resize(output.vertexCount);
    }

    // Add the total triangles to the "Create Vertex Buffer" status line.
    // Flush in case m_alloc.createBuffer fails, so we know the reason.
    LOGI("(triangles: %zu) ", triangleVertices.size());
    fflush(stdout);

    if(cvertices.empty() || triangleVertices.empty())
    {
      LOGW("\nWarning: Skipping empty highres batch %i\n", batch.batchIndex + 1);
      return false;
    }

    if(info.settings.debugDisplacedReferenceMeshCallback)
    {
      LOGI("\n");  // Break the "Create Vertex Buffer" line for logging in saveSimpleGeometry()

      // Convert to array of uncompressed vertices
      std::vector<Vertex> vertices(cvertices.size());
      for(size_t i = 0; i < cvertices.size(); i++)
      {
        vertices[i] = decompressVertex(cvertices[i]);
      }

      // Wrap input data in a MeshView with one slice referring to the lot
      meshops::MeshView meshView;
      meshView.triangleVertices = ArrayView<nvmath::vec3ui>(ArrayView(triangleVertices));
      meshView.vertexPositions =
          ArrayView(reinterpret_cast<nvmath::vec3f*>(&vertices[0].position), vertices.size(), sizeof(Vertex));
      meshView.vertexNormals = ArrayView(reinterpret_cast<nvmath::vec3f*>(&vertices[0].normal), vertices.size(), sizeof(Vertex));
      meshView.vertexTangents =
          ArrayView(reinterpret_cast<nvmath::vec4f*>(&vertices[0].tangent), vertices.size(), sizeof(Vertex));
      meshView.vertexDirections =
          ArrayView(reinterpret_cast<nvmath::vec3f*>(&vertices[0].displacementDirection), vertices.size(), sizeof(Vertex));

      info.settings.debugDisplacedReferenceMeshCallback(meshView, &info.baseMeshTransform, batch.batchIndex,
                                                        batch.totalBatches, info.settings.debugDisplacedReferenceMeshUserPtr);
    }

    auto v_buffer = m_alloc.createBuffer(cmdBuf, cvertices, usageFlag | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    vkScene.verticesBuf.push_back(v_buffer);
    vkScene.numVertices.push_back((uint32_t)cvertices.size());

    auto i_buffer = m_alloc.createBuffer(cmdBuf, triangleVertices, usageFlag | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    vkScene.indicesBuf.emplace_back(i_buffer);

    // Compute the max absolute displacement from the heightmap
    float maxDisp = 0.0f;
    if(dispInfo.texture)
    {
      nvmath::mat4f referenceMeshTransform(&info.referenceMeshTransform.columns[0].x);
      for(auto& direction : directions)
      {
        // The conservative (min, max) heightmap displacement is direction * (0 * scale + bias, 1 * scale + bias),
        // converted to world space. Since these scale the direction vector, its length in world space can be reused.
        float l = nvmath::length(referenceMeshTransform.get_rot_mat3()
                                 * (dispInfo.normalizeDirections ? nvmath::normalize(direction) : direction));
        maxDisp = std::max(maxDisp, std::abs(l * dispInfo.bias));
        maxDisp = std::max(maxDisp, std::abs(l * (dispInfo.scale + dispInfo.bias)));
      }
    }

    // Primitive information, material Id and addresses of buffers
    PrimMeshInfo info{};
    info.materialIndex     = 0;
    info.vertexAddress     = nvvk::getBufferDeviceAddress(m_device, v_buffer.buffer);
    info.indexAddress      = nvvk::getBufferDeviceAddress(m_device, i_buffer.buffer);
    info.numTriangles      = (uint32_t)triangleVertices.size();
    info.maxDisplacementWs = maxDisp;
    primInfo.emplace_back(info);

    vkScene.numTriangles.push_back(info.numTriangles);
    vkScene.totalTriangles += info.numTriangles;
  }

  // Creating the buffer of all primitive information
  vkScene.primInfoBuf = m_alloc.createBuffer(cmdBuf, primInfo, usageFlag);
  LOGI("%7.2fms\n", sw.elapsed());
  return true;
}


//--------------------------------------------------------------------------------------------------
// Converting a GLTF primitive in the Raytracing Geometry used for the BLAS
//
auto BakerVK::primitiveToGeometry(const meshops::MeshView& prim,
                                  VkDeviceAddress          vertexAddress,
                                  VkDeviceAddress          indexAddress,
                                  uint32_t                 numVertices,
                                  uint32_t                 numTriangles)
{
  // Describe buffer as array of VertexObj.
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
  triangles.vertexFormat             = VK_FORMAT_R32G32B32A32_SFLOAT;  // vec3 vertex position data.
  triangles.vertexData.deviceAddress = vertexAddress;
  triangles.vertexStride             = sizeof(CompressedVertex);
  triangles.indexType                = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress  = indexAddress;
  triangles.maxVertex                = numVertices;
  //triangles.transformData; // Identity

  // Identify the above data as containing opaque triangles.
  VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  asGeom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  asGeom.flags              = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
  asGeom.geometry.triangles = triangles;

  VkAccelerationStructureBuildRangeInfoKHR offset{};
  offset.firstVertex     = 0;
  offset.primitiveCount  = numTriangles;
  offset.primitiveOffset = 0;
  offset.transformOffset = 0;

  // Our blas is made from only one geometry, but could be made of many geometries
  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);

  return input;
}


//--------------------------------------------------------------------------------------------------
// Create all bottom level acceleration structures (BLAS)
//
void BakerVK::createBottomLevelAS(const meshops::MeshView& meshView, const SceneVK& vkScene, const GeometryBatch& batch)
{
  ScopedTimer t("Create Bottom Level AS");

  // BLAS - Storing each primitive in a geometry, but we don't batch more than one together
  std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;

  // Only index buffers in this batch are created
  uint32_t p_idx = 0;
  uint32_t v_idx = 0;

  auto vertexAddress = nvvk::getBufferDeviceAddress(m_device, vkScene.verticesBuf[v_idx].buffer);
  auto indexAddress  = nvvk::getBufferDeviceAddress(m_device, vkScene.indicesBuf[p_idx].buffer);

  auto geo = primitiveToGeometry(meshView, vertexAddress, indexAddress, vkScene.numVertices[v_idx], vkScene.numTriangles[p_idx]);
  allBlas.push_back({geo});
  m_rtBuilder.buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
// Create the top level acceleration structures, referencing all BLAS
//
void BakerVK::createTopLevelAS(const meshops::OpBake_input& info, const meshops::MeshView& meshView, const GeometryBatch& batch)
{
  ScopedTimer t("Create Top Level AS");

  std::vector<VkAccelerationStructureInstanceKHR> tlas;
  tlas.reserve(1);

  uint32_t primMeshID = 0;

  // Use the transform from the mesh's first instance
  nvmath::mat4f referenceMeshTransform(&info.referenceMeshTransform.columns[0].x);

  uint32_t blasId = 0;

  VkGeometryInstanceFlagsKHR flags{};
  // flags |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;                  // All opaque (faster)
  // flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;  // double sided

  VkAccelerationStructureInstanceKHR rayInst{};
  rayInst.transform           = nvvk::toTransformMatrixKHR(referenceMeshTransform);  // Position of the instance
  rayInst.instanceCustomIndex = primMeshID & 0xFFF;                                  // gl_InstanceCustomIndexEXT
  rayInst.accelerationStructureReference         = m_rtBuilder.getBlasDeviceAddress(blasId);
  rayInst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
  rayInst.flags                                  = flags & 0xFF;
  rayInst.mask                                   = 0xFF;

  tlas.emplace_back(rayInst);
  m_rtBuilder.buildTlas(tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
// Destroy Vulkan resources
//
void BakerVK::destroy()
{
  destroyVkScene(m_lowVk);

  m_alloc.destroy(m_distanceBuf);
  for(auto& b : m_baryCoordBuf)
    m_alloc.destroy(b);
  m_alloc.destroy(m_trianglesBuf);
  m_alloc.destroy(m_triangleMinMaxBuf);
  m_alloc.destroy(m_directionBoundsBuf);
  m_alloc.destroy(m_directionBoundsOrigBuf);
}

//--------------------------------------------------------------------------------------------------
// Destroy local scene resources
//
void BakerVK::destroyVkScene(SceneVK& vkScene)
{
  for(auto& v : vkScene.verticesBuf)
    m_alloc.destroy(v);
  vkScene.verticesBuf.clear();

  for(auto& i : vkScene.indicesBuf)
    m_alloc.destroy(i);
  vkScene.indicesBuf.clear();

  m_alloc.destroy(vkScene.primInfoBuf);

  // Clear all members
  vkScene = {};
}


//--------------------------------------------------------------------------------------------------
//
//
void BakerVK::createBakerPipeline()
{
  ScopedTimer t("Create Baker Pipeline");

  // Descriptors
  auto& d = m_dContainer;
  d.binder.clear();
  d.binder.addBinding(+SceneBindings::eSceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  d.binder.addBinding(+SceneBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
  d.layout = d.binder.createLayout(m_device);
  d.pool   = d.binder.createPool(m_device, 1);
  d.set    = nvvk::allocateDescriptorSet(m_device, d.pool, d.layout);

  // Writing to descriptors
  const auto&                                  tlas = m_rtBuilder.getAccelerationStructure();
  std::vector<VkWriteDescriptorSet>            writes;
  VkDescriptorBufferInfo                       b0{m_sceneDescBuf.buffer, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSetAccelerationStructureKHR t0{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                                                  nullptr, 1, &tlas};
  writes.emplace_back(d.binder.makeWrite(d.set, +SceneBindings::eSceneDesc, &b0));
  writes.emplace_back(d.binder.makeWrite(d.set, +SceneBindings::eTlas, &t0));
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  // Pipeline
  auto& p = m_pContainer;

  nvvk::Specialization specialization;
  specialization.add({{0, 0}});

  // Push constants in the compute shader
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushHighLow)};

  // Creating the pipeline layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &d.layout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &p.layout);

  // Shader spirv
  std::vector<uint32_t> compShader(std::begin(bary_trace_comp), std::end(bary_trace_comp));

  // Baker compute shader
  VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stageInfo.stage               = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.module              = nvvk::createShaderModule(m_device, bary_trace_comp, sizeof(bary_trace_comp));
  stageInfo.pName               = "main";
  stageInfo.pSpecializationInfo = specialization.getSpecialization();

  VkComputePipelineCreateInfo compInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  compInfo.layout = p.layout;
  compInfo.stage  = stageInfo;

  vkCreateComputePipelines(m_device, {}, 1, &compInfo, nullptr, &p.pipeline);

  vkDestroyShaderModule(m_device, stageInfo.module, nullptr);
}

void BakerVK::destroyBakerPipeline()
{
  auto& p = m_pContainer;
  vkDestroyPipeline(m_device, p.pipeline, nullptr);
  vkDestroyPipelineLayout(m_device, p.layout, nullptr);
  auto& d = m_dContainer;
  vkDestroyDescriptorPool(m_device, d.pool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, d.layout, nullptr);
}

void BakerVK::createResamplerPipeline(const std::vector<VkDescriptorImageInfo>& inputTextures,
                                      const std::vector<VkDescriptorImageInfo>& outputTextures,
                                      const std::vector<VkDescriptorImageInfo>& distanceTextures)
{
  ScopedTimer t("Create Resampler Pipeline");
  assert(inputTextures.size() && inputTextures.size() < MAX_RESAMPLE_TEXTURES);
  assert(inputTextures.size() == outputTextures.size());
  assert(inputTextures.size() == distanceTextures.size());

  // Bind input textures after removing any that are null. runResampler() builds
  // ResampleTextureInfo::inputIndex to reconstruct the mapping between input
  // and output textures. This is less convoluted than making vulkan accept null
  // descriptors.
  auto validInputTextures = inputTextures;
  validInputTextures.erase(std::remove_if(validInputTextures.begin(), validInputTextures.end(),
                                          [](const auto& texture) { return texture.imageView == VK_NULL_HANDLE; }),
                           validInputTextures.end());

  // Descriptors
  auto& d = m_resampling.descriptor;
  d.binder.clear();
  d.binder.addBinding(+SceneBindings::eSceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  d.binder.addBinding(+SceneBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
  d.binder.addBinding(+SceneBindings::eTexturesIn, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                      (uint32_t)validInputTextures.size(), VK_SHADER_STAGE_ALL);
  d.binder.addBinding(+SceneBindings::eTexturesOut, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, (uint32_t)outputTextures.size(),
                      VK_SHADER_STAGE_ALL);
  d.binder.addBinding(+SceneBindings::eTexturesDist, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                      (uint32_t)distanceTextures.size(), VK_SHADER_STAGE_ALL);

  d.layout = d.binder.createLayout(m_device);
  d.pool   = d.binder.createPool(m_device, 1);
  d.set    = nvvk::allocateDescriptorSet(m_device, d.pool, d.layout);

  // Writing to descriptors
  const auto&                                  tlas = m_rtBuilder.getAccelerationStructure();
  std::vector<VkWriteDescriptorSet>            writes;
  VkDescriptorBufferInfo                       b0{m_sceneDescBuf.buffer, 0, VK_WHOLE_SIZE};
  VkWriteDescriptorSetAccelerationStructureKHR t0{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                                                  nullptr, 1, &tlas};
  writes.emplace_back(d.binder.makeWrite(d.set, +SceneBindings::eSceneDesc, &b0));
  writes.emplace_back(d.binder.makeWrite(d.set, +SceneBindings::eTlas, &t0));
  if(validInputTextures.size())
  {
    writes.emplace_back(d.binder.makeWriteArray(d.set, +SceneBindings::eTexturesIn, validInputTextures.data()));
  }
  writes.emplace_back(d.binder.makeWriteArray(d.set, +SceneBindings::eTexturesOut, outputTextures.data()));
  writes.emplace_back(d.binder.makeWriteArray(d.set, +SceneBindings::eTexturesDist, distanceTextures.data()));
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  auto& p = m_resampling.pipeline;

  // Same push constants as the baker
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof(PushHighLow)};

  // Creating the pipeline layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &d.layout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &p.layout);

  // Resampling shader
  std::vector<uint32_t>                   resampleShaderVert(std::begin(resample_vert), std::end(resample_vert));
  std::vector<uint32_t>                   resampleShaderFrag(std::begin(resample_frag), std::end(resample_frag));
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, p.layout, {});
  gpb.rasterizationState.cullMode = VK_CULL_MODE_NONE;
  gpb.addShader(resampleShaderVert, VK_SHADER_STAGE_VERTEX_BIT);
  gpb.addShader(resampleShaderFrag, VK_SHADER_STAGE_FRAGMENT_BIT);
  p.pipeline = gpb.createPipeline();
}

void BakerVK::destroyResamplerPipeline()
{
  auto& p = m_resampling.pipeline;
  vkDestroyPipeline(m_device, p.pipeline, nullptr);
  vkDestroyPipelineLayout(m_device, p.layout, nullptr);
  auto& d = m_resampling.descriptor;
  vkDestroyDescriptorPool(m_device, d.pool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, d.layout, nullptr);
}

//--------------------------------------------------------------------------------------------------
//
//
void BakerVK::runComputePass(const meshops::OpBake_input& info, bool finalBatch)
{
  ScopedTimer t("Run Compute Pass");

  nvvk::CommandPool cmdPool(m_device, m_queueC.familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, m_queueC.queue);

  auto& p = m_pContainer;
  auto& d = m_dContainer;

  // Figuring out the offset for the distance vector
  std::vector<uint32_t> distOffsets;
  uint32_t              offset = 0;
  distOffsets.push_back(offset);

  // Use the transform from the mesh's first instance
  m_push.objectToWorld  = nvmath::mat4f(&info.baseMeshTransform.columns[0].x);
  m_push.worldToObject  = nvmath::invert(m_push.objectToWorld);
  m_push.primMeshID     = 0;
  m_push.distOffset     = distOffsets[0];
  m_push.lastBatch      = finalBatch ? 1 : 0;
  uint32_t numTriangles = static_cast<uint32_t>(info.baseMeshView.triangleCount());

  // Split up the draw calls into batches to avoid TDR/channel resets on long running jobs.
  uint32_t trianglesPerBatch = 1000;
  for(uint32_t startTriangle = 0; startTriangle < numTriangles; startTriangle += trianglesPerBatch)
  {
    VkCommandBuffer cmdBuf = cmdPool.createCommandBuffer();
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, p.layout, 0, 1, &d.set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);

    uint32_t batchTriangles     = std::min(trianglesPerBatch, numTriangles - startTriangle);
    m_push.baryTraceBatchOffset = startTriangle;
    vkCmdPushConstants(cmdBuf, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushHighLow), &m_push);
    vkCmdDispatch(cmdBuf, batchTriangles, 1, 1);
    cmdPool.submit(1, &cmdBuf);
  }

  VkCommandBuffer cmdBuf = cmdPool.createCommandBuffer();
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0,
                       nullptr, 0, nullptr);
  cmdPool.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
}

void BakerVK::runResampler(const meshops::OpBake_input& info, ArrayView<meshops::Texture> outputTextures)
{
  ScopedTimer t("Run Resampler");
  assert(outputTextures.size());

  glm::uvec2                     maxResolution{};
  std::unordered_set<glm::uvec2> uniqueResolutions;
  for(auto& texture : outputTextures)
  {
    glm::uvec2 resolution{texture->m_vk.imageCreateInfo.extent.width, texture->m_vk.imageCreateInfo.extent.height};
    uniqueResolutions.insert(resolution);
    maxResolution = glm::max(maxResolution, resolution);
  }

  auto& p = m_resampling.pipeline;
  auto& d = m_resampling.descriptor;

  m_push.objectToWorld         = nvmath::mat4f(&info.baseMeshTransform.columns[0].x);
  m_push.worldToObject         = nvmath::invert(m_push.objectToWorld);
  m_push.primMeshID            = 0;
  m_push.numResampleTextures   = (uint32_t)outputTextures.size();
  bool     generatingHeightmap = false;
  uint32_t nextInputIndex      = 0;
  for(uint32_t i = 0; i < m_push.numResampleTextures; ++i)
  {
    if(info.resamplerInput[i].texture != nullptr)
    {
      m_push.textureInfo[i].setInputIndex(nextInputIndex);
      nextInputIndex++;
    }

    // Set texture type
    m_push.textureInfo[i].setTextureType(uint32_t(info.resamplerInput[i].textureType));
    generatingHeightmap = generatingHeightmap || info.resamplerInput[i].textureType == TextureType::eHeightMap;
  }

  // If we're generating a heightmap during resampling, we should scale the
  // output by the bounds. This would need a second pass to first compute the
  // values we would write. Instead, we can approximate the bounds by using the
  // already-computed micromesh displacement values. These do get computed in
  // meshops_bake.cpp, but this operation is not that common anyway.
  m_push.globalMinMax = nvmath::vec2f(0.0f, 1.0f);
  if(generatingHeightmap)
  {
    auto minMaxs = ArrayView(static_cast<nvmath::vec2f*>(m_alloc.map(m_triangleMinMaxBuf)), info.baseMeshView.triangleCount());
    if(!getGlobalMinMax(minMaxs, m_push.globalMinMax, false, 0))
    {
      m_push.globalMinMax = {0.0f, 1.0f};
    }
    m_alloc.unmap(m_triangleMinMaxBuf);
  }

  // Create a list of resolutions. Each geometry instance will be rasterized at
  // this resolution by scaling vertices relative to the max resolution. The
  // order doesn't matter.
  size_t nextResolution = 0;
  for(auto resolution : uniqueResolutions)
    m_push.resampleInstanceResolutions[nextResolution++] = (resolution.y << 16) | resolution.x;
  m_push.resampleMaxResolution = (maxResolution.y << 16) | maxResolution.x;
  uint32_t instances           = uint32_t(uniqueResolutions.size());

  nvvk::CommandPool cmdPool(m_device, m_queueGCT.familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, m_queueGCT.queue);

  // Indices are fetched by the vertex shader
  // Split up the draw calls into batches to avoid TDR/channel resets on long running jobs.
  // TODO: use proper graphics pipeline for vertex reuse
  uint32_t trianglesPerBatch = 1000;
  uint32_t totalTriangles    = static_cast<uint32_t>(info.baseMeshView.triangleCount());
  for(uint32_t triOffset = 0; triOffset < totalTriangles; triOffset += trianglesPerBatch)
  {
    VkCommandBuffer cmdBuf = cmdPool.createCommandBuffer();

    VkRenderingInfoKHR renderInfo{VK_STRUCTURE_TYPE_RENDERING_INFO_KHR};
    renderInfo.renderArea = {{0, 0}, {maxResolution.x, maxResolution.y}};
    renderInfo.layerCount = 1;
    vkCmdBeginRendering(cmdBuf, &renderInfo);

    // Dynamic Viewport
    VkViewport viewport{0.0f, 0.0f, (float)maxResolution.x, (float)maxResolution.y, 0.0f, 1.0f};
    vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
    VkRect2D scissor{{0, 0}, {maxResolution.x, maxResolution.y}};
    vkCmdSetScissor(cmdBuf, 0, 1, &scissor);

    vkCmdPushConstants(cmdBuf, p.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof(PushHighLow), &m_push);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, p.layout, 0, 1, &d.set, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipeline);

    uint32_t batchTriangles = std::min(trianglesPerBatch, totalTriangles - triOffset);
    vkCmdDraw(cmdBuf, batchTriangles * 3, instances, triOffset * 3, 0);

    vkCmdEndRendering(cmdBuf);
    cmdPool.submit(1, &cmdBuf);
  }

  VkCommandBuffer cmdBuf = cmdPool.createCommandBuffer();
#if 0
  // Enable to debug resampler output, clobbering all textures with solid red.
  VkClearColorValue clearColor{1, 0, 0, 1};
  VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  for(auto& texture : outputTextures)
    vkCmdClearColorImage(cmdBuf, texture->m_vk.image, texture->m_vk.imageLayout, &clearColor, 1, &range);
#endif
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0, 1, &barrier, 0, nullptr, 0, nullptr);

  cmdPool.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
}

void BakerVK::fitDirectionBounds(const meshops::OpBake_input& info, MutableArrayView<float> distances)
{
  // TODO: implement nvvk::ResourceAllocator::map() with a range
  ScopedTimer                 t("Fit min/max bounds");
  auto&                       meshView = info.baseMeshView;
  micromesh::MeshTopologyUtil topo(*info.baseMeshTopology);
  auto minMaxs = ArrayView(static_cast<nvmath::vec2f*>(m_alloc.map(m_triangleMinMaxBuf)), meshView.triangleCount());
  auto directionBounds = ArrayView(static_cast<nvmath::vec2f*>(m_alloc.map(m_directionBoundsBuf)), meshView.vertexCount());

#if 0
  // Brute force min/max
  {
    auto distancesGpu = static_cast<float*>(m_alloc.map(m_distanceBuf));
    size_t dist = 0;
    for(size_t i = 0; i < info.baseMeshView.triangleSubdivisionLevels.size(); ++i)
    {
      uint32_t microVertices = bary::baryValueFrequencyGetCount(bary::ValueFrequency::ePerVertex, info.baseMeshView.triangleSubdivisionLevels[i]);
      auto minMax = std::minmax_element(distancesGpu + dist, distancesGpu + dist + microVertices);
      minMaxs[i] = {*minMax.first, *minMax.second};
      dist += microVertices;
    }
    m_alloc.unmap(m_distanceBuf);
  }
#endif

  // Compute min/max distances for each vertex as the min/max distance in adjacent triangles. The adjacent triangles are
  // only position-unique, which may produce more relaxed bounds than if a position+direction unique topology were
  // created.
  nvh::parallel_batches(
      meshView.vertexCount(),
      [&](uint64_t vertIdx) {
        nvmath::vec2f adjMinMax{std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()};
        nvmath::vec3f dir          = nvmath::normalize(meshView.vertexDirections[vertIdx]);
        float         parallelDirs = 1.0f;
        for(auto& triIdx : makeArrayView(topo.getVertexTrianglesArray(uint32_t(vertIdx))))
        {
          assert(minMaxs[triIdx].x <= minMaxs[triIdx].y);
          adjMinMax.x = std::min(adjMinMax.x, minMaxs[triIdx].x);
          adjMinMax.y = std::max(adjMinMax.y, minMaxs[triIdx].y);

          micromesh::Vector_uint32_3 tri = topo.getTriangleVertices(triIdx);
          parallelDirs = std::min(parallelDirs, nvmath::dot(dir, nvmath::normalize(meshView.vertexDirections[tri.x])));
          parallelDirs = std::min(parallelDirs, nvmath::dot(dir, nvmath::normalize(meshView.vertexDirections[tri.y])));
          parallelDirs = std::min(parallelDirs, nvmath::dot(dir, nvmath::normalize(meshView.vertexDirections[tri.z])));
        }

        // If the direction vectors of neighboring triangles don't align well,
        // bounds fitting can be unstable and actually produce very large
        // bounds.
        if(parallelDirs < -0.49f)  // discard if outside acos(-0.49) ~ 120 degrees
          return;

        // Update the bounds based on the new min/maxes. The min/max values are distance relative to within the bounds,
        // i.e. a uniDirectional min of 0.0 and a max of 1.0 means use the current bounds as-is. These new bounds will
        // be used in a second pass of the baker, since points at the displacement bounds form new direction vectors
        // when interpolated. Depending on info.uniDirectional, values may be negative and outside the segment. Nothing
        // special is needed for bi-directional tracing here since the trace bounds are found by intersecting the
        // initial displacement bounds. In that case, the first iteration will produce values near [-1, 1] and
        // subsequent passes should produce [0, 1].
        const float epsilon = 1e-6F;
        BiasScalef  biasScale(directionBounds[vertIdx]);
        biasScale *= BiasScalef::minmax_unit(adjMinMax);
        directionBounds[vertIdx] = {biasScale.bias, std::max(epsilon, biasScale.scale)};
      },
      m_threadCount);

  // Copy direction bounds from position-unique/watertight vertices.
  auto wtTriangleVertices = ArrayView<nvmath::vec3ui>(makeArrayView(info.baseMeshTopology->triangleVertices));
  nvh::parallel_batches(
      meshView.triangleCount(),
      [&](uint64_t triIdx) {
        auto& tri   = meshView.triangleVertices[triIdx];
        auto  triWt = wtTriangleVertices[triIdx];
        if(tri != nvmath::vec3ui(triWt))
        {
          directionBounds[tri.x] = directionBounds[triWt.x];
          directionBounds[tri.y] = directionBounds[triWt.y];
          directionBounds[tri.z] = directionBounds[triWt.z];
        }
      },
      m_threadCount);

  // Restore distances for the next pass
  auto distancesGpu = ArrayView(static_cast<float*>(m_alloc.map(m_distanceBuf)), distances.size());
  std::fill(distancesGpu.data(), distancesGpu.data() + distancesGpu.size(), std::numeric_limits<float>::max());
  m_alloc.unmap(m_distanceBuf);

  // Restore min/maxes for the next pass
  std::fill(minMaxs.begin(), minMaxs.end(),
            nvmath::vec2f(std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()));

  m_alloc.unmap(m_directionBoundsBuf);
  m_alloc.unmap(m_triangleMinMaxBuf);
}

bool BakerVK::getGlobalMinMax(ArrayView<const nvmath::vec2f> minMaxs, nvmath::vec2f& globalMinMax, bool filterZeroToOne, const uint32_t maxFilterWarnings)
{
  // Compute min/max distances for the whole scene. Since direction bounds
  // fitting is a numerical root finding method, there is always a chance some
  // displacement values are outside the 0-1 range and need filtering out.
  uint32_t    filteredTriangles = 0;
  const float filterThreshold   = 0.1f;
  globalMinMax                  = {std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest()};
  for(size_t i = 0; i < minMaxs.size(); ++i)
  {
    assert(minMaxs[i].x <= minMaxs[i].y);

    // With per-triangle fitting, values just outside [0, 1] are expected, but only just
    if(filterZeroToOne && (minMaxs[i].x < -filterThreshold || minMaxs[i].y > 1.0f + filterThreshold))
    {
      ++filteredTriangles;
      if(filteredTriangles <= maxFilterWarnings)
      {
        LOGW("Warning: Clamping values for triangle %zu with bad range [%f, %f].%s\n", i, minMaxs[i].x, minMaxs[i].y,
             filteredTriangles == maxFilterWarnings ? " Last report." : "");
      }
      continue;
    }

    globalMinMax.x = std::min(globalMinMax.x, minMaxs[i].x);
    globalMinMax.y = std::max(globalMinMax.y, minMaxs[i].y);
  }
  if(maxFilterWarnings > 0 && filteredTriangles > 0)
  {
    LOGW("Warning: %u triangles had displacements outside direction bounds. Their displacements will be clamped\n", filteredTriangles);
  }

  // Filtering out everything is a failure
  return filteredTriangles < minMaxs.size();
}

//--------------------------------------------------------------------------------------------------
// Retrieve the distances that was computed
//
void BakerVK::getDistanceFromBuffer(const meshops::OpBake_input&    info,
                                    MutableArrayView<nvmath::vec2f> outDirectionBounds,
                                    MutableArrayView<float>         distances,
                                    MutableArrayView<nvmath::vec2f> triangleMinMaxs,
                                    nvmath::vec2f&                  globalMinMax)
{
  ScopedTimer t("Get Distance Buffer");
  const float epsilon = 1e-6F;
  auto&       meshView = info.baseMeshView;
  auto minMaxs = ArrayView(static_cast<nvmath::vec2f*>(m_alloc.map(m_triangleMinMaxBuf)), info.baseMeshView.triangleCount());

#if 0
  // Brute force min/max
  {
    auto distancesGpu = static_cast<float*>(m_alloc.map(m_distanceBuf));
    size_t dist = 0;
    for(size_t i = 0; i < info.baseMeshView.triangleSubdivisionLevels.size(); ++i)
    {
      uint32_t microVertices = bary::baryValueFrequencyGetCount(bary::ValueFrequency::ePerVertex, info.baseMeshView.triangleSubdivisionLevels[i]);
      auto minMax = std::minmax_element(distancesGpu + dist, distancesGpu + dist + microVertices);
      minMaxs[i] = {*minMax.first, *minMax.second};
      dist += microVertices;
    }
    m_alloc.unmap(m_distanceBuf);
  }
#endif

  if(info.settings.fitDirectionBounds || !m_push.hasDirectionBounds)
  {
    if(!getGlobalMinMax(minMaxs, globalMinMax, info.settings.fitDirectionBounds, 5))
    {
      LOGW("Warning: All triangle bounds were filtered. Displacements will all be clamped\n");
      globalMinMax = {0.0f, 1.0f};
    }
  }
  else
  {
    // Direction bounds are provided to the baker, but further fitting is
    // disabled. Assume the bounds are good and just clamp any displacemens and
    // min/maxs outside the range.
    globalMinMax = {0.0f, 1.0f};
  }

  BiasScalef globalBiasScale{};
  if(info.settings.fitDirectionBounds)
  {
    globalBiasScale       = BiasScalef::minmax_unit(globalMinMax);
    globalBiasScale.scale = std::max(epsilon, globalBiasScale.scale);
    globalMinMax          = {0.0f, 1.0f};
  }
  BiasScalef globalBiasScaleInv = globalBiasScale.inverse();

  // Apply the global same bias/scale to the per-triangle min-maxs that will be applied to the displacements
  assert(triangleMinMaxs.size() == meshView.triangleCount());
  for(size_t i = 0; i < meshView.triangleCount(); ++i)
  {
    triangleMinMaxs[i] = globalBiasScaleInv * minMaxs[i];

    // Some min/maxs may have been filtered out from the global min/max during
    // getGlobalMinMax() and they need to be clamped.
    if(m_push.hasDirectionBounds)
    {
      triangleMinMaxs[i].x = std::max(triangleMinMaxs[i].x, 0.0f);
      triangleMinMaxs[i].y = std::min(triangleMinMaxs[i].y, 1.0f);
    }
  }

  m_alloc.unmap(m_triangleMinMaxBuf);

  auto distancesGpu = static_cast<float*>(m_alloc.map(m_distanceBuf));
  nvh::parallel_batches(
      distances.size(),
      [&](uint64_t idx) {
        float distance = distancesGpu[idx];
        assert(distance != std::numeric_limits<float>::max());

        // Invert the global bias/scale. This is OK without re-baking uniform direction bounds changes does not affect
        // the displacement direction.
        float normalized = globalBiasScaleInv * distance;

        distances[idx] = m_push.hasDirectionBounds ? std::max(0.0f, std::min(1.0f, normalized)) : distance;
        assert(!std::isnan(distances[idx]));
      },
      m_threadCount);
  m_alloc.unmap(m_distanceBuf);

  // Only rewrite the direction bounds if they were modified.
  if(info.settings.fitDirectionBounds)
  {
    nvmath::vec2f* directionBounds = static_cast<nvmath::vec2f*>(m_alloc.map(m_directionBoundsBuf));
    for(size_t i = 0; i < outDirectionBounds.size(); ++i)
    {
      auto bounds           = BiasScalef(directionBounds[i]) * globalBiasScale;
      outDirectionBounds[i] = {bounds.bias, std::max(epsilon, bounds.scale)};
    }
    m_alloc.unmap(m_directionBoundsBuf);
  }
}

}  // namespace meshops
