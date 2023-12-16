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

#include <meshops/meshops_vk.h>
#include <meshops/meshops_api.h>
#include <meshops/meshops_operations.h>
#include "meshops_bake_batch.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "vulkan/vulkan_core.h"

namespace meshops {

#include "shaders/host_device.h"

struct PipelineContainer
{
  VkPipeline       pipeline{VK_NULL_HANDLE};
  VkPipelineLayout layout{VK_NULL_HANDLE};
};

struct DescriptorContainer
{
  nvvk::DescriptorSetBindings binder;
  VkDescriptorSet             set{VK_NULL_HANDLE};
  VkDescriptorSetLayout       layout{VK_NULL_HANDLE};
  VkDescriptorPool            pool{VK_NULL_HANDLE};
};

struct BakerMeshVK
{
  void create(nvvk::ResourceAllocator* alloc, VkCommandBuffer cmdBuf, const meshops::MeshView& meshView, bool requireDirectionBounds);
  bool            createTessellated(micromesh::OpContext         micromeshContext,
                                    nvvk::ResourceAllocator*     alloc,
                                    const meshops::OpBake_input& input,
                                    VkCommandBuffer              cmdBuf,
                                    const meshops::MeshView&     meshView,
                                    const GeometryBatch&         batch,
                                    int                          maxSubdivLevel);
  void            destroy(nvvk::ResourceAllocator* alloc);
  static uint64_t estimateGpuMemory(uint64_t triangles, uint64_t vertices, bool requireDirectionBounds);

  nvvk::Buffer verticesBuf;
  nvvk::Buffer directionBoundsBuf;
  nvvk::Buffer directionBoundsOrigBuf;
  nvvk::Buffer indicesBuf;
  nvvk::Buffer primInfoBuf;  // Array of BakerMeshInfo
  uint32_t     numVertices  = 0;
  uint32_t     numTriangles = 0;  // Buffer per primitive
};

// Pairs a VkPipeline with the descriptor set for a single instance.
struct BakerPipeline
{
  void create(VkDevice device, VkBuffer sceneDescBuf, VkAccelerationStructureKHR referenceSceneTlas);
  void destroy(VkDevice device);
  void run(meshops::ContextVK& vk, const meshops::OpBake_input& input, shaders::BakerPushConstants& pushConstants, bool finalBatch);

  PipelineContainer   pipeline;
  DescriptorContainer descriptor;
};

struct ResamplerPipeline
{
  void create(VkDevice                                  device,
              VkBuffer                                  sceneDescBuf,
              VkAccelerationStructureKHR                referenceSceneTlas,
              const std::vector<VkDescriptorImageInfo>& inputTextures,
              const std::vector<VkDescriptorImageInfo>& outputTextures,
              const std::vector<VkDescriptorImageInfo>& distanceTextures);
  void destroy(VkDevice device);
  void run(meshops::ContextVK&          vk,
           const meshops::OpBake_input& input,
           ArrayView<meshops::Texture>  outputTextures,
           shaders::BakerPushConstants& pushConstants,
           nvvk::Buffer&                triangleMinMaxBuf);

  PipelineContainer   pipeline;
  DescriptorContainer descriptor;
};

struct BakerReferenceScene
{
  using BlasInput = nvvk::RaytracingBuilderKHR::BlasInput;
  bool            create(micromesh::OpContext         micromeshContext,
                         meshops::ContextVK&          vk,
                         const meshops::OpBake_input& input,
                         const meshops::MeshView&     meshView,
                         const GeometryBatch&         batch);
  void            destroy(nvvk::ResourceAllocator* alloc);
  static uint64_t estimateGpuMemory(VkDevice device, uint64_t triangles, uint64_t vertices);
  static BlasInput createBlasInput(VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress, uint32_t numVertices, uint32_t numTriangles);
  void createBottomLevelAS(VkDevice device);
  void createTopLevelAS(const meshops::OpBake_input& input);

  BakerMeshVK referenceVk;

  // The baker and resampler use raytracing to find intersections with the
  // reference scene
  nvvk::RaytracingBuilderKHR rtBuilder;
};

class BakerVK
{
public:
  BakerVK(micromesh::OpContext micromeshContext, meshops::ContextVK& vkContext);
  ~BakerVK();

  bool bakeAndResample(const meshops::OpBake_input&              input,
                       const GeometryBatch&                      batch,
                       bool                                      resample,
                       const std::vector<VkDescriptorImageInfo>& inputTextures,
                       const std::vector<VkDescriptorImageInfo>& outputTextures,
                       const std::vector<VkDescriptorImageInfo>& distanceTextures,
                       ArrayView<meshops::Texture>               outputTextureInfo);
  void create(const meshops::OpBake_input& input, MutableArrayView<float> distances);
  static uint64_t estimateBaseGpuMemory(uint64_t distances, uint64_t triangles, uint64_t vertices, bool requireDirectionBounds);
  static uint64_t estimateBatchGpuMemory(VkDevice device, uint64_t triangles, uint64_t vertices);
  void            fitDirectionBounds(const meshops::OpBake_input& input, MutableArrayView<float> distances);
  void            getDistanceFromBuffer(const meshops::OpBake_input&    input,
                                        MutableArrayView<nvmath::vec2f> outDirectionBounds,
                                        MutableArrayView<float>         distances,
                                        MutableArrayView<nvmath::vec2f> triangleMinMaxs,
                                        nvmath::vec2f&                  globalMinMax);
  void            destroy();

private:
  meshops::ContextVK& m_vk;

  micromesh::OpContext m_micromeshContext;

  BakerMeshVK  m_baseVk;
  nvvk::Buffer m_distanceBuf;        // baker result - a linear array of floats
  nvvk::Buffer m_trianglesBuf;       // per-triangle microvertex offsets, shaders::Triangle
  nvvk::Buffer m_triangleMinMaxBuf;  // per-triangle direction-length-relative displacement distance (min, max) pairs
  std::vector<nvvk::Buffer> m_baryCoordBuf;  // Micro-triangle coordinates in bary space

  // Shader push constants. These persist between calls to bakeAndResample().
  shaders::BakerPushConstants m_push;
};

}  // namespace meshops
