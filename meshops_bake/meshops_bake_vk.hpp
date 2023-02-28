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
#include "nvvk/context_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"

namespace meshops {

// TODO: (glm?) compile errors without namespaces
namespace glsl_shared {
#include "shaders/host_device.h"
}  // namespace glsl_shared

struct GeometryBatch
{
  uint32_t triangleOffset;
  uint32_t triangleCount;
  uint32_t batchIndex;
  uint32_t totalBatches;
};

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

struct SceneVK
{
  std::vector<nvvk::Buffer> verticesBuf;  // One buffer per primitive (Vertex)
  std::vector<nvvk::Buffer> indicesBuf;   // One buffer per primitive (uint32_t)
  nvvk::Buffer              primInfoBuf;  // Array of PrimMeshInfo
  std::vector<uint32_t>     numVertices;
  std::vector<uint32_t>     numTriangles;  // Buffer per primitive
  size_t                    totalTriangles{};
};

class BakerVK
{
public:
  BakerVK(micromesh::OpContext micromeshContext, meshops::ContextVK& vkContext, unsigned int threadCount);
  ~BakerVK();

  bool bakeAndResample(const meshops::OpBake_input&              input,
                       const GeometryBatch&                      batch,
                       bool                                      resample,
                       const std::vector<VkDescriptorImageInfo>& inputTextures,
                       const std::vector<VkDescriptorImageInfo>& outputTextures,
                       const std::vector<VkDescriptorImageInfo>& distanceTextures,
                       ArrayView<meshops::Texture>               outputTextureInfo);
  void createVkResources(const meshops::OpBake_input& info,
                         const meshops::MeshView&     lowScene,
                         const meshops::MeshView&     highScene,
                         MutableArrayView<float>      distances);
  bool createVkHighGeometry(const meshops::OpBake_input& info, const meshops::MeshView& highScene, const GeometryBatch& batch);
  void destroyVkHighGeometry();
  void createVertexBuffer(VkCommandBuffer cmdBuf, const meshops::MeshView& meshView, SceneVK& vkScene);
  bool createTessellatedVertexBuffer(const meshops::OpBake_input& info,
                                     VkCommandBuffer              cmdBuf,
                                     const meshops::MeshView&     meshView,
                                     SceneVK&                     vkScene,
                                     const GeometryBatch&         batch,
                                     int                          maxSubdivLevel);
  auto primitiveToGeometry(const meshops::MeshView& prim,
                           VkDeviceAddress          vertexAddress,
                           VkDeviceAddress          indexAddress,
                           uint32_t                 numVertices,
                           uint32_t                 numTriangles);
  void createBottomLevelAS(const meshops::MeshView& meshView, const SceneVK& vkScene, const GeometryBatch& batch);
  void createTopLevelAS(const meshops::OpBake_input& info, const meshops::MeshView& meshView, const GeometryBatch& batch);
  void fitDirectionBounds(const meshops::OpBake_input& info, MutableArrayView<float> distances);
  [[nodiscard]] bool getGlobalMinMax(ArrayView<const nvmath::vec2f> minMaxs,
                                     nvmath::vec2f&                 globalMinMax,
                                     bool                           filterZeroToOne,
                                     const uint32_t                 maxFilterWarnings);
  void getDistanceFromBuffer(const meshops::OpBake_input&    info,
                             MutableArrayView<nvmath::vec2f> outDirectionBounds,
                             MutableArrayView<float>         distances,
                             MutableArrayView<nvmath::vec2f> triangleMinMaxs,
                             nvmath::vec2f&                  globalMinMax);
  void destroy();
  void destroyVkScene(SceneVK& vkScene);
  void createBakerPipeline();
  void destroyBakerPipeline();
  void createResamplerPipeline(const std::vector<VkDescriptorImageInfo>& inputTextures,
                               const std::vector<VkDescriptorImageInfo>& outputTextures,
                               const std::vector<VkDescriptorImageInfo>& distanceTextures);
  void destroyResamplerPipeline();
  void runComputePass(const meshops::OpBake_input& info, bool finalBatch);
  void runResampler(const meshops::OpBake_input& info, ArrayView<meshops::Texture> outputTextures);

private:
  // Vulkan Info
  VkDevice         m_device{VK_NULL_HANDLE};
  VkPhysicalDevice m_physicalDevice{VK_NULL_HANDLE};

  nvvk::Context::Queue m_queueT;
  nvvk::Context::Queue m_queueC;
  nvvk::Context::Queue m_queueGCT;

  micromesh::OpContext m_micromeshContext;

  // Vulkan resources
  SceneVK                   m_lowVk;
  SceneVK                   m_highVk;
  nvvk::Buffer              m_distanceBuf;
  nvvk::Buffer              m_trianglesBuf;
  nvvk::Buffer              m_triangleMinMaxBuf;       // per-triangle direction-length-relative (min, max) pairs
  nvvk::Buffer              m_directionBoundsBuf;      // per-vertex direction-length-relative (start, end) pairs
  nvvk::Buffer              m_directionBoundsOrigBuf;  // copy of initial values for maintaining a max trace dist
  std::vector<nvvk::Buffer> m_baryCoordBuf;            // 0..5
  nvvk::Buffer              m_sceneDescBuf;
  PipelineContainer         m_pContainer;
  DescriptorContainer       m_dContainer;

  struct TextureResampling
  {
    PipelineContainer   pipeline;
    DescriptorContainer descriptor;
  };

  TextureResampling m_resampling;

  // Memory allocator
  nvvk::ResourceAllocator& m_alloc;  // Allocator for buffer, images, acceleration structures

  // Utilities
  nvvk::RaytracingBuilderKHR m_rtBuilder;  // To build BLAS and TLAS

  glsl_shared::PushHighLow m_push;

  uint32_t m_maxNumBaryCoord{0};

  unsigned int m_threadCount;
};

}  // namespace meshops
