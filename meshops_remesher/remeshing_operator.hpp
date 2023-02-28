/*
 * Copyright (c) 2014-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#pragma once
#include <micromesh/micromesh_displacement_remeshing.h>

#ifdef __cplusplus  // GLSL Type
#include <glm/glm.hpp>
using namespace glm;
#endif
#include "shaders/remeshing_host_device.h"
#include "micromesh/micromesh_gpu.h"


#include "nvvk/images_vk.hpp"
#include "nvh/timesampler.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvh/nvprint.hpp"
#include "nvvk/memallocator_vma_vk.hpp"
#include "vk_mem_alloc.h"
#include "nvvk/descriptorsets_vk.hpp"

#include "nvvk/compute_vk.hpp"
#include "meshops/meshops_operations.h"
#include <array>

namespace meshops {

class RemeshingOperator_c
{
public:
  bool create(Context context);
  bool destroy(Context context);


  micromesh::Result remesh(Context               context,
                           const OpRemesh_input& input,
                           OpRemesh_modified&    modified,
                           DeviceMesh            modifiedMesh,
                           uint32_t*             outputTriangleCount,
                           uint32_t*             outputVertexCount);

  std::vector<DeviceMesh> inputDeviceMeshes;
  std::vector<DeviceMesh> modifiedDeviceMeshes;
  std::vector<DeviceMesh> localDeviceMeshes;

private:
  bool beginRemeshTask(Context context);
  bool endRemeshTask(Context context);

  bool m_isInitialized{false};

  micromesh::gpu::GpuRemeshing m_remesher = nullptr;
  micromesh::gpu::SetupInfo    m_remesherSetupInfo;

  micromesh::OpRemeshing_settings m_remesherParams;

  struct PipelineLayout
  {
    VkPipelineLayout               layout;
    nvvk::DescriptorSetBindings    bindings;
    VkDescriptorSetLayout          descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool               descriptorPool{VK_NULL_HANDLE};
    std::array<VkDescriptorSet, 8> descriptorSets;
    uint32_t                       currentDescriptorSet;
    VkDescriptorSet getNextDescriptorSet() { return descriptorSets[currentDescriptorSet++ % descriptorSets.size()]; }
  };


  std::vector<PipelineLayout>                                                     m_pipelineLayouts;
  std::vector<nvvk::Buffer>                                                       m_scratchPersistentResources;
  std::vector<VkPipeline>                                                         m_pipelines;
  std::vector<VkPipeline>                                                         m_userPipelines;
  std::vector<micromesh::gpu::ResourceInfo<micromesh::gpu::GpuRemeshingResource>> m_readResourceInfos;
  std::vector<void*>                                                              m_readResourceDatas;
  std::vector<uint64_t>                                                           m_readResourceSizes;
  nvvk::Buffer                                                                    m_globalConstantBuffer;

  void createPipelineLayout(Context context, const micromesh::gpu::PipelineLayoutInfo& info, size_t index);
  void createPipeline(Context context, const micromesh::gpu::PipelineInfo& info, size_t index);
  void createRemesherResources(Context context, const OpRemesh_input& input, OpRemesh_modified& modified, DeviceMesh modifiedMesh);
  void freeRemesherBuffers(Context context);
  void copyMeshToRemesher(VkCommandBuffer cmd, const OpRemesh_input& inputs);

  struct TaskBuffer
  {
    VkBuffer     deviceBuffer{};
    size_t       size{0ull};
    nvvk::Buffer hostVisibleBuffer{};
  };

  struct TaskData
  {
    std::vector<nvvk::Buffer> scratchTaskResources;
    std::vector<TaskBuffer>   allResourceHandles;

    RemeshingOperator_c* sysData;
    //SystemData*                                      sysData;
    VkCommandBuffer                                                        cmd;
    micromesh::gpu::ReadResourceData<micromesh::gpu::GpuRemeshingResource> readData;
    bool                                                                   hadRead = false;


    std::vector<std::vector<uint8_t>> hostReadBuffers;

    meshops::Context context;

    std::vector<nvvk::Buffer> localTaskResources;

  } m_taskData;
  micromesh::gpu::GpuRemeshingTask                                          m_task = nullptr;
  micromesh::gpu::CommandSequenceInfo<micromesh::gpu::GpuRemeshingResource> m_seq;
  micromesh::gpu::GpuRemeshing_input                                        m_input;
  micromesh::gpu::GpuRemeshing_output                                       m_output;


  micromesh::RemeshingCurrentState                                        m_currentState{};
  nvvk::PushComputeDispatcher<VertexCopyConstants, VertexKernelBindings>  m_vertexCopy;
  nvvk::PushComputeDispatcher<VertexMergeConstants, VertexKernelBindings> m_vertexMerge;

  meshops::MeshAttributeFlags m_preservedAttributes = {};

  uint32_t m_primId{};


  struct TriangleSubdivisionInfo
  {
    uint16_t edgeFlags;
    uint16_t subdivLevel;
  };

  // FIXME: should be provided outside
  float m_curvaturePower = 1.f;

  uint32_t m_texcoordCount{};

  uint32_t m_texcoordIndex{};
  uint32_t m_heightmapTextureCoord = ~0u;
};
}  // namespace meshops