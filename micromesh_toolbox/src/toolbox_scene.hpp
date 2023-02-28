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
#include "nvvk/sbtwrapper_vk.hpp"
#include <bitset>
#include "toolbox_scene_rtx.hpp"
#include "settings.hpp"

// Dirty system
enum SceneDirtyFlags
{
  eDeviceMesh,        // Micromesh ToolScene has changed
  eRasterPipeline,    // Change of specialization
  eRtxPipeline,       // Change of specialization
  eRasterRecord,      // Change of anything related to drawing, wireframe, nb-elements, ...
  eDescriptorSets,    // Re-writing the descriptor sets
  eRtxAccelerations,  // Building the acceleration structures
  eNumFlags
};

// List of solid, transparent, or all nodes
enum class SceneNodeMethods
{
  eSolid,  // Only solid nodes of specialization
  eBlend,  // Only nodes that are not solid
  eAll,    // All nodes, solid or not
};

// Using micromesh, or not, or we don't care
enum class SceneNodeMicromesh
{
  eMicromeshWith,
  eMicromeshWithout,
  eMicromeshDontCare,
};


//--------------------------------------------------------------------------------------------------
// This is one scene to be render
// Contains all the resources to be render
//
class ToolboxScene
{
public:
  ToolboxScene(nvvk::Context* ctx, nvvkhl::AllocVma* alloc, nvvk::Context::Queue extraQueue, VkCommandPool cmdPool);
  ~ToolboxScene();

  // Create the scene from a filename
  void createFromFile(const std::string& filename);
  void destroy();

  // Resource creations
  void createRtxAccelerations(bool useMicroMesh);
  void createVulkanBuffers();
  void createRtxPipeline(const std::vector<VkDescriptorSetLayout> extraLayouts);
  void createRasterPipeline(const ViewerSettings&                    settings,
                            const std::vector<VkDescriptorSetLayout> extraLayouts,
                            VkFormat                                 colorFormat,
                            VkFormat                                 depthformat);


  // Information getters
  bool                                                 valid() const { return m_toolscene && m_toolscene->valid(); }
  const std::filesystem::path&                         getPathName() const { return m_pathFilename; }
  meshops::Context                                     getContext() { return m_context; }
  bool                                                 hasBary() const { return !m_toolscene->barys().empty(); }
  const std::optional<micromesh_tool::ToolSceneStats>& stats() const { return m_sceneStats; }

  // Pipeline getters
  VkDescriptorSet                  getDescSet() const { return m_sceneSet->getSet(); }
  VkDescriptorSet                  getRtxDescSet() const { return m_rtxSet->getSet(); }
  VkDescriptorSetLayout            getDescLayout() const { return m_sceneSet->getLayout(); }
  VkDescriptorSetLayout            getRtxDescLayout() const { return m_rtxSet->getLayout(); }
  const nvvkhl::PipelineContainer& getRasterPipeline() const { return m_rasterPipe; }
  const nvvkhl::PipelineContainer& getRtxPipeline() const { return m_rtxPipe; }


  // RTX info
  VkAccelerationStructureKHR                     getTlas() { return m_toolsceneRtx->tlas(); }
  std::array<VkStridedDeviceAddressRegionKHR, 4> getSbtRegions() { return m_sbt->getRegions(); }

  // Scene info
  std::unique_ptr<micromesh_tool::ToolScene>&           getToolScene() { return m_toolscene; }
  std::unique_ptr<ToolboxSceneVk>&                      getToolSceneVK() { return m_toolsceneVk; }    // Host scene
  std::unique_ptr<ToolboxSceneRtx>&                     getToolSceneRtx() { return m_toolsceneRtx; }  // Host scene
  std::unique_ptr<micromesh_tool::ToolSceneDimensions>& getDimensions() { return m_scnDimensions; }

  // Recorded commands for displaying the raster scene
  VkCommandBuffer getRecordedCommandBuffer() { return m_recordedSceneCmd; }
  VkCommandBuffer createRecordCommandBuffer();
  void            freeRecordCommandBuffer();

  // Returns nodes ID for the type of rendering: solid, blend, with or without µMesh
  std::vector<uint32_t> getNodes(SceneNodeMethods method, SceneNodeMicromesh micromesh) const;

  // Writing to descriptor sets all the resources, to call when GBuffers are changed
  void writeSets(VkDescriptorImageInfo& outImage, VkDescriptorBufferInfo& frameInfo);

  // Dirty system
  void setDirty(SceneDirtyFlags flag, bool v = true) { m_dirty.set(flag, v); }
  bool noneDirty() { return m_dirty.none(); }
  bool isDirty(SceneDirtyFlags flag) { return m_dirty.test(flag); }
  void resetDirty(SceneDirtyFlags flag) { m_dirty.reset(flag); }


private:
  void createRtxSet();
  void createSceneSet();
  void writeRtxSet(VkDescriptorImageInfo& outImage);
  void writeSceneSet(VkDescriptorBufferInfo& frameInfo);
  void setShadeNodes();
  void createSbt(VkPipeline rtPipeline, VkRayTracingPipelineCreateInfoKHR rayPipelineInfo);
  void setCameraFromScene(const std::filesystem::path& filename);

  nvvkhl::AllocVma*    m_alloc            = nullptr;
  nvvk::Context*       m_ctx              = nullptr;
  VkDevice             m_device           = {VK_NULL_HANDLE};
  nvvk::Context::Queue m_qGCT1            = {};
  VkCommandBuffer      m_recordedSceneCmd = {VK_NULL_HANDLE};
  VkCommandPool        m_cmdPool          = {VK_NULL_HANDLE};
  meshops::Context     m_context          = nullptr;

  std::unique_ptr<nvvk::SBTWrapper>                    m_sbt;            // Shading binding table wrapper
  std::unique_ptr<nvvk::DescriptorSetContainer>        m_rtxSet;         // Descriptor set
  std::unique_ptr<nvvk::DescriptorSetContainer>        m_sceneSet;       // Descriptor set
  std::unique_ptr<nvvk::DebugUtil>                     m_dutil;          // Debug utility
  std::unique_ptr<micromesh_tool::ToolScene>           m_toolscene;      // Host scene
  std::unique_ptr<ToolboxSceneVk>                      m_toolsceneVk;    // Device scene
  std::unique_ptr<ToolboxSceneRtx>                     m_toolsceneRtx;   // Device RTX scene
  std::unique_ptr<micromesh_tool::ToolSceneDimensions> m_scnDimensions;  // On load: scene dimensions
  std::filesystem::path                                m_pathFilename;   // On load: scene path
  std::optional<micromesh_tool::ToolSceneStats>        m_sceneStats;     // On load: summary of scene data
  std::bitset<SceneDirtyFlags::eNumFlags>              m_dirty;          // Dirty flags
  std::vector<std::bitset<2>>                          m_shadeNodes;     // Nodes with shading info: blend/µMesh
  nvvkhl::PipelineContainer                            m_rasterPipe;     // Pipeline for raster
  nvvkhl::PipelineContainer                            m_rtxPipe;        // Pipeline for RTX
};
