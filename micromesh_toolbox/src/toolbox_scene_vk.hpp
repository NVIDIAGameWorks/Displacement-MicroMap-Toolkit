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


#include "nvvkhl/alloc_vma.hpp"
#include "tool_scene.hpp"
#include "micromap/device_micromap.hpp"

using ToolImageVector = std::vector<std::unique_ptr<micromesh_tool::ToolImage>>;

// Create the Vulkan version of the Scene
// Allocate the buffers, etc.
class ToolboxSceneVk
{
public:
  ToolboxSceneVk(nvvk::Context* ctx, nvvkhl::AllocVma* alloc, meshops::Context context, nvvk::Context::Queue extraQueue);
  ~ToolboxSceneVk();

  void create(VkCommandBuffer cmd, micromesh_tool::ToolScene& scn);
  void destroy();

  // Getters
  const nvvk::Buffer&                             material() const { return m_bMaterial; }
  const nvvk::Buffer&                             primInfo() const { return m_bDeviceMeshInfo; }
  const nvvk::Buffer&                             instances() const { return m_bInstances; }
  const nvvk::Buffer&                             sceneDesc() const { return m_bSceneDesc; }
  const std::vector<nvvk::Texture>&               textures() const { return m_textures; }
  const std::vector<std::unique_ptr<DeviceBary>>& barys() const { return m_barys; }
  uint32_t nbTextures() const { return static_cast<uint32_t>(m_textures.size()); }
  int32_t  baryInfoIndex(int32_t bary, int32_t group) const { return m_deviceBaryInfoMap.find({bary, group})->second; }

  const meshops::DeviceMesh& deviceMesh(uint32_t m) const { return m_deviceMeshes[m]; }
  bool                       hasRtxMicromesh() const { return !m_barys.empty() && m_hasDisplacementMicromeshExt; }

private:
  struct SceneImage  // Image to be loaded and created
  {
    nvvk::Image       nvvkImage;
    VkImageCreateInfo createInfo;

    // Loading information
    bool                              srgb{false};
    std::string                       imgName;
    VkExtent2D                        size{0, 0};
    VkFormat                          format{VK_FORMAT_UNDEFINED};
    std::vector<std::vector<uint8_t>> mipData;
  };

  void               createMaterialBuffer(VkCommandBuffer cmd, const micromesh_tool::ToolScene& scn);
  void               createInstanceInfoBuffer(VkCommandBuffer cmd, const micromesh_tool::ToolScene& scn);
  [[nodiscard]] bool createDeviceMeshBuffer(VkCommandBuffer cmd, micromesh_tool::ToolScene& scn);
  [[nodiscard]] bool createDeviceBaryBuffer(VkCommandBuffer cmd, nvvk::Context::Queue extraQueue, micromesh_tool::ToolScene& scn);
  nvvk::Buffer createWatertightIndicesBuffer(VkCommandBuffer                          cmd,
                                             meshops::ArrayView<const nvmath::vec3ui> triVertices,
                                             const meshops::MeshTopologyData&         topology);
  void createTextureImages(VkCommandBuffer cmd, const std::vector<tinygltf::Texture>& textures, const ToolImageVector& images);
  void loadImage(micromesh_tool::ToolImage& toolImage, SceneImage& image);
  bool createImage(const VkCommandBuffer& cmd, SceneImage& image);

  //--
  nvvk::Context*                   m_ctx;
  nvvkhl::AllocVma*                m_alloc;
  std::unique_ptr<nvvk::DebugUtil> m_dutil;
  nvvk::Context::Queue             m_qGCT1 = {};

  //--
  meshops::Context m_context;

  nvvk::Buffer m_bMaterial;
  nvvk::Buffer m_bDeviceMeshInfo;
  nvvk::Buffer m_bDeviceBaryInfo;
  nvvk::Buffer m_bInstances;
  nvvk::Buffer m_bSceneDesc;

  std::vector<meshops::DeviceMesh> m_deviceMeshes;

  // Buffers of per-triangle WatertightIndices structures for rendering
  // heightmaps without cracks.
  std::vector<nvvk::Buffer> m_meshWatertightIndices;

  // Common tables of micro-vertex positions and topology. Used when rasterizing
  // micromeshes and heightmaps.
  microdisp::MicromeshSplitPartsVk m_micromeshSplitPartsVK;

  // Device equivalents of ToolScene::barys(). Typically there is only one, with
  // a group/micromap per ToolMesh.
  std::vector<std::unique_ptr<DeviceBary>> m_barys;

  // The barys+groups are linearized. In the case meshes reference multiple bary
  // files (e.g. after ToolMerge), this map translates the (bary, group) key to
  // a single DeviceBaryInfo index.
  std::map<std::pair<int32_t, int32_t>, int32_t> m_deviceBaryInfoMap;

  std::vector<SceneImage>    m_images;
  std::vector<nvvk::Texture> m_textures;  // Vector of all textures of the scene

  // True if VK_NV_displacement_micromap exists
  bool m_hasDisplacementMicromeshExt = false;
};
