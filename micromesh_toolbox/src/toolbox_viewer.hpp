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

#include "nvvk/profiler_vk.hpp"
#include "nvvkhl/application.hpp"
#include "tool_context.hpp"
#include "toolbox_scene.hpp"
#include <future>


class UiRendering;
class UiRaytracing;
class UiRaster;
class UiEnvironment;
class UiMicromeshProcess;
class UiMicromeshProcessPipeline;

namespace nvvk {
class DebugUtil;
class DescriptorSetContainer;
struct RayPickerKHR;
class SBTWrapper;
}  // namespace nvvk

namespace nvvkhl {
class AllocVma;
class GBuffer;
class HdrEnv;
class HdrEnvDome;
class Scene;
class SceneRtx;
class SceneVk;
class SkyDome;
struct TonemapperPostProcess;
}  // namespace nvvkhl

//////////////////////////////////////////////////////////////////////////
/*

 This sample can load GLTF scene and render using the raster or RTX (path tracer)

*/
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
/// </summary> Ray trace multiple primitives
class ToolboxViewer : public nvvkhl::IAppElement
{
  enum GBufferType
  {
    eLdr,     // Tone mapped (display image)
    eResult,  // Result from Path tracer / raster
  };
  ViewerSettings m_settings;

public:
  ToolboxViewer() = default;
  ~ToolboxViewer() override;

  void onAttach(nvvkhl::Application* app) override;
  void onDetach() override;
  void onResize(uint32_t width, uint32_t height) override;
  void onUIMenu() override;
  void onFileDrop(const char* filename) override;
  void onUIRender() override;
  void onRender(VkCommandBuffer cmd) override;

  ViewerSettings& settings() { return m_settings; }
  void            waitForLoad();

private:
  friend UiRendering;
  friend UiRaster;
  friend UiRaytracing;
  friend UiEnvironment;
  friend UiMicromeshProcess;
  friend UiMicromeshProcessPipeline;

  [[nodiscard]] bool createScene(const std::string& filename, SceneVersion sceneVersion);
  void               createGbuffers(const nvmath::vec2f& size);
  void               createVulkanBuffers();
  bool               updateFrame();
  void               resetFrame();
  void               windowTitle();
  void               screenPicking();
  void               rtxPicking(const ImVec2& mousePosNorm);
  void               raytraceScene(VkCommandBuffer cmd);
  void               recordRasterScene(VkCommandBuffer& scnCmd);
  void               renderNodes(VkCommandBuffer              cmd,
                                 const std::vector<uint32_t>& nodeIDs,
                                 ToolboxScene*                toolbox_scene,
                                 int                          numIndexed  = 1,
                                 int                          numDraw     = 0,
                                 bool                         useMeshTask = false);
  void               renderRasterScene(VkCommandBuffer cmd);
  void               rasterScene(VkCommandBuffer cmd);
  [[nodiscard]] bool createHdr(const std::string& filename);
  void               destroyResources();
  void               addSettingsHandler();
  void               rasterPicking(const ImVec2& mousePosNorm);
  void               updateDirty();
  void               updateHbao();
  bool               keyShortcuts();
  void               updateFrameInfo(VkCommandBuffer cmd);
  float              getDepth(int x, int y);

  void setAllDirty(SceneDirtyFlags flag, bool v = true)
  {
    for(auto& scene : m_scenes)
      scene->setDirty(flag, v);
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*              m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>  m_dutil;
  std::unique_ptr<nvvkhl::AllocVma> m_alloc;

  nvmath::vec2f                    m_viewSize{1, 1};
  VkClearColorValue                m_clearColor{{0.3F, 0.3F, 0.3F, 1.0F}};  // Clear color
  VkDevice                         m_device{VK_NULL_HANDLE};                // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                              // G-Buffers: color + depth
  VkFormat                         m_ldrFormat{VK_FORMAT_R8G8B8A8_UNORM};
  VkFormat                         m_resultFormat{VK_FORMAT_R32G32B32A32_SFLOAT};
  uint32_t                         m_driverMaxSubdivLevel = 0;


  // Resources
  nvvk::Buffer         m_bFrameInfo;
  nvvk::Buffer         m_pixelBuffer;
  nvvk::Context::Queue m_qGCT1{};

  // Async loading
  std::future<bool> m_loadingScene;
  std::future<bool> m_loadingHdr;

  // Pipeline
  shaders::PushConstant m_pushConst{};  // Information sent to the shader
  int                   m_frame{-1};
  shaders::FrameInfo    m_frameInfo{};

  std::unique_ptr<nvvkhl::HdrEnv>                m_hdrEnv;
  std::unique_ptr<nvvkhl::HdrEnvDome>            m_hdrDome;
  std::unique_ptr<nvvkhl::SkyDome>               m_sky;
  std::unique_ptr<nvvkhl::TonemapperPostProcess> m_tonemapper;
  std::unique_ptr<nvvk::RayPickerKHR>            m_picker;  // For ray picking info
  std::unique_ptr<HbaoPass>                      m_hbao;
  std::unique_ptr<nvvk::ProfilerVK>              m_profilerVk;

  // Micromesh tools
  std::unique_ptr<micromesh_tool::ToolContext> m_toolContext;

  // There are NUMSCENES scenes in the application: 
  // Reference, Base, Scratch (intermediate one for backup, used optionally)
  std::array<std::unique_ptr<ToolboxScene>, NUMSCENES> m_scenes;
  ToolboxScene*                                        getScene(SceneVersion v) { return m_scenes[v].get(); }
  ToolboxScene*                                        getScene(ViewerSettings::RenderViewSlot v);
  void                                                 saveScene(const std::string& filename, SceneVersion s = eBase);
};
