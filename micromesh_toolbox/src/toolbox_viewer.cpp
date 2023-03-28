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

// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const nvmath::vec2f& f) {x = f.x; y = f.y;} operator nvmath::vec2f() const { return nvmath::vec2f(x, y); }
// clang-format on

#include <thread>

#include "toolbox_viewer.hpp"


#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"

#include "nvh/timesampler.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/raypicker_vk.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/hdr_env.hpp"
#include "nvvkhl/hdr_env_dome.hpp"
#include "nvvkhl/sky.hpp"
#include "nvvkhl/tonemap_postprocess.hpp"
#include "ui/ui_about.hpp"
#include "ui/ui_axis.hpp"
#include "ui/ui_busy_window.hpp"
#include "ui/ui_environment.hpp"
#include "ui/ui_raster.hpp"
#include "ui/ui_raytracing.hpp"
#include "ui/ui_rendering.hpp"
#include "ui/ui_statistics.hpp"


extern std::shared_ptr<nvvkhl::ElementCamera> g_elem_camera;


// for MICRO_GROUP_SIZE

#define RASTER_SS_SIZE 2.0F  // Change this for the default Super-Sampling resolution multiplier for raster

#include "GLFW/glfw3.h"

#ifdef USE_NSIGHT_AFTERMATH
#include "aftermath/nsight_aftermath_gpu_crash_tracker.h"
extern std::unique_ptr<::GpuCrashTracker> g_aftermathTracker;
#endif

#include "vulkan_mutex.h"
#include "ui/ui_micromesh_process.hpp"
#include "microutils/microutils.hpp"


namespace fs = std::filesystem;


ToolboxViewer::~ToolboxViewer() {}

//--------------------------------------------------------------------------------------------------
// This is called by the Application, when this "Element" is added.
//
void ToolboxViewer::onAttach(nvvkhl::Application* app)
{
  auto scope_t = nvh::ScopedTimer("onAttach\n");

  m_app    = app;
  m_device = m_app->getDevice();

  nvvk::Context* ctx = app->getContext().get();

  const uint32_t c_queue_index = ctx->m_queueC.familyIndex;

  // Create an extra queue for loading in parallel
  m_qGCT1 = ctx->createQueue(VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT, "GCT1", 1.0F);

  m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);  // Debug utility
  m_alloc = std::make_unique<nvvkhl::AllocVma>(ctx);      // Allocator

  // The two scenes this application can deal with
  for(auto&& scene : m_scenes)
    scene = std::make_unique<ToolboxScene>(ctx, m_alloc.get(), m_qGCT1, m_app->getCommandPool());

  m_tonemapper = std::make_unique<nvvkhl::TonemapperPostProcess>(ctx, m_alloc.get());
  m_sky        = std::make_unique<nvvkhl::SkyDome>(ctx, m_alloc.get());
  m_hdrEnv     = std::make_unique<nvvkhl::HdrEnv>(ctx, m_alloc.get(), c_queue_index);      // HDR Generic
  m_hdrDome    = std::make_unique<nvvkhl::HdrEnvDome>(ctx, m_alloc.get(), c_queue_index);  // HDR raster
  m_picker     = std::make_unique<nvvk::RayPickerKHR>(ctx, m_alloc.get(), c_queue_index);  // Raytrace picker
  m_gBuffers   = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get());

  meshops::ContextVK sharedContextVK;
  sharedContextVK.context  = ctx;
  sharedContextVK.vma      = static_cast<struct VmaAllocator_T*>(m_alloc->vma());
  sharedContextVK.queueGCT = m_qGCT1;

  // Micromesh Tools
  meshops::ContextConfig meshopContexConfig;
  meshopContexConfig.messageCallback = microutils::makeDefaultMessageCallback();
  meshopContexConfig.threadCount     = std::thread::hardware_concurrency();
  meshopContexConfig.verbosityLevel  = 999;
  // Not sharing any of the Queues, so there won't be any conflicts with the current Application
  // But this means that resources created there, can't be shared and have to be re-created from host.
  m_toolContext = std::make_unique<micromesh_tool::ToolContext>(meshopContexConfig, sharedContextVK);

  // Vulkan profiler
  m_profilerVk = std::make_unique<nvvk::ProfilerVK>(m_device, ctx->m_physicalDevice);

  // HDR environment lighting
  m_hdrEnv->loadEnvironment("");  // Initialize the environment with nothing (constant white: for now)
  m_hdrDome->create(m_hdrEnv->getDescriptorSet(), m_hdrEnv->getDescriptorSetLayout());  // Same as above

  // HBAO Pass
  HbaoPass::Config config{};
  config.maxFrames    = 1;
  config.targetFormat = m_resultFormat;
  m_hbao              = std::make_unique<HbaoPass>(m_device, m_alloc.get(), config);

  // Create Vulkan resources
  createGbuffers(m_viewSize);
  createVulkanBuffers();

  m_tonemapper->createComputePipeline();

  // Query the maximum subdivision level supported by the vulkan implementation
  if(ctx->hasDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME))
  {
    VkPhysicalDeviceProperties2                      prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    VkPhysicalDeviceDisplacementMicromapPropertiesNV displacementMicromapProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_PROPERTIES_NV};
    prop2.pNext = &displacementMicromapProperties;
    vkGetPhysicalDeviceProperties2(ctx->m_physicalDevice, &prop2);
    m_driverMaxSubdivLevel = displacementMicromapProperties.maxDisplacementMicromapSubdivisionLevel;
  }

  // For saving Viewer related settings
  addSettingsHandler();
}

//--------------------------------------------------------------------------------------------------
// Called by Application when it is shutting down
//
void ToolboxViewer::onDetach()
{
  destroyResources();
}

//--------------------------------------------------------------------------------------------------
// Called by Application when the viewport size changed
//
void ToolboxViewer::onResize(uint32_t width, uint32_t height)
{
  createGbuffers({width, height});
  m_tonemapper->updateComputeDescriptorSets(m_gBuffers->getDescriptorImageInfo(eResult),  // From
                                            m_gBuffers->getDescriptorImageInfo(eLdr));    // To

  if(m_settings.activtyStatus.isBusy())
    return;

  ToolboxScene* toolbox_scene = getScene(m_settings.geometryView.slot);
  toolbox_scene->setDirty(SceneDirtyFlags::eDescriptorSets);
}

//--------------------------------------------------------------------------------------------------
// Called by Application for rendering the Menu Bar (separated from general UI)
//
void ToolboxViewer::onUIMenu()
{
  if(m_settings.activtyStatus.updateState())
  {
    setAllDirty(SceneDirtyFlags::eRasterRecord);
    setAllDirty(SceneDirtyFlags::eDescriptorSets);
    setAllDirty(SceneDirtyFlags::eRasterPipeline);
    setAllDirty(SceneDirtyFlags::eRtxPipeline);
    setAllDirty(SceneDirtyFlags::eRtxAccelerations);
    resetFrame();
  }

  bool load_file{false};
  bool save_file{false};

  static bool close_app{false};
  static bool show_about{false};
  bool        v_sync = m_app->isVsync();
#ifdef _DEBUG
  static bool show_demo{false};
#endif

  windowTitle();

  if(ImGui::BeginMenu("File"))
  {
    if(ImGui::MenuItem("Load", "Ctrl+O"))
    {
      load_file = true;
    }
    if(ImGui::MenuItem("Save Base As", "Ctrl+S", nullptr, m_scenes[eBase]->valid()))
    {
      save_file = true;
    }
    if(ImGui::MenuItem("Exit", "Ctrl+Q"))
    {
      close_app = true;
    }
    ImGui::EndMenu();
  }

  if(ImGui::BeginMenu("Tools"))
  {
    ImGui::MenuItem("Show Advanced Menu", nullptr, &m_settings.showAdvancedUI);
    ImGui::MenuItem("Show Scene Info", "", &m_settings.showStats);
    ImGui::MenuItem("Use Non-Pipeline Mode", nullptr, &m_settings.nonpipelineUI);
    ImGui::Separator();
    ImGui::MenuItem("V-Sync", "Ctrl+Shift+V", &v_sync);
    ImGui::Separator();
    ImGui::EndMenu();
  }

  if(ImGui::BeginMenu("Help"))
  {
    // #TODO: Add Help
    ImGui::MenuItem("About", nullptr, &show_about);

#ifdef _DEBUG
    ImGui::MenuItem("Show Demo", nullptr, &show_demo);
#endif  // _DEBUG
    ImGui::EndMenu();
  }

  if(m_settings.activtyStatus.isBusy())
    return;

  // Shortcuts
  load_file |= ImGui::Shortcut(ImGuiMod_Shortcut | ImGuiKey_O, 0U, ImGuiInputFlags_RouteAlways);
  save_file |= ImGui::Shortcut(ImGuiMod_Shortcut | ImGuiKey_S, 0U, ImGuiInputFlags_RouteAlways);
  close_app |= ImGui::Shortcut(ImGuiMod_Shortcut | ImGuiKey_Q, 0U, ImGuiInputFlags_RouteAlways);
  v_sync |= ImGui::Shortcut(ImGuiMod_Shortcut | ImGuiKey_V, 0U, ImGuiInputFlags_RouteAlways) ? !v_sync : v_sync;

  if(load_file)
  {
    std::string filename = NVPSystem::windowOpenFileDialog(m_app->getWindowHandle(), "Load glTF | OBJ | HDR",
                                                           "glTF(.gltf, .glb), OBJ(.obj), "
                                                           "HDR(.hdr)|*.gltf;*.glb;*.hdr;*.obj");
    onFileDrop(filename.c_str());
  }

  if(save_file)
  {
    std::string filename =
        NVPSystem::windowSaveFileDialog(m_app->getWindowHandle(), "Save glTF", "glTF(.gltf, .glb)|*.gltf;*.glb;");
    if(!filename.empty())
      saveScene(filename);
  }

  if(close_app)
  {
    m_app->close();
  }

  uiAbout(&show_about);

#ifdef _DEBUG
  if(show_demo)
  {
    ImGui::ShowDemoWindow(&show_demo);
  }
#endif  // DEBUG

  if(m_app->isVsync() != v_sync)
  {
    m_app->setVsync(v_sync);
  }
}

//--------------------------------------------------------------------------------------------------
// Called by Application when Drag & Drop happened
//
void ToolboxViewer::onFileDrop(const char* filename)
{
  if(m_settings.activtyStatus.isBusy())
    return;

  namespace fs = std::filesystem;
  m_settings.activtyStatus.activate("Loading File");
  const std::string tfile = filename;

  vkDeviceWaitIdle(m_device);

  std::thread([&, tfile]() {
    const std::string extension = fs::path(tfile).extension().string();
    if(extension == ".gltf" || extension == ".glb" || extension == ".obj")
    {
      m_settings.geometryView.slot  = ViewerSettings::eReference;
      m_settings.geometryView.baked = false;
      createScene(tfile, SceneVersion::eReference);
    }
    else if(extension == ".hdr")
    {
      createHdr(tfile);
      m_settings.envSystem = ViewerSettings::eHdr;
    }

    m_settings.activtyStatus.stop();
  }).detach();
}

//--------------------------------------------------------------------------------------------------
// Called by Application for rendering UI elements
// Note: the final rendered frame is a UI element, it is an image covering up the Viewport
//
void ToolboxViewer::onUIRender()
{
  using PE = ImGuiH::PropertyEditor;

  // Ui Elements
  static UiMicromeshProcess         micro_process(this);
  static UiMicromeshProcessPipeline micro_process_pipeline(this);
  static UiRendering                ui_rendering(this);
  static UiRaytracing               ui_raytracing(this);
  static UiRaster                   ui_raster(this);
  static UiEnvironment              ui_environment(this);

  bool reset{false};

  // Pick under mouse cursor
  if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) || ImGui::IsKeyPressed(ImGuiKey_Space))
  {
    screenPicking();
  }

  // Setting menu
  if(ImGui::Begin("Settings"))
  {
    if(ImGui::CollapsingHeader("Camera"))
      ImGuiH::CameraWidget();

    ViewerSettings::RenderSystem rs = m_settings.renderSystem;
    if(ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen))
      reset |= ui_rendering.onUI(m_settings);

    if(rs != m_settings.renderSystem)
    {
      // Force recreation of G-Buffers because raster used 2x the size and display downscaled
      // making cheap AA
      onResize(m_app->getViewportSize().width, m_app->getViewportSize().height);  // Force recreation of G-Buffers
    }

    switch(m_settings.renderSystem)
    {
      case ViewerSettings::ePathtracer:
        if(ImGui::CollapsingHeader("Raytracing", ImGuiTreeNodeFlags_DefaultOpen))
          reset |= ui_raytracing.onUI(m_settings);
        break;
      case ViewerSettings::eRaster:
        if(ImGui::CollapsingHeader("Raster", ImGuiTreeNodeFlags_DefaultOpen))
          reset |= ui_raster.onUI(m_settings);
        break;
    }

    if(ImGui::CollapsingHeader("Environment"))
    {
      reset |= ui_environment.onUI(m_settings);
    }

    if(ImGui::CollapsingHeader("Tonemapper"))
    {
      ImGui::PushID("Tonemapper");
      reset |= m_tonemapper->onUI();
      ImGui::PopID();
    }
  }
  ImGui::End();  // Settings

  // Micromesh-Processing Pipeline UI
  if(m_settings.nonpipelineUI)
    micro_process.onUI();
  else
    micro_process_pipeline.onUI();

  if(reset)
  {
    resetFrame();
  }


  if(m_settings.showStats)
  {  // Opening a window
    UiStatistics r;
    if(ImGui::Begin("Statistics", &m_settings.showStats))
    {
      ToolboxScene* toolbox_scene = getScene(m_settings.geometryView.slot);
      r.onUI(toolbox_scene->getToolScene().get());
    }
    ImGui::End();
  }

  {  // Rendering Viewport
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("Viewport");

    reset |= keyShortcuts();

    // Display the G-Buffer image
    ImGui::Image(m_gBuffers->getDescriptorSet(eLdr), ImGui::GetContentRegionAvail());

    // Adding Axis at the bottom left corner of the viewport
    if(m_settings.showAxis)
    {
      float  size        = 25.F;
      ImVec2 window_pos  = ImGui::GetWindowPos();
      ImVec2 window_size = ImGui::GetWindowSize();
      ImVec2 offset      = ImVec2(size * 1.1F, -size * 1.1F) * ImGui::GetWindowDpiScale();
      ImVec2 pos         = ImVec2(window_pos.x, window_pos.y + window_size.y) + offset;
      ImGuiH::Axis(pos, CameraManip.getMatrix(), size);
    }

    m_frameInfo.mouseCoord = {-1, -1};
#ifdef _DEBUG
    // #debug_printf : Pick the mouse coordinate in Viewport coordinated when the mouse button is down
    if(ImGui::IsMouseClicked(ImGuiMouseButton_Left))
    {
      const nvmath::vec2f mouse_pos  = ImGui::GetMousePos();   // Current mouse pos in window
      const nvmath::vec2f window_pos = ImGui::GetWindowPos();  // Corner of the viewport
      m_frameInfo.mouseCoord         = mouse_pos - window_pos;
      m_frameInfo.mouseCoord *= RASTER_SS_SIZE;
      reset = true;
    }
#endif  // _DEBUG

    if(reset)
      resetFrame();

    ImGui::End();
    ImGui::PopStyleVar();
  }

  showBusyWindow(m_settings.activtyStatus.status());
}

//--------------------------------------------------------------------------------------------------
// Deal with all shortcut key press
//
bool ToolboxViewer::keyShortcuts()
{
  bool reset{false};

  {
    // Shortcut Keys: F1, F2, F3
    // Changing Geometry and Overlay
    auto& view  = ImGui::IsKeyDown(ImGuiKey_ModShift) ? m_settings.overlayView : m_settings.geometryView;
    auto  bview = view;
    if(ImGui::IsKeyPressed(ImGuiKey_F1))
    {
      view.slot  = ViewerSettings::eReference;
      view.baked = false;
    }
    if(ImGui::IsKeyPressed(ImGuiKey_F2))
    {
      view.slot  = ViewerSettings::eBase;
      view.baked = false;
    }
    if(ImGui::IsKeyPressed(ImGuiKey_F3))
    {
      view.slot  = ViewerSettings::eBase;
      view.baked = true;
    }

    if((bview.slot != view.slot) || (bview.baked != view.baked))
    {
      setAllDirty(SceneDirtyFlags::eRasterRecord);
      reset = true;
    }
  }

  {
    // Shortcut Keys: F5, F6, F7
    // Changing Shading Display
    auto& shading  = m_settings.shading;
    auto  bshading = shading;
    if(ImGui::IsKeyPressed(ImGuiKey_F5))
      shading = eRenderShading_default;
    if(ImGui::IsKeyPressed(ImGuiKey_F6))
      shading = eRenderShading_faceted;
    if(ImGui::IsKeyPressed(ImGuiKey_F7))
      shading = eRenderShading_phong;

    if(bshading != m_settings.shading)
    {
      setAllDirty(SceneDirtyFlags::eRasterPipeline);
      reset = true;
    }
  }

  if(ImGui::Shortcut(ImGuiKey_R))  // Need to be in focus with viewport
  {
    m_settings.renderSystem =
        m_settings.renderSystem == ViewerSettings::ePathtracer ? ViewerSettings::eRaster : ViewerSettings::ePathtracer;
    onResize(m_app->getViewportSize().width, m_app->getViewportSize().height);  // Force recreation of G-Buffers
    reset = true;
  }

  return reset;
}

//--------------------------------------------------------------------------------------------------
// Called by Application and passing the current frame command buffer.
// This is where all Vulkan related element should be rendered
//
void ToolboxViewer::onRender(VkCommandBuffer cmd)
{
  ToolboxScene* toolbox_scene = getScene(m_settings.geometryView.slot);

  if(!toolbox_scene->valid())
  {
    VkClearColorValue       clear_value{0.F, 0.F, 0.F, 0.F};
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, m_gBuffers->getColorImage(eLdr), VK_IMAGE_LAYOUT_GENERAL, &clear_value, 1, &range);
    return;
  }

  if(!updateFrame() || m_settings.activtyStatus.isBusy())
  {
    return;
  }

  m_profilerVk->beginFrame();

  // Dirty Flags
  updateDirty();


  nvvk::DebugUtil::ScopedCmdLabel scope_dbg = m_dutil->DBG_SCOPE(cmd);

  // Update the frame info buffer to the device
  updateFrameInfo(cmd);

  updateHbao();


  // Push constant
  m_pushConst.maxDepth        = m_settings.maxDepth;
  m_pushConst.maxSamples      = m_settings.maxSamples;
  m_pushConst.frame           = m_frame;
  m_pushConst.bakeSubdivLevel = m_settings.tools.subdivLevel;

  // Update the sky
  m_sky->skyParams().directionUp = CameraManip.getUp();
  m_sky->updateParameterBuffer(cmd);

  if(m_settings.renderSystem == ViewerSettings::ePathtracer)
  {
    raytraceScene(cmd);
  }
  else
  {
    rasterScene(cmd);
  }

  // Apply tonemapper - take GBuffer-1 and output to GBuffer-0
  m_tonemapper->runCompute(cmd, m_gBuffers->getSize());

  m_profilerVk->endFrame();
}

void ToolboxViewer::updateHbao()
{
  // hbao setup
  if(!m_settings.hbao.active)
    return;

  const nvmath::vec2f& clip     = CameraManip.getClipPlanes();
  auto&                hbaoView = m_settings.hbao.settings.view;
  hbaoView.farPlane             = clip.y;
  hbaoView.nearPlane            = clip.x;
  hbaoView.isOrtho              = false;
  hbaoView.projectionMatrix     = m_frameInfo.proj;
  m_settings.hbao.settings.radius = getScene(m_settings.geometryView.slot)->getDimensions()->radius * m_settings.hbao.radius;
  vec4 hi = m_frameInfo.projInv * vec4(1, 1, -0.9, 1);
  hi /= hi.w;
  float tany           = hi.y / fabsf(hi.z);
  hbaoView.halfFovyTan = tany;
}

//--------------------------------------------------------------------------------------------------
// The frame buffer, is a buffer that is updated at each frame.
//
void ToolboxViewer::updateFrameInfo(VkCommandBuffer cmd)
{
  const nvmath::vec2f& clip = CameraManip.getClipPlanes();

  m_frameInfo.view         = CameraManip.getMatrix();
  m_frameInfo.proj         = nvmath::perspectiveVK(CameraManip.getFov(), m_gBuffers->getAspectRatio(), clip.x, clip.y);
  m_frameInfo.projInv      = nvmath::inverse(m_frameInfo.proj);
  m_frameInfo.viewInv      = nvmath::inverse(m_frameInfo.view);
  m_frameInfo.metallic     = m_settings.metallic;
  m_frameInfo.roughness    = m_settings.roughness;
  m_frameInfo.colormap     = m_settings.colormap;
  m_frameInfo.vectorLength = m_settings.vectorLength;

  vec3 linear = toLinear(vec3(m_settings.overlayColor.x, m_settings.overlayColor.y, m_settings.overlayColor.z));
  m_frameInfo.overlayColor = ImGui::ColorConvertFloat4ToU32({linear.x, linear.y, linear.z, 1.0F});

  if(m_settings.envSystem == ViewerSettings::eSky)
  {
    m_frameInfo.useSky       = 1;
    m_frameInfo.nbLights     = static_cast<int>(m_settings.lights.size());
    m_frameInfo.light[0]     = m_sky->getSun();
    m_frameInfo.maxLuminance = m_sky->skyParams().intensity * m_sky->skyParams().brightness;
  }
  else
  {
    m_frameInfo.useSky       = 0;
    m_frameInfo.nbLights     = 0;
    m_frameInfo.envColor     = m_settings.envColor;
    m_frameInfo.envRotation  = m_settings.envRotation;
    m_frameInfo.maxLuminance = m_hdrEnv->getIntegral();
  }

  vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(FrameInfo), &m_frameInfo);

  // Barrier
  // Make sure the buffer is available when using it
  VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
  mb.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV,
                       VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Updating and any pipeline of system that could have been dirty
//
void ToolboxViewer::updateDirty()
{
  ToolboxScene* toolbox_scene = getScene(m_settings.geometryView.slot);
  if(toolbox_scene->noneDirty())
    return;

  // Will be doing only one wait, even if multiple elements are dirty
  bool wait_done = false;
  auto waitFct   = [&] {
    if(wait_done)
      return;
    wait_done = true;
    vkDeviceWaitIdle(m_device);
  };

  if(toolbox_scene->isDirty(SceneDirtyFlags::eDeviceMesh))
  {
    toolbox_scene->createVulkanBuffers();
  }


  // Create/Rebuild for RTX
  if(m_settings.renderSystem == ViewerSettings::ePathtracer)
  {
    if(toolbox_scene->isDirty(SceneDirtyFlags::eRtxPipeline))
    {
      waitFct();
      toolbox_scene->createRtxPipeline({m_sky->getDescriptorSetLayout(), m_hdrEnv->getDescriptorSetLayout()});
    }

    if(toolbox_scene->isDirty(SceneDirtyFlags::eRtxAccelerations))
    {
      waitFct();
      toolbox_scene->createRtxAccelerations(m_settings.geometryView.baked);
    }
  }

  // Create/Rebuild for Raster
  if(m_settings.renderSystem == ViewerSettings::eRaster)
  {
    if(toolbox_scene->isDirty(SceneDirtyFlags::eRasterPipeline))
    {
      waitFct();
      toolbox_scene->createRasterPipeline(m_settings, {m_hdrDome->getDescLayout(), m_sky->getDescriptorSetLayout()},
                                          m_gBuffers->getColorFormat(), m_gBuffers->getDepthFormat());
    }

    if(toolbox_scene->isDirty(SceneDirtyFlags::eRasterRecord))
    {
      waitFct();
      toolbox_scene->freeRecordCommandBuffer();
    }
  }

  if(toolbox_scene->isDirty(SceneDirtyFlags::eDescriptorSets))
  {
    waitFct();
    VkDescriptorBufferInfo frameInfo{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorImageInfo  out_image = m_gBuffers->getDescriptorImageInfo(eResult);
    toolbox_scene->writeSets(out_image, frameInfo);
  }
}

//--------------------------------------------------------------------------------------------------
// Returning the scene based on the scene to visualize
//
ToolboxScene* ToolboxViewer::getScene(ViewerSettings::RenderViewSlot v)
{
  // Unless UI ask for Modified, the Reference is displayed
  switch(v)
  {
    case ViewerSettings::eReference:
      return getScene(SceneVersion::eReference);
    case ViewerSettings::eBase:
      return getScene(SceneVersion::eBase);
    case ViewerSettings::eScratch:
      return getScene(SceneVersion::eScratch);
  }
  // Default
  return getScene(SceneVersion::eReference);
}

//--------------------------------------------------------------------------------------------------
// Saving the base scene
//
void ToolboxViewer::saveScene(const std::string& filename, SceneVersion s)
{
  // Making sure the extension is either .glb or .gltf
  fs::path saveName = filename;
  if(saveName.extension() != ".glb")
  {
    saveName.replace_extension(".gltf");
  }

  m_scenes[s]->getToolScene()->save(saveName.string());
}

//--------------------------------------------------------------------------------------------------
// Load a glTF scene and create its various representations.
// - m_scene hold the raw glTF scene
// - m_sceneVk holds the glTF scene on the GPU in Vulkan buffers
// - m_sceneRtx has the Vulkan acceleration structure of the glTF
//
void ToolboxViewer::createScene(const std::string& filename, SceneVersion sceneVersion)
{
  ToolboxScene* toolbox_scene = getScene(sceneVersion);
  toolbox_scene->createFromFile(filename);

  if(!toolbox_scene->valid())
    return;

  // When loading a new reference, we are not keeping the Base
  // TL: I think it isn't a good idea. We should be able to keep the Base
  //if(sceneVersion == SceneVersion::eReference)
  //  getScene(SceneVersion::eBase)->destroy();

  // Find the size of the vectors (normals, directions) for raster rendering
  m_settings.vectorLength = toolbox_scene->getDimensions()->radius * 0.01F;

  // If the scene we are loading contains Bary data, we want to display it
  // and if there are no normal map, we use the faceted mode.
  if(toolbox_scene->hasBary())
  {
    m_settings.geometryView.baked = true;

    bool has_normalmap = false;
    for(auto& mat : toolbox_scene->getToolScene()->materials())
    {
      has_normalmap |= mat.normalTexture.index > -1;
    }
    if(has_normalmap == false)
      m_settings.shading = eRenderShading_faceted;
  }
}

//--------------------------------------------------------------------------------------------------
// Create all G-Buffers needed when rendering the scene
//
void ToolboxViewer::createGbuffers(const nvmath::vec2f& size)
{
  static VkFormat depth_format = nvvk::findDepthFormat(m_app->getPhysicalDevice());  // Not all depth are supported

  m_viewSize = size;

  // For raster we are rendering in a 2x image, which is making nice AA
  if(m_settings.renderSystem == ViewerSettings::eRaster && ImGui::GetWindowDpiScale() <= 1.0F)
  {
    m_viewSize *= RASTER_SS_SIZE;
  }

  VkExtent2D buffer_size = {static_cast<uint32_t>(m_viewSize.x), static_cast<uint32_t>(m_viewSize.y)};

  // Two GBuffers: RGBA8 and RGBA32F, rendering to RGBA32F and tone mapped to RGBA8
  const std::vector<VkFormat> color_buffers = {m_ldrFormat, m_resultFormat};
  // Creation of the GBuffers
  m_gBuffers->destroy();
  m_gBuffers->create(buffer_size, color_buffers, depth_format);

  m_sky->setOutImage(m_gBuffers->getDescriptorImageInfo(eResult));
  m_hdrDome->setOutImage(m_gBuffers->getDescriptorImageInfo(eResult));

  if(m_settings.renderSystem == ViewerSettings::eRaster)
  {
    VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
    HbaoPass::FrameConfig config{};
    config.blend                   = true;
    config.sourceHeightScale       = 1;
    config.sourceWidthScale        = 1;
    config.targetWidth             = buffer_size.width;
    config.targetHeight            = buffer_size.height;
    config.sourceDepth.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    config.sourceDepth.imageView   = m_gBuffers->getDepthImageView();
    config.sourceDepth.sampler     = VK_NULL_HANDLE;
    config.targetColor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    config.targetColor.imageView   = m_gBuffers->getColorImageView(eResult);
    config.targetColor.sampler     = VK_NULL_HANDLE;

    m_hbao->initFrame(m_settings.hbao.frame, config, cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
  }


  // Indicate the renderer to reset its frame
  resetFrame();

  // Need to clear because the viewport size is part of the record
  // #TODO : could it be replaced by inheritance extension to avoid this ?
  for(auto& s : m_scenes)
    s->freeRecordCommandBuffer();
}

//--------------------------------------------------------------------------------------------------
// Create extra Vulkan buffer data
void ToolboxViewer::createVulkanBuffers()
{
  auto lock = GetVkQueueOrAllocatorLock();

  VkCommandBuffer cmd = m_app->createTempCmdBuffer();

  // Create the buffer of the current frame, changing at each frame
  m_bFrameInfo = m_alloc->createBuffer(sizeof(FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  m_dutil->DBG_NAME(m_bFrameInfo.buffer);

  m_pixelBuffer = m_alloc->createBuffer(sizeof(float) * 4, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  m_dutil->DBG_NAME(m_pixelBuffer.buffer);

  m_app->submitAndWaitTempCmdBuffer(cmd);
}

//--------------------------------------------------------------------------------------------------
// If the camera matrix has changed, resets the frame.
// otherwise, increments frame.
//
bool ToolboxViewer::updateFrame()
{
  static nvmath::mat4f ref_cam_matrix;
  static float         ref_fov{CameraManip.getFov()};

  const nvmath::mat4f& m   = CameraManip.getMatrix();
  const float          fov = CameraManip.getFov();

  if(memcmp(&ref_cam_matrix.a00, &m.a00, sizeof(nvmath::mat4f)) != 0 || ref_fov != fov)
  {
    resetFrame();
    ref_cam_matrix = m;
    ref_fov        = fov;
  }

  int pre_frame = m_frame;

  switch(m_settings.renderSystem)
  {
    case ViewerSettings::eRaster:
      if(m_frame < 0)
        ++m_frame;
      break;
    case ViewerSettings::ePathtracer:
      if(m_frame < m_settings.maxFrames)
        ++m_frame;
  }

  return pre_frame != m_frame;
}

//--------------------------------------------------------------------------------------------------
// To be call when renderer need to re-start
//
void ToolboxViewer::resetFrame()
{
  m_frame = -1;
}

//--------------------------------------------------------------------------------------------------
// Change the window title to display real-time informations
//
void ToolboxViewer::windowTitle()
{
  // Window Title
  static float dirty_timer = 0.0F;
  dirty_timer += ImGui::GetIO().DeltaTime;
  if(dirty_timer > 1.0F)  // Refresh every seconds
  {
    const VkExtent2D&     size = m_app->getViewportSize();  // m_gBuffers->getSize();
    std::array<char, 256> buf{};

    const std::string scene_name = getScene(m_settings.geometryView.slot)->getPathName().filename().string();
    const int ret = snprintf(buf.data(), buf.size(), "Micromesh Toolbox: %s | %dx%d | %d FPS / %.3fms | Frame %d",
                             scene_name.c_str(), static_cast<int>(size.width), static_cast<int>(size.height),
                             static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate, m_frame);
    if(ret > 0)
      glfwSetWindowTitle(m_app->getWindowHandle(), buf.data());
    dirty_timer = 0;
  }
}

//--------------------------------------------------------------------------------------------------
// Calling the path tracer RTX
//
void ToolboxViewer::raytraceScene(VkCommandBuffer cmd)
{
  const nvvk::DebugUtil::ScopedCmdLabel scope_dbg     = m_dutil->DBG_SCOPE(cmd);
  ToolboxScene*                         toolbox_scene = getScene(m_settings.geometryView.slot);

  const nvvkhl::PipelineContainer& pipeline = toolbox_scene->getRtxPipeline();

  // Ray trace
  std::vector<VkDescriptorSet> desc_sets{toolbox_scene->getRtxDescSet(), toolbox_scene->getDescSet(),
                                         m_sky->getDescriptorSet(), m_hdrEnv->getDescriptorSet()};
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.plines[0]);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.layout, 0,
                          static_cast<uint32_t>(desc_sets.size()), desc_sets.data(), 0, nullptr);
  vkCmdPushConstants(cmd, pipeline.layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant), &m_pushConst);

  const std::array<VkStridedDeviceAddressRegionKHR, 4>& regions = toolbox_scene->getSbtRegions();
  const VkExtent2D&                                     size    = m_gBuffers->getSize();
  vkCmdTraceRaysKHR(cmd, regions.data(), &regions[1], &regions[2], &regions[3], size.width, size.height, 1);

  // Making sure the rendered image is ready to be used
  VkImage              out_image = m_gBuffers->getColorImage(eResult);
  VkImageMemoryBarrier image_memory_barrier =
      nvvk::makeImageMemoryBarrier(out_image, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                   VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                       0, nullptr, 1, &image_memory_barrier);
}


//--------------------------------------------------------------------------------------------------
// Recording in a secondary command buffer, the raster rendering of the scene.
//
void ToolboxViewer::recordRasterScene(VkCommandBuffer& scnCmd)
{
  VkFormat color_format = m_gBuffers->getColorFormat(eResult);  // Using the RGBA32F

  VkCommandBufferInheritanceRenderingInfoKHR inheritance_rendering_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO_KHR};
  inheritance_rendering_info.colorAttachmentCount    = 1;
  inheritance_rendering_info.pColorAttachmentFormats = &color_format;
  inheritance_rendering_info.depthAttachmentFormat   = m_gBuffers->getDepthFormat();
  inheritance_rendering_info.rasterizationSamples    = VK_SAMPLE_COUNT_1_BIT;

  VkCommandBufferInheritanceInfo inherit_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
  inherit_info.pNext = &inheritance_rendering_info;

  VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
  begin_info.pInheritanceInfo = &inherit_info;
  vkBeginCommandBuffer(scnCmd, &begin_info);
  renderRasterScene(scnCmd);
  vkEndCommandBuffer(scnCmd);
}

//--------------------------------------------------------------------------------------------------
// Rendering the GLTF nodes (instances) contained in the list
// The list should be: solid, blend-able, all
void ToolboxViewer::renderNodes(VkCommandBuffer              cmd,
                                const std::vector<uint32_t>& nodeIDs,
                                ToolboxScene*                toolbox_scene,
                                int                          numIndexed /*= 1*/,
                                int                          numDraw /*= 0*/,
                                bool                         useMeshTask /*=false*/)
{
  nvvk::DebugUtil::ScopedCmdLabel scope_dbg = m_dutil->DBG_SCOPE(cmd);
  const VkDeviceSize              offsets   = 0;

  const std::unique_ptr<micromesh_tool::ToolScene>&                tool_scene    = toolbox_scene->getToolScene();
  const std::unique_ptr<ToolboxSceneVk>&                           tool_scene_vk = toolbox_scene->getToolSceneVK();
  meshops::ArrayView<micromesh_tool::ToolScene::PrimitiveInstance> prim_inst     = tool_scene->instances();
  const std::vector<std::unique_ptr<micromesh_tool::ToolMesh>>&    meshes        = tool_scene->meshes();
  const nvvkhl::PipelineContainer&                                 pipeline      = toolbox_scene->getRasterPipeline();

  for(const uint32_t& node_id : nodeIDs)
  {
    const micromesh_tool::ToolScene::PrimitiveInstance& instance   = prim_inst[node_id];
    uint32_t                                            ref_id     = instance.primMeshRef;
    const std::unique_ptr<micromesh_tool::ToolMesh>&    mesh       = meshes[ref_id];
    int32_t                                             baryIndex  = mesh->relations().bary;
    int32_t                                             groupIndex = mesh->relations().group;

    m_pushConst.materialID     = std::max(0, prim_inst[node_id].material);
    m_pushConst.instanceID     = static_cast<int>(node_id);
    m_pushConst.primMeshID     = static_cast<int>(ref_id);
    m_pushConst.microMax       = 0;
    m_pushConst.microScaleBias = {1.0f, 0.0f};
    m_pushConst.baryInfoID     = 0;

    const meshops::DeviceMesh& device_mesh  = tool_scene_vk->deviceMesh(ref_id);
    meshops::DeviceMeshVK*     device_vk    = meshops::meshopsDeviceMeshGetVK(device_mesh);
    int32_t                    index_count  = static_cast<int32_t>(mesh->view().indexCount());
    int32_t                    vertex_count = static_cast<int32_t>(mesh->view().vertexCount());

    // All buffers sent to the vertex shader
    const VkBuffer& vbuffer = device_vk->vertexPositionNormalBuffer.buffer;

    VkShaderStageFlags stages = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_TASK_BIT_NV | VK_SHADER_STAGE_MESH_BIT_NV;

    if(useMeshTask)
    {
      if(baryIndex != -1 && groupIndex != -1)
      {
        const bary::BasicView&             basic     = tool_scene->barys()[baryIndex]->groups()[groupIndex].basic;
        const std::vector<DeviceMicromap>& micromaps = tool_scene_vk->barys()[baryIndex]->micromaps();
        assert(static_cast<size_t>(groupIndex) < micromaps.size());
        const DeviceMicromap& micromap = micromaps[groupIndex];
        if(micromap.raster())
        {
          m_pushConst.microMax   = micromap.raster()->micromeshSet.meshDatas[0].microTriangleCount - 1;
          m_pushConst.baryInfoID = tool_scene_vk->baryInfoIndex(baryIndex, groupIndex);

          // The bary bias and scale always gets applied to the DeviceMesh
          // bounds buffer, since the raytracing API has no global option.
          // Micromeshes should always have direction bounds anyway for perf
          // reasons (whether they're baked into the positions/directions or
          // not) and a global bias and scale is unusual.
          if(!device_vk->vertexDirectionBoundsBuffer.buffer)
            m_pushConst.microScaleBias = {basic.groups[0].floatScale.r, basic.groups[0].floatBias.r};

          vkCmdPushConstants(cmd, pipeline.layout, stages, 0, sizeof(PushConstant), &m_pushConst);

          // Use mesh shaders to generate tessellated geometry for meshes with
          // micromesh displacement
          int numBaseTriangles = index_count / 3;
          int numWorkgroups    = (numBaseTriangles + MICRO_GROUP_SIZE - 1) / MICRO_GROUP_SIZE;
          vkCmdDrawMeshTasksNV(cmd, numWorkgroups, 0);
        }
      }
    }
    else
    {
      vkCmdPushConstants(cmd, pipeline.layout, stages, 0, sizeof(PushConstant), &m_pushConst);
      vkCmdBindVertexBuffers(cmd, 0, 1, &vbuffer, &offsets);
      vkCmdBindIndexBuffer(cmd, device_vk->triangleVertexIndexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(cmd, index_count, numIndexed, 0, 0, 0);

      // Then we draw the directions from the inner to the outer shell by  performing a non-indexed draw of 3*vertexCount vertices, and
      // setting baseInstance to 2 to signal to the vertex shader that  we're indexing in this special way.
      // We use a factor of 3 so that we can continue to use the triangle topology in this pipeline.
      vkCmdDraw(cmd, 3 * vertex_count, numDraw, 0, 2 /* firstInstance */);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Render the entire scene for raster. Splitting the solid and blend-able element and rendering
// on top, the wireframe if active.
// This is done in a recoded command buffer to be replay
void ToolboxViewer::renderRasterScene(VkCommandBuffer cmd)
{
  nvvk::DebugUtil::ScopedCmdLabel scope_dbg     = m_dutil->DBG_SCOPE(cmd);
  ToolboxScene*                   toolbox_scene = getScene(m_settings.geometryView.slot);
  ToolboxScene*                   overlay_scene = getScene(m_settings.overlayView.slot);
  ToolboxScene*                   shell_scene   = getScene(m_settings.shellView.slot);

  const nvvkhl::PipelineContainer& pipeline         = toolbox_scene->getRasterPipeline();
  const nvvkhl::PipelineContainer& overlay_pipeline = overlay_scene->getRasterPipeline();
  const nvvkhl::PipelineContainer& shell_pipeline   = shell_scene->getRasterPipeline();


  const VkExtent2D& render_size = m_gBuffers->getSize();

  const VkViewport viewport{0.0F, 0.0F, static_cast<float>(render_size.width), static_cast<float>(render_size.height),
                            0.0F, 1.0F};
  vkCmdSetViewport(cmd, 0, 1, &viewport);

  const VkRect2D scissor{{0, 0}, {render_size.width, render_size.height}};
  vkCmdSetScissor(cmd, 0, 1, &scissor);


  // Draw solid
  if(m_settings.geometryView.slot != ViewerSettings::eNone)
  {
    std::vector dset = {toolbox_scene->getDescSet(), m_hdrDome->getDescSet(), m_sky->getDescriptorSet()};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0,
                            static_cast<uint32_t>(dset.size()), dset.data(), 0, nullptr);

    // Baked/micromesh draws solid and blend
    bool use_bake = m_settings.geometryView.baked;
    if(use_bake)
    {
      std::vector<uint32_t> nodes = toolbox_scene->getNodes(SceneNodeMethods::eAll, SceneNodeMicromesh::eMicromeshWith);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.plines[RasterPipelines::eRasterPipelineMicromeshSolid]);
      renderNodes(cmd, nodes, toolbox_scene, 0, 0, true /*use mesh*/);
    }

    //
    {
      // Draw solid
      SceneNodeMicromesh micro_method = use_bake ? SceneNodeMicromesh::eMicromeshWithout : SceneNodeMicromesh::eMicromeshDontCare;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.plines[RasterPipelines::eRasterPipelineSolid]);
      std::vector<uint32_t> solid_nodes = toolbox_scene->getNodes(SceneNodeMethods::eSolid, micro_method);
      renderNodes(cmd, solid_nodes, toolbox_scene);

      // Draw blend-able
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.plines[RasterPipelines::eRasterPipelineBlend]);
      std::vector<uint32_t> blend_nodes = toolbox_scene->getNodes(SceneNodeMethods::eBlend, micro_method);
      renderNodes(cmd, blend_nodes, toolbox_scene);
    }
  }

  // Draw overlay
  if(m_settings.overlayView.slot != ViewerSettings::eNone && overlay_scene->valid())
  {
    bool use_bake = m_settings.overlayView.baked;

    std::vector dset = {overlay_scene->getDescSet(), m_hdrDome->getDescSet(), m_sky->getDescriptorSet()};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, overlay_pipeline.layout, 0,
                            static_cast<uint32_t>(dset.size()), dset.data(), 0, nullptr);
    if(use_bake)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, overlay_pipeline.plines[RasterPipelines::eRasterPipelineMicromeshWire]);
      std::vector<uint32_t> nodes = overlay_scene->getNodes(SceneNodeMethods::eAll, SceneNodeMicromesh::eMicromeshWith);
      renderNodes(cmd, nodes, overlay_scene, 0, 0, true);
    }

    {
      SceneNodeMicromesh micro_method = use_bake ? SceneNodeMicromesh::eMicromeshWithout : SceneNodeMicromesh::eMicromeshDontCare;
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, overlay_pipeline.plines[RasterPipelines::eRasterPipelineWire]);
      renderNodes(cmd, overlay_scene->getNodes(SceneNodeMethods::eAll, micro_method), overlay_scene);
    }
  }

  // Draw shell (same geometry as geometry view)
  if(m_settings.shellView.slot != ViewerSettings::eNone && shell_scene->valid())
  {
    std::vector dset = {shell_scene->getDescSet(), m_hdrDome->getDescSet(), m_sky->getDescriptorSet()};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shell_pipeline.layout, 0,
                            static_cast<uint32_t>(dset.size()), dset.data(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shell_pipeline.plines[RasterPipelines::eRasterPipelineShell]);
    renderNodes(cmd, shell_scene->getNodes(SceneNodeMethods::eAll, SceneNodeMicromesh::eMicromeshDontCare), shell_scene, 2, 1);
  }

  if(m_settings.debugMethod == eDbgMethod_normal || m_settings.debugMethod == eDbgMethod_direction)
  {
    std::vector dset = {toolbox_scene->getDescSet(), m_hdrDome->getDescSet(), m_sky->getDescriptorSet()};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.layout, 0,
                            static_cast<uint32_t>(dset.size()), dset.data(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline.plines[RasterPipelines::eRasterPipelineVector]);
    renderNodes(cmd, toolbox_scene->getNodes(SceneNodeMethods::eAll, SceneNodeMicromesh::eMicromeshDontCare), toolbox_scene, 0, 1);
  }
}

//--------------------------------------------------------------------------------------------------
// Rendering the scene for raster.
// If the scene is viewed in this state for the first time, it will be recorded in a secondary
// buffer which will be replayed the next time.
//
// First it renders in the eResult buffer, the sky or the HDR dome. Then, without clearing the
// color buffer, only the depth buffer, it replays the secondary command buffer.
//
void ToolboxViewer::rasterScene(VkCommandBuffer cmd)
{
  nvvk::DebugUtil::ScopedCmdLabel scope_dbg     = m_dutil->DBG_SCOPE(cmd);
  ToolboxScene*                   toolbox_scene = getScene(m_settings.geometryView.slot);

  // Rendering Dome/Background
  {
    const float          aspect_ratio = m_gBuffers->getAspectRatio();
    const nvmath::mat4f& view         = CameraManip.getMatrix();
    const nvmath::mat4f  proj         = nvmath::perspectiveVK(CameraManip.getFov(), aspect_ratio, 0.1F, 1000.0F);

    VkExtent2D img_size = m_gBuffers->getSize();
    if(m_settings.envSystem == ViewerSettings::eSky)
    {
      m_sky->draw(cmd, view, proj, img_size);
    }
    else
    {
      m_hdrDome->draw(cmd, view, proj, img_size, &m_settings.envColor.x, m_settings.envRotation);
    }
  }

  // Get pre-recorded command buffer to execute faster
  VkCommandBuffer scn_cmd = toolbox_scene->getRecordedCommandBuffer();
  if(scn_cmd == VK_NULL_HANDLE)
  {
    scn_cmd = toolbox_scene->createRecordCommandBuffer();
    recordRasterScene(scn_cmd);
  }

  // Execute recorded command buffer
  {
    // Drawing the primitives in the RGBA32F G-Buffer, don't clear or it will erase the
    nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView(eResult)},
                                     m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_CLEAR,
                                     m_clearColor, {1.0F, 0}, VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT_KHR);
    r_info.pStencilAttachment = nullptr;

    vkCmdBeginRendering(cmd, &r_info);
    vkCmdExecuteCommands(cmd, 1, &scn_cmd);
    vkCmdEndRendering(cmd);
  }

  if(m_settings.hbao.active)
  {
    nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getDepthImage(), VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    m_hbao->cmdCompute(cmd, m_settings.hbao.frame, m_settings.hbao.settings);
    nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getDepthImage(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_DEPTH_BIT);
  }
}

//--------------------------------------------------------------------------------------------------
// Loading an HDR image, create the acceleration structure used by the path tracer and
// create a convoluted version of it (Dome) used by the raster.
void ToolboxViewer::createHdr(const std::string& filename)
{
  auto lock = GetVkQueueOrAllocatorLock();

  const uint32_t c_family_queue = m_app->getContext()->m_queueC.familyIndex;
  m_hdrEnv  = std::make_unique<nvvkhl::HdrEnv>(m_app->getContext().get(), m_alloc.get(), c_family_queue);
  m_hdrDome = std::make_unique<nvvkhl::HdrEnvDome>(m_app->getContext().get(), m_alloc.get(), c_family_queue);

  m_hdrEnv->loadEnvironment(filename);
  m_hdrDome->create(m_hdrEnv->getDescriptorSet(), m_hdrEnv->getDescriptorSetLayout());
  m_hdrDome->setOutImage(m_gBuffers->getDescriptorImageInfo(eResult));


  for(auto& s : m_scenes)
  {
    s->setDirty(eDescriptorSets);
    s->setDirty(eRasterRecord);
  }

  m_frameInfo.maxLuminance = m_hdrEnv->getIntegral();  // Remove firefly
}

//--------------------------------------------------------------------------------------------------
// Destroy all allocated resources
//
void ToolboxViewer::destroyResources()
{
  auto lock = GetVkQueueOrAllocatorLock();
  m_alloc->destroy(m_bFrameInfo);
  m_alloc->destroy(m_pixelBuffer);

  m_gBuffers.reset();


  m_sky->destroy();
  m_picker->destroy();
  m_hbao->deinitFrame(m_settings.hbao.frame);

  m_tonemapper.reset();

  for(auto&& scene : m_scenes)
    scene.reset();
}

//--------------------------------------------------------------------------------------------------
// This goes in the .ini file and remember the information we store
void ToolboxViewer::addSettingsHandler()
{
  // Persisting the toolbox viewer info
  ImGuiSettingsHandler ini_handler{};
  ini_handler.TypeName   = "ToolboxViewer";
  ini_handler.TypeHash   = ImHashStr("ToolboxViewer");
  ini_handler.ClearAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler*) {};
  ini_handler.ApplyAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler*) {};
  ini_handler.ReadOpenFn = [](ImGuiContext*, ImGuiSettingsHandler*, const char* name) -> void* { return (void*)1; };
  ini_handler.ReadLineFn = [](ImGuiContext*, ImGuiSettingsHandler* handler, void* entry, const char* line) {
    ToolboxViewer* viewer = (ToolboxViewer*)handler->UserData;
    int            i1;
    ImU32          u1;
    float          f1, f2;
    // clang-format off
    if(std::sscanf(line, "ShowStats=%d", &i1) == 1)             { viewer->m_settings.showStats = (i1 == 1); }
    else if(std::sscanf(line, "ShowAxis=%d", &i1) == 1)         { viewer->m_settings.showAxis = (i1 == 1);}
    else if(std::sscanf(line, "OverlayColor=0x%X", &u1) == 1)   { viewer->m_settings.overlayColor = ImGui::ColorConvertU32ToFloat4(u1); }
    else if(std::sscanf(line, "Colormap=%d", &i1) == 1)         { viewer->m_settings.colormap = ViewerSettings::ColormapMode(i1);}
    else if(std::sscanf(line, "Metallic/Roughness= %f %f", &f1, &f2) == 2) { viewer->m_settings.metallic = f1; viewer->m_settings.roughness = f2; }
    else if(std::sscanf(line, "NonPipelineMode= %d", &i1) == 1) { viewer->m_settings.nonpipelineUI = i1; }
    // clang-format on
  };
  ini_handler.WriteAllFn = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf) {
    ToolboxViewer* s = (ToolboxViewer*)handler->UserData;
    buf->appendf("[%s][State]\n", handler->TypeName);
    buf->appendf("ShowStats=%d\n", s->m_settings.showStats ? 1 : 0);
    buf->appendf("ShowAxis=%d\n", s->m_settings.showAxis ? 1 : 0);
    buf->appendf("Colormap=%d\n", s->m_settings.colormap);
    buf->appendf("OverlayColor=0x%X\n", ImGui::ColorConvertFloat4ToU32(s->m_settings.overlayColor));
    buf->appendf("Metallic/Roughness= %.3f %.3f\n", s->m_settings.metallic, s->m_settings.roughness);
    buf->appendf("NonPipelineMode= %d\n", s->m_settings.nonpipelineUI);
    buf->appendf("\n");
  };
  ini_handler.UserData = this;
  ImGui::AddSettingsHandler(&ini_handler);
}

//--------------------------------------------------------------------------------------------------
// Return the 3D position of the screen 2D + depth
//
static nvmath::vec3f unprojectScreenPosition(const VkExtent2D&    size,
                                             const nvmath::vec3f& screenPos,
                                             const nvmath::mat4f& view,
                                             const nvmath::mat4f& proj)
{
  // Transformation of normalized coordinates between -1 and 1
  //const VkExtent2D size = m_gBuffers->getSize();
  nvmath::vec4f win_norm;
  win_norm.x = screenPos.x / static_cast<float>(size.width) * 2.0F - 1.0F;
  win_norm.y = screenPos.y / static_cast<float>(size.height) * 2.0F - 1.0F;
  win_norm.z = screenPos.z;
  win_norm.w = 1.0;

  // Transform to world space
  const nvmath::mat4f mat       = proj * view;
  const nvmath::mat4f mat_inv   = nvmath::invert(mat);
  nvmath::vec4f       world_pos = mat_inv * win_norm;
  world_pos.w                   = 1.0F / world_pos.w;
  world_pos.x                   = world_pos.x * world_pos.w;
  world_pos.y                   = world_pos.y * world_pos.w;
  world_pos.z                   = world_pos.z * world_pos.w;

  return nvmath::vec3f(world_pos);
}


//--------------------------------------------------------------------------------------------------
// Send a ray under mouse coordinates, and retrieve the information
// - Set new camera interest point on hit position
//
void ToolboxViewer::screenPicking()
{
  ImGui::Begin("Viewport");  // ImGui, picking within "viewport"
  bool         is_hovered     = ImGui::IsWindowHovered();
  ImVec2       mouse_pos      = ImGui::GetMousePos();
  const ImVec2 main_size      = ImGui::GetContentRegionAvail();
  const ImVec2 corner         = ImGui::GetCursorScreenPos();  // Corner of the viewport
  mouse_pos                   = mouse_pos - corner;
  const ImVec2 norm_mouse_pos = mouse_pos / main_size;
  ImGui::End();

  if(is_hovered)  // Do picking only if the mouse is the viewport window
  {
    if(m_settings.renderSystem == ViewerSettings::ePathtracer)
      rtxPicking(norm_mouse_pos);
    else
      rasterPicking(norm_mouse_pos);
  }
}

//--------------------------------------------------------------------------------------------------
// Using the RTX engine, send a ray and return Hit information
//
void ToolboxViewer::rtxPicking(const ImVec2& mousePosNorm)
{

  ToolboxScene* toolbox_scene = getScene(m_settings.geometryView.slot);
  if(toolbox_scene->getTlas() == VK_NULL_HANDLE)
    return;

  // Finding current camera matrices
  const nvmath::mat4f& view = CameraManip.getMatrix();
  const nvmath::mat4f  proj = nvmath::perspectiveVK(CameraManip.getFov(), m_gBuffers->getAspectRatio(), 0.1F, 1000.0F);

  // Setting up the data to do picking
  VkCommandBuffer              cmd = m_app->createTempCmdBuffer();
  nvvk::RayPickerKHR::PickInfo pick_info;
  pick_info.pickX          = mousePosNorm.x;
  pick_info.pickY          = mousePosNorm.y;
  pick_info.modelViewInv   = nvmath::invert(view);
  pick_info.perspectiveInv = nvmath::invert(proj);

  // Run and wait for result
  m_picker->setTlas(toolbox_scene->getTlas());
  m_picker->run(cmd, pick_info);
  //curScene()->screenPicking(cmd, pick_info);
  m_app->submitAndWaitTempCmdBuffer(cmd);


  // Retrieving picking information
  const nvvk::RayPickerKHR::PickResult pr = m_picker->getResult();
  if(pr.instanceID == ~0)
  {
    LOGI("Nothing Hit\n");
    return;
  }

  if(pr.hitT <= 0.F)
  {
    LOGI("Hit Distance == 0.0\n");
    return;
  }

  // Find where the hit point is and set the interest position
  const nvmath::vec3f world_pos = nvmath::vec3f(pr.worldRayOrigin + pr.worldRayDirection * pr.hitT);
  nvmath::vec3f       eye;
  nvmath::vec3f       center;
  nvmath::vec3f       up;
  CameraManip.getLookat(eye, center, up);
  CameraManip.setLookat(eye, world_pos, up, false);

  auto float_as_uint = [](float f) { return *reinterpret_cast<uint32_t*>(&f); };

  // Logging picking info.
  const std::unique_ptr<micromesh_tool::ToolScene>& tool_scene = toolbox_scene->getToolScene();
  const std::unique_ptr<micromesh_tool::ToolMesh>&  mesh       = tool_scene->meshes()[pr.instanceCustomIndex];

  LOGI("Hit(%d): %s, PrimId: %d, ", pr.instanceCustomIndex, mesh->meta().name.c_str(), pr.primitiveID);
  LOGI("{%3.2f, %3.2f, %3.2f}, Dist: %3.2f\n", world_pos.x, world_pos.y, world_pos.z, pr.hitT);
  LOGI("PrimitiveID: %d\n", pr.primitiveID);
}

//--------------------------------------------------------------------------------------------------
// Find the 3D position under the mouse cursor and set the camera interest to this position
// using the depth buffer
//
void ToolboxViewer::rasterPicking(const ImVec2& mousePosNorm)
{
  const float          aspect_ratio = m_viewSize.x / m_viewSize.y;
  const nvmath::vec2f& clip         = CameraManip.getClipPlanes();
  const nvmath::mat4f  view         = CameraManip.getMatrix();
  const nvmath::mat4f  proj         = nvmath::perspectiveVK(CameraManip.getFov(), aspect_ratio, clip.x, clip.y);

  // Find the distance under the cursor
  int         x = static_cast<int>(static_cast<float>(m_gBuffers->getSize().width) * mousePosNorm.x);
  int         y = static_cast<int>(static_cast<float>(m_gBuffers->getSize().height) * mousePosNorm.y);
  const float d = getDepth(x, y);

  if(d < 1.0F)  // Ignore infinite
  {
    const nvmath::vec3f hit_pos = unprojectScreenPosition(m_gBuffers->getSize(), {x, y, d}, view, proj);

    // Set the interest position
    nvmath::vec3f eye, center, up;
    CameraManip.getLookat(eye, center, up);
    CameraManip.setLookat(eye, hit_pos, up, false);
  }
}


//--------------------------------------------------------------------------------------------------
// Read the depth buffer at the X,Y coordinates
// Note: depth format is VK_FORMAT_D32_SFLOAT
//
float ToolboxViewer::getDepth(int x, int y)
{
  VkCommandBuffer cmd = m_app->createTempCmdBuffer();

  // Transit the depth buffer image in eTransferSrcOptimal
  const VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
  nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getDepthImage(), VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, range);

  // Copy the pixel under the cursor
  VkBufferImageCopy copy_region{};
  copy_region.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
  copy_region.imageOffset      = {x, y, 0};
  copy_region.imageExtent      = {1, 1, 1};
  vkCmdCopyImageToBuffer(cmd, m_gBuffers->getDepthImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_pixelBuffer.buffer, 1, &copy_region);

  // Put back the depth buffer as  it was
  nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getDepthImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                              VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL, range);
  m_app->submitAndWaitTempCmdBuffer(cmd);


  // Grab the value
  float value{1.0F};
  void* mapped = m_alloc->map(m_pixelBuffer);
  switch(m_gBuffers->getDepthFormat())
  {
    case VK_FORMAT_X8_D24_UNORM_PACK32:
    case VK_FORMAT_D24_UNORM_S8_UINT: {
      uint32_t ivalue{0};
      memcpy(&ivalue, mapped, sizeof(uint32_t));
      const uint32_t mask = (1 << 24) - 1;
      ivalue              = ivalue & mask;
      value               = float(ivalue) / float(mask);
    }
    break;
    case VK_FORMAT_D32_SFLOAT: {
      memcpy(&value, mapped, sizeof(float));
    }
    break;
  }
  m_alloc->unmap(m_pixelBuffer);

  return value;
}
