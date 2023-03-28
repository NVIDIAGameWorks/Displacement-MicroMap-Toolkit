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

#include <thread>

#include "toolbox_scene.hpp"
#include "../src/tool_meshops_objects.hpp"
#include "microutils/microutils.hpp"
#include "nvh/timesampler.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/scene_camera.hpp"
#include "shaders/dh_bindings.h"
#include "nvvk/specialization.hpp"


#include "_autogen/pathtrace.rahit.h"
#include "_autogen/pathtrace.rchit.h"
#include "_autogen/pathtrace.rgen.h"
#include "_autogen/pathtrace.rmiss.h"
#include "_autogen/raster.frag.h"
#include "_autogen/raster.vert.h"
#include "_autogen/raster_shell.frag.h"
#include "_autogen/raster_shell.vert.h"
#include "_autogen/raster_overlay.frag.h"
#include "_autogen/raster_vectors.frag.h"
#include "_autogen/raster_vectors.vert.h"
#include "_autogen/draw_compressed_basic.task.glsl.h"
#include "_autogen/draw_compressed_basic.mesh.glsl.h"
#include "_autogen/draw_compressed_basic.frag.glsl.h"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/shaders_vk.hpp"


extern std::shared_ptr<nvvkhl::ElementCamera> g_elem_camera;
namespace fs = std::filesystem;

#ifdef USE_NSIGHT_AFTERMATH
#include "aftermath/nsight_aftermath_gpu_crash_tracker.h"
extern std::unique_ptr<::GpuCrashTracker> g_aftermathTracker;
#endif


ToolboxScene::ToolboxScene(nvvk::Context* ctx, nvvkhl::AllocVma* alloc, nvvk::Context::Queue extraQueue, VkCommandPool cmdPool)
    : m_alloc(alloc)
    , m_qGCT1(extraQueue)
    , m_device(ctx->m_device)
    , m_cmdPool(cmdPool)
    , m_ctx(ctx)
{
  const uint32_t compute_family_index = ctx->m_queueC.familyIndex;

  m_sbt      = std::make_unique<nvvk::SBTWrapper>();                      // Shader Binding Table
  m_rtxSet   = std::make_unique<nvvk::DescriptorSetContainer>(m_device);  // DescSet with TLAS
  m_sceneSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);  // DescSet of the Scene
  m_dutil    = std::make_unique<nvvk::DebugUtil>(m_device);               // Debug utility

  // Micromesh Scene
  meshops::ContextConfig config = {};
  config.messageCallback        = microutils::makeDefaultMessageCallback();
  config.threadCount            = std::thread::hardware_concurrency();
  config.verbosityLevel         = 999;
  config.requiresDeviceContext  = true;

  meshops::ContextVK sharedContextVK;
  sharedContextVK.context  = ctx;
  sharedContextVK.vma      = static_cast<struct VmaAllocator_T*>(alloc->vma());
  sharedContextVK.queueGCT = m_qGCT1;

  meshops::meshopsContextCreateVK(config, sharedContextVK, &m_context);
  //  meshops::meshopsContextCreateVK(config, nullptr, nullptr, &m_context);

  m_toolscene    = std::make_unique<micromesh_tool::ToolScene>();
  m_toolsceneVk  = std::make_unique<ToolboxSceneVk>(ctx, alloc, m_context, m_qGCT1);
  m_toolsceneRtx = std::make_unique<ToolboxSceneRtx>(ctx, alloc, compute_family_index);

  // Requesting ray tracing properties
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR rt_prop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  prop2.pNext = &rt_prop;
  vkGetPhysicalDeviceProperties2(ctx->m_physicalDevice, &prop2);

  // Create utilities to create the Shading Binding Table (SBT)
  m_sbt->setup(ctx->m_device, compute_family_index, alloc, rt_prop);

  m_dirty.set();  // all dirty
}

//--------------------------------------------------------------------------------------------------
// Default destructor
//
ToolboxScene::~ToolboxScene()
{
  destroy();

  // Must be destroyed after m_toolsceneVk
  meshops::meshopsContextDestroy(m_context);
}

void ToolboxScene::destroy()
{
  m_toolscene->destroy();
  m_toolsceneVk->destroy();
  m_toolsceneRtx->destroy();
  m_dirty.set();  // Fully dirty
  m_pathFilename.clear();
  m_sceneStats.reset();

  m_rasterPipe.destroy(m_device);
  m_rtxPipe.destroy(m_device);

  freeRecordCommandBuffer();
  m_sbt->destroy();
  m_rtxSet->deinit();
  m_sceneSet->deinit();
}

//--------------------------------------------------------------------------------------------------
// Creating the scene by loading a file
//
void ToolboxScene::createFromFile(const std::string& filename)
{
  nvh::ScopedTimer st("Create From File: ");

  // Early freeing up memory and resources
  destroy();

  // Loading the scene
  if(m_toolscene->create(filename) != micromesh::Result::eSuccess)
  {
    return;
  }

  // Finding the dimension of the scene
  m_scnDimensions = std::make_unique<micromesh_tool::ToolSceneDimensions>(*m_toolscene);

  // Search the scene's materials to see if any have heightmaps applied
  m_sceneStats = micromesh_tool::ToolSceneStats(*m_toolscene);

  m_pathFilename = filename;

  // Adjusting camera
  setCameraFromScene(m_pathFilename);
}

//--------------------------------------------------------------------------------------------------
// This is updating the DeviceMesh and other Vulkan buffers for displaying the scene
//
// While be called if the flag eDeviceMesh is dirty
//
void ToolboxScene::createVulkanBuffers()
{
  nvh::ScopedTimer st("Create Vulkan Buffers\n");

  assert(m_toolscene->valid());

  // Finding the dimension of the scene
  m_scnDimensions = std::make_unique<micromesh_tool::ToolSceneDimensions>(*m_toolscene);


  {  // Create the Vulkan side of the scene
     // Since we load and display simultaneously, we need to use a second GTC queue
    nvvk::CommandPool cmd_pool(m_device, m_qGCT1.familyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, m_qGCT1.queue);

    {
      VkCommandBuffer cmd = cmd_pool.createCommandBuffer();

      m_toolsceneVk->create(cmd, *m_toolscene);

      cmd_pool.submitAndWait(cmd);
      m_alloc->finalizeAndReleaseStaging();  // Make sure there are no pending staging buffers and clear them up
    }
  }

  // Creating descriptor set and writing values
  createSceneSet();
  createRtxSet();
  setShadeNodes();

  // Clear the dirty flag
  resetDirty(eDeviceMesh);
}

//--------------------------------------------------------------------------------------------------
// Creating the Shader Binding Table for ray tracing
//
void ToolboxScene::createSbt(VkPipeline rtPipeline, VkRayTracingPipelineCreateInfoKHR rayPipelineInfo)
{
  m_sbt->create(rtPipeline, rayPipelineInfo);
}

//--------------------------------------------------------------------------------------------------
// Returns the list of nodes that are
// - shading : opaque, or not, or any shading
// - micromesh: with, without or don't care about it
//
std::vector<uint32_t> ToolboxScene::getNodes(SceneNodeMethods shading, SceneNodeMicromesh micromesh) const
{
  std::vector<uint32_t> nodes;  // return a vector of nodes corresponding on the requested values
  std::bitset<2>        test;

  for(uint32_t node_id = 0; node_id < m_toolscene->instances().size(); node_id++)
  {
    // If we don't care about the value, we use the same as the one from the m_shadeNodes
    test.set(0, shading == SceneNodeMethods::eAll ? m_shadeNodes[node_id][0] : shading == SceneNodeMethods::eSolid);
    test.set(1, micromesh == SceneNodeMicromesh::eMicromeshDontCare ? m_shadeNodes[node_id][1] :
                                                                      micromesh == SceneNodeMicromesh::eMicromeshWith);

    if(test == m_shadeNodes[node_id])
    {
      nodes.push_back(node_id);
    }
  }

  return nodes;
}

//--------------------------------------------------------------------------------------------------
// Writing the information in the descriptor sets
// Will be called if eDescriptorSets is dirty
//
void ToolboxScene::writeSets(VkDescriptorImageInfo& outImage, VkDescriptorBufferInfo& frameInfo)
{
  writeSceneSet(frameInfo);
  writeRtxSet(outImage);
  resetDirty(SceneDirtyFlags::eDescriptorSets);
}

//--------------------------------------------------------------------------------------------------
// Create the RTX acceleration structure using the toolscene and the Vulkan buffer information
// Will be called if eDescriptorSets is dirty
//
void ToolboxScene::createRtxAccelerations(bool useMicroMesh)
{
  bool hasDisplacementMicromeshExt = m_ctx->hasDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME);

  m_toolsceneRtx->create(m_toolscene, m_toolsceneVk, useMicroMesh && hasDisplacementMicromeshExt);  // Create BLAS / TLAS
  resetDirty(SceneDirtyFlags::eRtxAccelerations);

  // When the acceleration structure is created the descriptor sets
  // need to be updated with the TLAS information
  setDirty(SceneDirtyFlags::eDescriptorSets);
}

//--------------------------------------------------------------------------------------------------
// Creating the descriptor set of the path tracer (Set: 0)
//
void ToolboxScene::createRtxSet()
{
  m_rtxSet->deinit();

  // This descriptor set, holds the top level acceleration structure and the output image
  m_rtxSet->addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
  m_rtxSet->addBinding(RtxBindings::eOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
  m_rtxSet->initLayout();
  m_rtxSet->initPool(1);
  m_dutil->DBG_NAME(m_rtxSet->getLayout());
  m_dutil->DBG_NAME(m_rtxSet->getSet());
}

//--------------------------------------------------------------------------------------------------
// Creating the descriptor set of the scene (Set: 1)
//
void ToolboxScene::createSceneSet()
{
  m_sceneSet->deinit();

  // This descriptor set, holds scene information and the textures
  m_sceneSet->addBinding(SceneBindings::eFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_sceneSet->addBinding(SceneBindings::eSceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_sceneSet->addBinding(SceneBindings::eSceneDescTools, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  m_sceneSet->addBinding(SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                         m_toolsceneVk->nbTextures(), VK_SHADER_STAGE_ALL);
  m_sceneSet->initLayout();
  m_sceneSet->initPool(1);
  m_dutil->DBG_NAME(m_sceneSet->getLayout());
  m_dutil->DBG_NAME(m_sceneSet->getSet());
}


//--------------------------------------------------------------------------------------------------
// Updating the descriptor set for the Path tracer (Set: 0)
//
void ToolboxScene::writeRtxSet(VkDescriptorImageInfo& outImage)
{
  if(!valid())
  {
    return;
  }

  // Write to descriptors
  VkAccelerationStructureKHR tlas = m_toolsceneRtx->tlas();
  VkWriteDescriptorSetAccelerationStructureKHR desc_as_info{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  desc_as_info.accelerationStructureCount = 1;
  desc_as_info.pAccelerationStructures    = &tlas;

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(m_rtxSet->makeWrite(0, RtxBindings::eTlas, &desc_as_info));
  writes.emplace_back(m_rtxSet->makeWrite(0, RtxBindings::eOutImage, &outImage));
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Updating the descriptor set of the scene (Set: 1)
//
void ToolboxScene::writeSceneSet(VkDescriptorBufferInfo& frameInfo)
{
  if(!valid())
  {
    return;
  }

  // Write to descriptors
  const VkDescriptorBufferInfo scene_desc{m_toolsceneVk->sceneDesc().buffer, 0, VK_WHOLE_SIZE};
  const VkDescriptorBufferInfo scene_desc_tool{m_toolsceneVk->sceneDesc().buffer, 0, VK_WHOLE_SIZE};

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(m_sceneSet->makeWrite(0, SceneBindings::eFrameInfo, &frameInfo));
  writes.emplace_back(m_sceneSet->makeWrite(0, SceneBindings::eSceneDesc, &scene_desc));
  writes.emplace_back(m_sceneSet->makeWrite(0, SceneBindings::eSceneDescTools, &scene_desc_tool));
  std::vector<VkDescriptorImageInfo> diit;
  for(const nvvk::Texture& texture : m_toolsceneVk->textures())  // All texture samplers
  {
    diit.emplace_back(texture.descriptor);
  }
  writes.emplace_back(m_sceneSet->makeWriteArray(0, SceneBindings::eTextures, diit.data()));

  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Raster commands are recorded to be replayed, this allocates that command buffer
//
VkCommandBuffer ToolboxScene::createRecordCommandBuffer()
{
  VkCommandBufferAllocateInfo alloc_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  alloc_info.commandPool        = m_cmdPool;
  alloc_info.level              = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
  alloc_info.commandBufferCount = 1;
  vkAllocateCommandBuffers(m_device, &alloc_info, &m_recordedSceneCmd);
  return m_recordedSceneCmd;
}


//--------------------------------------------------------------------------------------------------
// Freeing the raster recoded command buffer
// Will be called if eRasterRecord is dirty
//
void ToolboxScene::freeRecordCommandBuffer()
{
  vkFreeCommandBuffers(m_device, m_cmdPool, 1, &m_recordedSceneCmd);

  m_recordedSceneCmd = VK_NULL_HANDLE;
  resetDirty(SceneDirtyFlags::eRasterRecord);
}

// Adjusting camera
void ToolboxScene::setCameraFromScene(const std::filesystem::path& filename)
{
  // Re-adjusting camera to fit the new scene
  CameraManip.fit(m_scnDimensions->min, m_scnDimensions->max, true);
  CameraManip.setClipPlanes(nvmath::vec2f(0.001F * m_scnDimensions->radius, 100.0F * m_scnDimensions->radius));

  // UI - camera
  ImGuiH::SetCameraJsonFile(filename.stem().string());
  ImGuiH::SetHomeCamera(CameraManip.getCamera());

  // Adjustment of camera navigation speed
  g_elem_camera->setSceneRadius(m_scnDimensions->radius);
}

//--------------------------------------------------------------------------------------------------
// Setting the information about how the node is shaded: solid or blend/cutout
// and if the node has micromesh data.
// This will be used by getNodes to get only the code that have the requested values
//
// Position 0: solid
// Position 1: micromesh
void ToolboxScene::setShadeNodes()
{
  const auto& prim_inst = m_toolscene->instances();
  const auto& meshes    = m_toolscene->meshes();

  m_shadeNodes.resize(prim_inst.size());  // Storing the

  for(uint32_t node_id = 0; node_id < prim_inst.size(); node_id++)
  {
    const micromesh_tool::ToolScene::PrimitiveInstance& instance   = prim_inst[node_id];
    uint32_t                                            ref_id     = instance.primMeshRef;
    const std::unique_ptr<micromesh_tool::ToolMesh>&    mesh       = meshes[ref_id];
    int32_t                                             baryIndex  = mesh->relations().bary;
    int32_t                                             groupIndex = mesh->relations().group;
    int32_t                                             mat_id     = mesh->relations().material;

    // Find if the mesh on the instance have bary info
    if(baryIndex != -1 && groupIndex != -1)
    {
      const std::vector<std::unique_ptr<DeviceBary>>& barys = m_toolsceneVk->barys();
      if(barys.size() > static_cast<size_t>(baryIndex))
      {
        const std::vector<DeviceMicromap>& micromaps = barys[baryIndex]->micromaps();
        if(micromaps.size() > static_cast<size_t>(groupIndex))
        {
          const DeviceMicromap& micromap = micromaps[groupIndex];
          if(micromap.raster())
          {
            m_shadeNodes[node_id].set(1);
          }
        }
      }
    }

    // Find if the material is set as "OPAQUE"
    tinygltf::Material* mat = nullptr;
    if(mat_id >= 0 && static_cast<size_t>(mat_id) < m_toolscene->materials().size())
      mat = &m_toolscene->materials()[mat_id];

    if(!mat || mat->alphaMode == "OPAQUE")
      m_shadeNodes[node_id].set(0);
  }
}


//--------------------------------------------------------------------------------------------------
// Creating the pipeline for the raster, for solid and transparent, and the wireframe
//
void ToolboxScene::createRasterPipeline(const ViewerSettings&                    settings,
                                        const std::vector<VkDescriptorSetLayout> extraLayouts,
                                        VkFormat                                 colorFormat,
                                        VkFormat                                 depthformat)
{
  auto scope_t = nvh::ScopedTimer("Create Raster Pipeline: ");

  m_rasterPipe.destroy(m_device);
  m_rasterPipe.plines.resize(eRasterPipelineNum);

  nvvk::Specialization specialization;
  specialization.add(0, settings.shading);      // Adding shading method to constant_id=0
  specialization.add(1, settings.debugMethod);  // Adding debug method to constant_id=1

  // Creating the Pipeline Layout
  std::vector<VkDescriptorSetLayout> layouts{getDescLayout()};
  layouts.insert(layouts.end(), extraLayouts.begin(), extraLayouts.end());

  VkShaderStageFlags stages = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_TASK_BIT_NV | VK_SHADER_STAGE_MESH_BIT_NV;
  const VkPushConstantRange  push_constant_ranges = {stages, 0, sizeof(PushConstant)};
  VkPipelineLayoutCreateInfo create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  create_info.setLayoutCount         = static_cast<uint32_t>(layouts.size());
  create_info.pSetLayouts            = layouts.data();
  create_info.pushConstantRangeCount = 1;
  create_info.pPushConstantRanges    = &push_constant_ranges;
  vkCreatePipelineLayout(m_device, &create_info, nullptr, &m_rasterPipe.layout);

  // Shader source (Spir-V)
  const std::vector<uint32_t> raster_v(std::begin(raster_vert), std::end(raster_vert));
  const std::vector<uint32_t> raster_f(std::begin(raster_frag), std::end(raster_frag));
  const std::vector<uint32_t> overlay_f(std::begin(raster_overlay_frag), std::end(raster_overlay_frag));
  const std::vector<uint32_t> shell_v(std::begin(raster_shell_vert), std::end(raster_shell_vert));
  const std::vector<uint32_t> shell_f(std::begin(raster_shell_frag), std::end(raster_shell_frag));
  const std::vector<uint32_t> vector_v(std::begin(raster_vectors_vert), std::end(raster_vectors_vert));
  const std::vector<uint32_t> vector_f(std::begin(raster_vectors_frag), std::end(raster_vectors_frag));
  const std::vector<uint32_t> raster_micromesh_t(std::begin(draw_compressed_basic_task_glsl),
                                                 std::end(draw_compressed_basic_task_glsl));
  const std::vector<uint32_t> raster_micromesh_m(std::begin(draw_compressed_basic_mesh_glsl),
                                                 std::end(draw_compressed_basic_mesh_glsl));
  const std::vector<uint32_t> raster_micromesh_f(std::begin(draw_compressed_basic_frag_glsl),
                                                 std::end(draw_compressed_basic_frag_glsl));

#ifdef USE_NSIGHT_AFTERMATH
  g_aftermathTracker->addShaderBinary(raster_v);
  g_aftermathTracker->addShaderBinary(raster_f);
  g_aftermathTracker->addShaderBinary(shell_v);
  g_aftermathTracker->addShaderBinary(shell_f);
  g_aftermathTracker->addShaderBinary(vector_v);
  g_aftermathTracker->addShaderBinary(vector_f);
  g_aftermathTracker->addShaderBinary(overlay_f);
  g_aftermathTracker->addShaderBinary(raster_micromesh_t);
  g_aftermathTracker->addShaderBinary(raster_micromesh_m);
  g_aftermathTracker->addShaderBinary(raster_micromesh_f);
#endif  // USE_NSIGHT_AFTERMATH

  //VkFormat color_format = m_gBuffers->getColorFormat(eResult);  // Using the RGBA32F

  VkPipelineRenderingCreateInfoKHR rf_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
  rf_info.colorAttachmentCount    = 1;
  rf_info.pColorAttachmentFormats = &colorFormat;
  rf_info.depthAttachmentFormat   = depthformat;  //m_gBuffers->getDepthFormat();

  // Creating the Pipeline
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_rasterPipe.layout, {});
  gpb.createInfo.pNext = &rf_info;

  {
    gpb.addBindingDescriptions({{0, sizeof(vec4)}});
    gpb.addAttributeDescriptions({{0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0}});

    // Solid
    gpb.rasterizationState.depthBiasEnable         = VK_TRUE;
    gpb.rasterizationState.depthBiasConstantFactor = -1;
    gpb.rasterizationState.depthBiasSlopeFactor    = 1;
    gpb.rasterizationState.cullMode = settings.forceDoubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
    gpb.addShader(raster_v, VK_SHADER_STAGE_VERTEX_BIT).pSpecializationInfo   = specialization.getSpecialization();
    gpb.addShader(raster_f, VK_SHADER_STAGE_FRAGMENT_BIT).pSpecializationInfo = specialization.getSpecialization();
    m_rasterPipe.plines[RasterPipelines::eRasterPipelineSolid]                = gpb.createPipeline();
    m_dutil->DBG_NAME(m_rasterPipe.plines[RasterPipelines::eRasterPipelineSolid]);

    // Blend
    gpb.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    VkPipelineColorBlendAttachmentState blend_state{};
    blend_state.blendEnable = VK_TRUE;
    blend_state.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blend_state.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend_state.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    gpb.setBlendAttachmentState(0, blend_state);
    m_rasterPipe.plines[RasterPipelines::eRasterPipelineBlend] = gpb.createPipeline();
    m_dutil->DBG_NAME(m_rasterPipe.plines[RasterPipelines::eRasterPipelineBlend]);

    // Revert Blend Mode
    blend_state.blendEnable = VK_FALSE;
    gpb.setBlendAttachmentState(0, blend_state);
  }

  // Micromesh
  {
    gpb.clearShaders();
    gpb.clearAttributeDescriptions();  // No need and avoid calling vkCmdBindVertexBuffers
    gpb.clearBindingDescriptions();
    gpb.rasterizationState.cullMode = settings.forceDoubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
    gpb.addShader(raster_micromesh_t, VK_SHADER_STAGE_TASK_BIT_NV).pSpecializationInfo = specialization.getSpecialization();
    gpb.addShader(raster_micromesh_m, VK_SHADER_STAGE_MESH_BIT_NV).pSpecializationInfo = specialization.getSpecialization();
    gpb.addShader(raster_micromesh_f, VK_SHADER_STAGE_FRAGMENT_BIT).pSpecializationInfo = specialization.getSpecialization();
    m_rasterPipe.plines[RasterPipelines::eRasterPipelineMicromeshSolid] = gpb.createPipeline();
    m_dutil->DBG_NAME(m_rasterPipe.plines[RasterPipelines::eRasterPipelineMicromeshSolid]);
  }


  // Overlays
  {
    gpb.rasterizationState.cullMode = VK_CULL_MODE_NONE;

    // Add description back, removed for micromesh
    gpb.addBindingDescriptions({{0, sizeof(vec4)}});
    gpb.addAttributeDescriptions({{0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0}});

    gpb.rasterizationState.depthBiasEnable = VK_FALSE;
    gpb.rasterizationState.polygonMode     = VK_POLYGON_MODE_LINE;
    gpb.rasterizationState.lineWidth       = 1.0F;
    gpb.depthStencilState.depthWriteEnable = VK_FALSE;

    // Wireframe
    gpb.clearShaders();
    gpb.addShader(raster_v, VK_SHADER_STAGE_VERTEX_BIT);
    gpb.addShader(overlay_f, VK_SHADER_STAGE_FRAGMENT_BIT);
    m_rasterPipe.plines[RasterPipelines::eRasterPipelineWire] = gpb.createPipeline();
    m_dutil->DBG_NAME(m_rasterPipe.plines[RasterPipelines::eRasterPipelineWire]);

    // Shell
    gpb.clearShaders();
    gpb.addShader(shell_v, VK_SHADER_STAGE_VERTEX_BIT);
    gpb.addShader(shell_f, VK_SHADER_STAGE_FRAGMENT_BIT);
    m_rasterPipe.plines[RasterPipelines::eRasterPipelineShell] = gpb.createPipeline();
    m_dutil->DBG_NAME(m_rasterPipe.plines[RasterPipelines::eRasterPipelineShell]);

    // Vector
    gpb.clearShaders();
    gpb.addShader(vector_v, VK_SHADER_STAGE_VERTEX_BIT).pSpecializationInfo = specialization.getSpecialization();
    gpb.addShader(vector_f, VK_SHADER_STAGE_FRAGMENT_BIT);
    m_rasterPipe.plines[RasterPipelines::eRasterPipelineVector] = gpb.createPipeline();
    m_dutil->DBG_NAME(m_rasterPipe.plines[RasterPipelines::eRasterPipelineVector]);

    // Micromesh-wireframe
    gpb.clearShaders();
    gpb.clearAttributeDescriptions();  // No need and avoid calling vkCmdBindVertexBuffers
    gpb.clearBindingDescriptions();
    gpb.addShader(raster_micromesh_t, VK_SHADER_STAGE_TASK_BIT_NV).pSpecializationInfo = specialization.getSpecialization();
    gpb.addShader(raster_micromesh_m, VK_SHADER_STAGE_MESH_BIT_NV).pSpecializationInfo = specialization.getSpecialization();
    gpb.addShader(overlay_f, VK_SHADER_STAGE_FRAGMENT_BIT);
    m_rasterPipe.plines[RasterPipelines::eRasterPipelineMicromeshWire] = gpb.createPipeline();
    m_dutil->DBG_NAME(m_rasterPipe.plines[RasterPipelines::eRasterPipelineMicromeshWire]);
  }

  resetDirty(SceneDirtyFlags::eRasterPipeline);
  setDirty(SceneDirtyFlags::eRasterRecord);  // Recording will need to be redone
}

//--------------------------------------------------------------------------------------------------
// Creating the pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void ToolboxScene::createRtxPipeline(const std::vector<VkDescriptorSetLayout> extraLayouts)
{
  auto scope_t = nvh::ScopedTimer("createRtxPipeline\n");

  m_rtxPipe.destroy(m_device);
  m_rtxPipe.plines.resize(1);

  // Creating all shaders
  enum StageIndices
  {
    eRaygen,
    eMiss,
    eClosestHit,
    eAnyHit,
    eShaderGroupCount
  };

  // All SPIR-V shaders
  const std::vector<uint32_t> rgen(std::begin(pathtrace_rgen), std::end(pathtrace_rgen));
  const std::vector<uint32_t> rmiss(std::begin(pathtrace_rmiss), std::end(pathtrace_rmiss));
  const std::vector<uint32_t> rchit(std::begin(pathtrace_rchit), std::end(pathtrace_rchit));
  const std::vector<uint32_t> rahit(std::begin(pathtrace_rahit), std::end(pathtrace_rahit));

#ifdef USE_NSIGHT_AFTERMATH
  g_aftermathTracker->addShaderBinary(rgen);
  g_aftermathTracker->addShaderBinary(rmiss);
  g_aftermathTracker->addShaderBinary(rchit);
  g_aftermathTracker->addShaderBinary(rahit);
#endif  // USE_NSIGHT_AFTERMATH

  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.pName = "main";  // All the same entry point
  // Raygen
  stage.module    = nvvk::createShaderModule(m_device, rgen);
  stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stages[eRaygen] = stage;
  m_dutil->setObjectName(stage.module, "Raygen");
  // Miss
  stage.module  = nvvk::createShaderModule(m_device, rmiss);
  stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss] = stage;
  m_dutil->setObjectName(stage.module, "Miss");
  // Hit Group - Closest Hit
  stage.module        = nvvk::createShaderModule(m_device, rchit);
  stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stages[eClosestHit] = stage;
  m_dutil->setObjectName(stage.module, "Closest Hit");
  // AnyHit
  stage.module    = nvvk::createShaderModule(m_device, rahit);
  stage.stage     = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
  stages[eAnyHit] = stage;
  m_dutil->setObjectName(stage.module, "Any Hit");

  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = VK_SHADER_UNUSED_KHR;
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;
  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  shader_groups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  shader_groups.push_back(group);

  // Hit Group-0
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  group.anyHitShader     = eAnyHit;
  shader_groups.push_back(group);

  // Push constant: we want to be able to update constants used by the shaders
  const VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant)};

  VkPipelineLayoutCreateInfo pipeline_layout_create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipeline_layout_create_info.pushConstantRangeCount = 1;
  pipeline_layout_create_info.pPushConstantRanges    = &push_constant;

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<VkDescriptorSetLayout> rt_desc_set_layouts = {getRtxDescLayout(), getDescLayout()};
  rt_desc_set_layouts.insert(rt_desc_set_layouts.end(), extraLayouts.begin(), extraLayouts.end());

  pipeline_layout_create_info.setLayoutCount = static_cast<uint32_t>(rt_desc_set_layouts.size());
  pipeline_layout_create_info.pSetLayouts    = rt_desc_set_layouts.data();
  vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &m_rtxPipe.layout);
  m_dutil->DBG_NAME(m_rtxPipe.layout);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  ray_pipeline_info.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
  ray_pipeline_info.pStages                      = stages.data();
  ray_pipeline_info.groupCount                   = static_cast<uint32_t>(shader_groups.size());
  ray_pipeline_info.pGroups                      = shader_groups.data();
  ray_pipeline_info.maxPipelineRayRecursionDepth = 2;  // Ray depth
  ray_pipeline_info.layout                       = m_rtxPipe.layout;
  ray_pipeline_info.flags = m_toolsceneVk->hasRtxMicromesh() ? VK_PIPELINE_CREATE_RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV : 0;
  vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, (m_rtxPipe.plines).data());
  m_dutil->DBG_NAME(m_rtxPipe.plines[0]);

  // Creating the SBT
  createSbt(m_rtxPipe.plines[0], ray_pipeline_info);

  // Removing temp modules
  for(VkPipelineShaderStageCreateInfo& s : stages)
  {
    vkDestroyShaderModule(m_device, s.module, nullptr);
  }

  resetDirty(SceneDirtyFlags::eRtxPipeline);
}
