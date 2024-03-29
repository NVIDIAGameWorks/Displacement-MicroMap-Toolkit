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

//////////////////////////////////////////////////////////////////////////
/*

  This sample raytraces a plane made of 6x6 triangles with Micro-Mesh displacement
  - The scene is created in createScene()
  - Micro-mesh creation uses the MicromapProcess class
  - Vulkan buffers holding the scene are created in createVkBuffers()
  - Bottom and Top level acceleration structures are using the Vulkan buffers 
    and scene description in createBottomLevelAS() and createTopLevelAS()
  - The raytracing pipeline, composed of RayGen, Miss, ClosestHit shaders
    and the creation of the shader binding table, is done increateRtxPipeline()
  - Rendering is done in onRender()

  Note: search for #MICROMESH for specific changes for Micro-Mesh

*/
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <glm/detail/type_half.hpp>  // for half float

#include <vulkan/vulkan_core.h>
#include "vulkan_nv/vk_nv_micromesh.h"

#define VMA_IMPLEMENTATION
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "nvmath/nvmath.h"
#include "nvh/primitives.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_testing.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"

#include "shaders/dh_bindings.h"
#include "shaders/device_host.h"
#include "nvvkhl/shaders/dh_sky.h"

#include "_autogen/pathtrace.rchit.h"
#include "_autogen/pathtrace.rgen.h"
#include "_autogen/pathtrace.rmiss.h"
#include "vulkan_nv/vk_nv_micromesh_prototypes.h"

#include "dmm_process.hpp"
#include "nesting_scoped_timer.hpp"


namespace {

using namespace nvvkhl;

//////////////////////////////////////////////////////////////////////////
/// </summary> Ray trace multiple primitives
class Raytracing : public nvvkhl::IAppElement
{
  struct Settings
  {
    float intensity{5.0F};
    float metallic{0.5F};
    float roughness{1.0F};
    int   maxDepth{5};
    // #MICROMESH
    bool          enableDisplacement{true};
    int           subdivlevel{3};
    nvmath::vec2f dispBiasScale{-0.3F, 1.0F};
    Terrain       terrain{};
    bool          showWireframe{true};
  } m_settings;


public:
  Raytracing()           = default;
  ~Raytracing() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    m_dutil    = std::make_unique<nvvk::DebugUtil>(m_device);            // Debug utility
    m_alloc    = std::make_unique<AllocVma>(m_app->getContext().get());  // Allocator
    m_rtSet    = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_micromap = std::make_unique<MicromapProcess>(m_app->getContext().get(), m_alloc.get());

    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create BLAS/TLAS and the Shader Binding Table (SBT)
    int32_t gct_queue_index = m_app->getContext()->m_queueGCT.familyIndex;
    m_rtBuilder.setup(m_device, m_alloc.get(), gct_queue_index);
    m_sbt.setup(m_device, gct_queue_index, m_alloc.get(), m_rtProperties);

    // Create resources
    createScene();
    createVkBuffers();
    // #MICROMESH
    {
      NestingScopedTimer stimer("Create Micromesh");
      VkCommandBuffer    cmd = m_app->createTempCmdBuffer();
      m_micromap->createMicromapData(cmd, m_meshes[0], static_cast<uint16_t>(m_settings.subdivlevel), m_settings.terrain);
      m_micromap->createMicromapBuffers(cmd, m_meshes[0], m_settings.dispBiasScale);
      m_app->submitAndWaitTempCmdBuffer(cmd);
      m_micromap->cleanBuildData();
    }
    createBottomLevelAS();
    createTopLevelAS();
    createRtxPipeline();
    createGbuffers(m_viewSize);
  }

  void onDetach() override { destroyResources(); }

  void onResize(uint32_t width, uint32_t height) override
  {
    createGbuffers({width, height});
    writeRtDesc();
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");
      ImGuiH::CameraWidget();

      using namespace ImGuiH;

      // #MICROMESH - begin
      ImGui::Text("Micro-Mesh");
      PropertyEditor::begin();
      if(PropertyEditor::entry("Enable", [&] { return ImGui::Checkbox("##ll", &m_settings.enableDisplacement); }))
      {
        vkDeviceWaitIdle(m_device);
        m_rtBuilder.destroy();
        createBottomLevelAS();
        createTopLevelAS();
        writeRtDesc();
      }

      bool level_changed{false};
      bool bias_scale_changed{false};
      level_changed |= PropertyEditor::entry("Subdivision Level",
                                             [&] { return ImGui::SliderInt("#1", &m_settings.subdivlevel, 0, 5); });
      bias_scale_changed |= PropertyEditor::entry("Displacement Bias", [&] {
        return ImGui::SliderFloat("#1", &m_settings.dispBiasScale.x, -1.0F, 1.0F);
      });
      bias_scale_changed |= PropertyEditor::entry("Displacement Scale", [&] {
        return ImGui::SliderFloat("#1", &m_settings.dispBiasScale.y, 0.0F, 2.0F);
      });
      if(PropertyEditor::treeNode("Terrain"))
      {
        PropertyEditor::entry("Show Wireframe", [&] { return ImGui::Checkbox("##ll", &m_settings.showWireframe); });

        level_changed |=
            PropertyEditor::entry("Seed", [&] { return ImGui::SliderFloat("#1", &m_settings.terrain.seed, -1.0F, 1.0F); });
        level_changed |= PropertyEditor::entry("Frequency", [&] {
          return ImGui::SliderFloat("#1", &m_settings.terrain.freq, 0.01F, 4.0F);
        });
        level_changed |= PropertyEditor::entry("Power", [&] {
          return ImGui::SliderFloat("#1", &m_settings.terrain.power, 1.1F, 4.0F);
        });
        level_changed |=
            PropertyEditor::entry("Octave", [&] { return ImGui::SliderInt("#1", &m_settings.terrain.octave, 1, 8); });
        PropertyEditor::treePop();
      }

      if(level_changed || bias_scale_changed)
      {
        NestingScopedTimer stimer("Create Micromesh");
        vkDeviceWaitIdle(m_device);
        auto* cmd = m_app->createTempCmdBuffer();
        if(level_changed)
        {  // Recreate all values
          m_micromap->createMicromapData(cmd, m_meshes[0], static_cast<uint16_t>(m_settings.subdivlevel), m_settings.terrain);
        }

        if(bias_scale_changed)
        {  // Recreate the buffers attached to the BLAS
          m_micromap->createMicromapBuffers(cmd, m_meshes[0], m_settings.dispBiasScale);
        }
        m_app->submitAndWaitTempCmdBuffer(cmd);
        m_micromap->cleanBuildData();

        // Recreate the acceleration structure
        m_rtBuilder.destroy();
        createBottomLevelAS();
        createTopLevelAS();
        writeRtDesc();
      }
      // #MICROMESH - end

      PropertyEditor::end();
      ImGui::Text("Material");
      PropertyEditor::begin();
      PropertyEditor::entry("Metallic", [&] { return ImGui::SliderFloat("#1", &m_settings.metallic, 0.0F, 1.0F); });
      PropertyEditor::entry("Roughness", [&] { return ImGui::SliderFloat("#1", &m_settings.roughness, 0.0F, 1.0F); });
      PropertyEditor::entry("Intensity", [&] { return ImGui::SliderFloat("#1", &m_settings.intensity, 0.0F, 10.0F); });
      PropertyEditor::end();
      ImGui::Separator();
      ImGui::Text("Sun Orientation");
      PropertyEditor::begin();
      glm::vec3 dir = m_skyParams.directionToLight;
      ImGuiH::azimuthElevationSliders(dir, false);
      m_skyParams.directionToLight = dir;
      PropertyEditor::end();
      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image(m_gBuffer->getDescriptorSet(), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }


  void onRender(VkCommandBuffer cmd) override
  {
    auto sdbg = m_dutil->DBG_SCOPE(cmd);

    float     view_aspect_ratio = m_viewSize.x / m_viewSize.y;
    glm::vec3 eye;
    glm::vec3 center;
    glm::vec3 up;
    CameraManip.getLookat(eye, center, up);

    // Update the uniform buffer containing frame info
    shaders::FrameInfo finfo{};
    const auto&        clip = CameraManip.getClipPlanes();
    finfo.view              = CameraManip.getMatrix();
    finfo.proj              = nvmath::perspectiveVK(CameraManip.getFov(), view_aspect_ratio, clip.x, clip.y);
    finfo.projInv           = nvmath::inverse(finfo.proj);
    finfo.viewInv           = nvmath::inverse(finfo.view);
    finfo.camPos            = eye;
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(shaders::FrameInfo), &finfo);

    // Update the sky
    vkCmdUpdateBuffer(cmd, m_bSkyParams.buffer, 0, sizeof(nvvkhl_shaders::ProceduralSkyShaderParameters), &m_skyParams);

    // Ray trace
    std::vector<VkDescriptorSet> desc_sets{m_rtSet->getSet()};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.plines[0]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.layout, 0,
                            static_cast<uint32_t>(desc_sets.size()), desc_sets.data(), 0, nullptr);

    m_pushConst.intensity = m_settings.intensity;
    m_pushConst.metallic  = m_settings.metallic;
    m_pushConst.roughness = m_settings.roughness;
    m_pushConst.maxDepth  = m_settings.maxDepth;
    m_pushConst.numBaseTriangles =
        m_settings.showWireframe ? static_cast<int>(pow(2.0F, static_cast<float>(m_settings.subdivlevel))) : 0;
    vkCmdPushConstants(cmd, m_rtPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(shaders::PushConstant), &m_pushConst);

    const auto& regions = m_sbt.getRegions();
    const auto& size    = m_app->getViewportSize();
    vkCmdTraceRaysKHR(cmd, regions.data(), &regions[1], &regions[2], &regions[3], size.width, size.height, 1);
  }

private:
  void createScene()
  {
    // Adding a plane & material
    m_materials.push_back({nvmath::vec4f(.7F, .7F, .7F, 1.0F)});
    m_meshes.emplace_back(nvh::createPlane(3.0F, 1.0F, 1.0F));
    auto& n       = m_nodes.emplace_back();
    n.mesh        = static_cast<int>(m_meshes.size()) - 1;
    n.material    = static_cast<int>(m_materials.size()) - 1;
    n.translation = {0.0F, 0.0F, 0.0F};

    // Setting camera to see the scene
    CameraManip.setClipPlanes({0.01F, 100.0F});
    CameraManip.setLookat({0.96777, 1.33764, 1.31298}, {-0.08092, 0.20461, -0.14889}, {0.00000, 1.00000, 0.00000});

    // Default Sky values
    m_skyParams = nvvkhl_shaders::initSkyShaderParameters();
  }


  void createGbuffers(const nvmath::vec2f& size)
  {
    vkDeviceWaitIdle(m_device);

    // Rendering image targets
    m_viewSize = size;
    m_gBuffer  = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(),
                                                  VkExtent2D{static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)},
                                                  m_colorFormat, m_depthFormat);
  }

  // Create all Vulkan buffer data
  void createVkBuffers()
  {
    auto* cmd = m_app->createTempCmdBuffer();
    m_bMeshes.resize(m_meshes.size());

    auto rt_usage_flag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                         | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    // Create a buffer of Vertex and Index per mesh
    std::vector<shaders::PrimMeshInfo> prim_info;
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      auto& m    = m_bMeshes[i];
      m.vertices = m_alloc->createBuffer(cmd, m_meshes[i].vertices, rt_usage_flag);
      m.indices  = m_alloc->createBuffer(cmd, m_meshes[i].triangles, rt_usage_flag);
      m_dutil->DBG_NAME_IDX(m.vertices.buffer, i);
      m_dutil->DBG_NAME_IDX(m.indices.buffer, i);

      // To find the buffers of the mesh (buffer reference)
      shaders::PrimMeshInfo info{};
      info.vertexAddress = nvvk::getBufferDeviceAddress(m_device, m.vertices.buffer);
      info.indexAddress  = nvvk::getBufferDeviceAddress(m_device, m.indices.buffer);
      prim_info.emplace_back(info);
    }

    // Creating the buffer of all primitive information
    m_bPrimInfo = m_alloc->createBuffer(cmd, prim_info, rt_usage_flag);
    m_dutil->DBG_NAME(m_bPrimInfo.buffer);

    // Create the buffer of the current frame, changing at each frame
    m_bFrameInfo = m_alloc->createBuffer(sizeof(shaders::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bFrameInfo.buffer);

    // Create the buffer of sky parameters, updated at each frame
    m_bSkyParams = m_alloc->createBuffer(sizeof(nvvkhl_shaders::ProceduralSkyShaderParameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bSkyParams.buffer);

    // Primitive instance information
    std::vector<shaders::InstanceInfo> inst_info;
    for(auto& node : m_nodes)
    {
      shaders::InstanceInfo info{};
      info.transform  = node.localMatrix();
      info.materialID = node.material;
      inst_info.emplace_back(info);
    }
    m_bInstInfoBuffer =
        m_alloc->createBuffer(cmd, inst_info, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bInstInfoBuffer.buffer);

    m_bMaterials = m_alloc->createBuffer(cmd, m_materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bMaterials.buffer);

    // Buffer references of all scene elements
    shaders::SceneDescription scene_desc{};
    scene_desc.materialAddress = nvvk::getBufferDeviceAddress(m_device, m_bMaterials.buffer);
    scene_desc.primInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_bPrimInfo.buffer);
    scene_desc.instInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_bInstInfoBuffer.buffer);
    m_bSceneDesc               = m_alloc->createBuffer(cmd, sizeof(shaders::SceneDescription), &scene_desc,
                                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bSceneDesc.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }


  //--------------------------------------------------------------------------------------------------
  // Converting a PrimitiveMesh as input for BLAS
  //
  static nvvk::RaytracingBuilderKHR::BlasInput primitiveToGeometry(const nvh::PrimitiveMesh& prim,
                                                                   VkDeviceAddress           vertexAddress,
                                                                   VkDeviceAddress           indexAddress)
  {
    auto max_primitive_count = static_cast<uint32_t>(prim.triangles.size());

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triangles.vertexFormat             = VK_FORMAT_R32G32B32A32_SFLOAT;  // vec3 vertex position data.
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride             = sizeof(nvh::PrimitiveVertex);
    triangles.indexType                = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress  = indexAddress;
    triangles.maxVertex                = static_cast<uint32_t>(prim.vertices.size()) - 1;
    //triangles.transformData; // Identity

    // Identify the above data as containing opaque triangles.
    VkAccelerationStructureGeometryKHR as_geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    as_geom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    as_geom.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR | VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    as_geom.geometry.triangles = triangles;

    VkAccelerationStructureBuildRangeInfoKHR offset{};
    offset.firstVertex     = 0;
    offset.primitiveCount  = max_primitive_count;
    offset.primitiveOffset = 0;
    offset.transformOffset = 0;

    // Our BLAS is made from only one geometry, but could be made of many geometries
    nvvk::RaytracingBuilderKHR::BlasInput input;
    input.asGeometry.emplace_back(as_geom);
    input.asBuildOffsetInfo.emplace_back(offset);

    return input;
  }

  //--------------------------------------------------------------------------------------------------
  // Create all bottom level acceleration structures (BLAS)
  //
  void createBottomLevelAS()
  {
    NestingScopedTimer stimer("Create BLAS");
    // BLAS - Storing each primitive in a geometry
    std::vector<nvvk::RaytracingBuilderKHR::BlasInput> all_blas;
    all_blas.reserve(m_meshes.size());

    // #MICROMESH
    assert(m_meshes.size() == 1);  // The micromap is created for only one mesh
    std::vector<VkAccelerationStructureTrianglesDisplacementMicromapNV> geometry_displacements;  // hold data until BLAS is created
    geometry_displacements.reserve(m_meshes.size());

    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      auto vertex_address = nvvk::getBufferDeviceAddress(m_device, m_bMeshes[p_idx].vertices.buffer);
      auto index_address  = nvvk::getBufferDeviceAddress(m_device, m_bMeshes[p_idx].indices.buffer);
      auto geo            = primitiveToGeometry(m_meshes[p_idx], vertex_address, index_address);

      // #MICROMESH
      VkAccelerationStructureTrianglesDisplacementMicromapNV displacement{
          VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_DISPLACEMENT_MICROMAP_NV};
      if(m_settings.enableDisplacement)
      {
        // micromap for this mesh
        auto primitive_flags_addr = nvvk::getBufferDeviceAddress(m_device, m_micromap->primitiveFlags().buffer);
        auto directions_addr      = nvvk::getBufferDeviceAddress(m_device, m_micromap->displacementDirections().buffer);
        auto direction_bounds_addr = nvvk::getBufferDeviceAddress(m_device, m_micromap->displacementBiasAndScale().buffer);

        displacement.micromap                  = m_micromap->micromap();
        displacement.usageCountsCount          = static_cast<uint32_t>(m_micromap->usages().size());
        displacement.pUsageCounts              = m_micromap->usages().data();
        displacement.baseTriangle              = 0U;                    // default
        displacement.indexBuffer.deviceAddress = 0ULL;                  // default
        displacement.indexStride               = 0;                     // default
        displacement.indexType                 = VK_INDEX_TYPE_UINT32;  // default

        assert(directions_addr);
        {
          displacement.displacementVectorBuffer.deviceAddress = directions_addr;
          displacement.displacementVectorStride               = sizeof(glm::detail::hdata) * 4;
          displacement.displacementVectorFormat               = VK_FORMAT_R16G16B16A16_SFLOAT;
        }

        if(direction_bounds_addr != 0u)  // optional
        {
          displacement.displacementBiasAndScaleBuffer.deviceAddress = direction_bounds_addr;
          displacement.displacementBiasAndScaleStride               = sizeof(nvmath::vec2f);
          displacement.displacementBiasAndScaleFormat               = VK_FORMAT_R32G32_SFLOAT;
        }

        if(primitive_flags_addr != 0u)  // optional
        {
          displacement.displacedMicromapPrimitiveFlags.deviceAddress = primitive_flags_addr;
          displacement.displacedMicromapPrimitiveFlagsStride         = sizeof(uint8_t);
        }

        // Adding micromap
        geometry_displacements.emplace_back(displacement);
        geo.asGeometry[0].geometry.triangles.pNext = &geometry_displacements.back();
      }


      all_blas.push_back({geo});
    }

    VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    m_rtBuilder.buildBlas(all_blas, flags);
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    NestingScopedTimer stimer("Create TLAS");

    std::vector<VkAccelerationStructureInstanceKHR> tlas;
    tlas.reserve(m_nodes.size());
    for(auto& node : m_nodes)
    {
      VkGeometryInstanceFlagsKHR flags{VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV};
      flags |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;

      VkAccelerationStructureInstanceKHR ray_inst{};
      ray_inst.transform           = nvvk::toTransformMatrixKHR(node.localMatrix());  // Position of the instance
      ray_inst.instanceCustomIndex = node.mesh & 0x00FFFFFF;                          // gl_InstanceCustomIndexEXT
      ray_inst.accelerationStructureReference         = m_rtBuilder.getBlasDeviceAddress(node.mesh);
      ray_inst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
      ray_inst.flags                                  = flags & 0xFFU;
      ray_inst.mask                                   = 0xFF;
      tlas.emplace_back(ray_inst);
    }

    // #MICROMESH
    VkBuildAccelerationStructureFlagsKHR build_flags =
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;

    m_rtBuilder.buildTlas(tlas, build_flags);
  }


  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    auto& p = m_rtPipe;
    auto& d = m_rtSet;
    p.plines.resize(1);

    // This descriptor set, holds the top level acceleration structure and the output image
    // Create Binding Set
    d->addBinding(BRtTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(BRtOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(BRtFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(BRtSceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    d->addBinding(BRtSkyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    d->initLayout();
    d->initPool(1);

    m_dutil->DBG_NAME(d->getLayout());
    m_dutil->DBG_NAME(d->getSet(0));

    // Creating all shaders
    enum StageIndices
    {
      eRaygen,
      eMiss,
      eClosestHit,
      eShaderGroupCount
    };
    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.pName = "main";  // All the same entry point
    // Raygen
    stage.module    = nvvk::createShaderModule(m_device, pathtrace_rgen, sizeof(pathtrace_rgen));
    stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eRaygen] = stage;
    m_dutil->setObjectName(stage.module, "Raygen");
    // Miss
    stage.module  = nvvk::createShaderModule(m_device, pathtrace_rmiss, sizeof(pathtrace_rmiss));
    stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eMiss] = stage;
    m_dutil->setObjectName(stage.module, "Miss");
    // Hit Group - Closest Hit
    stage.module        = nvvk::createShaderModule(m_device, pathtrace_rchit, sizeof(pathtrace_rchit));
    stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eClosestHit] = stage;
    m_dutil->setObjectName(stage.module, "Closest Hit");


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

    // Closest hit shader
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    shader_groups.push_back(group);

    // Push constant: we want to be able to update constants used by the shaders
    VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(shaders::PushConstant)};

    VkPipelineLayoutCreateInfo pipeline_layout_create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_create_info.pushConstantRangeCount = 1;
    pipeline_layout_create_info.pPushConstantRanges    = &push_constant;


    // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
    std::vector<VkDescriptorSetLayout> rt_desc_set_layouts = {d->getLayout()};  // , m_pContainer[eGraphic].dstLayout};
    pipeline_layout_create_info.setLayoutCount             = static_cast<uint32_t>(rt_desc_set_layouts.size());
    pipeline_layout_create_info.pSetLayouts                = rt_desc_set_layouts.data();
    NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &p.layout));
    m_dutil->DBG_NAME(p.layout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    ray_pipeline_info.flags      = VK_PIPELINE_CREATE_RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV;  // #MICROMESH
    ray_pipeline_info.stageCount = static_cast<uint32_t>(stages.size());                         // Stages are shaders
    ray_pipeline_info.pStages    = stages.data();
    ray_pipeline_info.groupCount = static_cast<uint32_t>(shader_groups.size());
    ray_pipeline_info.pGroups    = shader_groups.data();
    ray_pipeline_info.maxPipelineRayRecursionDepth = 10;  // Ray depth
    ray_pipeline_info.layout                       = p.layout;
    NVVK_CHECK(vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, (p.plines).data()));
    m_dutil->DBG_NAME(p.plines[0]);

    // Creating the SBT
    m_sbt.create(p.plines[0], ray_pipeline_info);

    // Removing temp modules
    for(auto& s : stages)
      vkDestroyShaderModule(m_device, s.module, nullptr);
  }

  void writeRtDesc()
  {
    auto& d = m_rtSet;

    // Write to descriptors
    VkAccelerationStructureKHR tlas = m_rtBuilder.getAccelerationStructure();
    VkWriteDescriptorSetAccelerationStructureKHR desc_as_info{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    desc_as_info.accelerationStructureCount = 1;
    desc_as_info.pAccelerationStructures    = &tlas;
    VkDescriptorImageInfo  image_info{{}, m_gBuffer->getColorImageView(), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo dbi_sky{m_bSkyParams.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo scene_desc{m_bSceneDesc.buffer, 0, VK_WHOLE_SIZE};

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(d->makeWrite(0, BRtTlas, &desc_as_info));
    writes.emplace_back(d->makeWrite(0, BRtOutImage, &image_info));
    writes.emplace_back(d->makeWrite(0, BRtFrameInfo, &dbi_unif));
    writes.emplace_back(d->makeWrite(0, BRtSceneDesc, &scene_desc));
    writes.emplace_back(d->makeWrite(0, BRtSkyParam, &dbi_sky));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void destroyResources()
  {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    for(auto& m : m_bMeshes)
    {
      m_alloc->destroy(m.vertices);
      m_alloc->destroy(m.indices);
    }
    m_alloc->destroy(m_bFrameInfo);
    m_alloc->destroy(m_bPrimInfo);
    m_alloc->destroy(m_bSceneDesc);
    m_alloc->destroy(m_bInstInfoBuffer);
    m_alloc->destroy(m_bMaterials);
    m_alloc->destroy(m_bSkyParams);

    m_rtSet->deinit();
    m_gBuffer.reset();

    m_rtPipe.destroy(m_device);

    m_sbt.destroy();
    m_rtBuilder.destroy();
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*                          m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>              m_dutil;
  std::unique_ptr<AllocVma>                     m_alloc;
  std::unique_ptr<nvvk::DescriptorSetContainer> m_rtSet;  // Descriptor set
  std::unique_ptr<MicromapProcess>              m_micromap;


  nvmath::vec2f                    m_viewSize    = {1, 1};
  VkFormat                         m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkClearColorValue                m_clearColor  = {{0.3F, 0.3F, 0.3F, 1.0F}};     // Clear color
  VkDevice                         m_device      = VK_NULL_HANDLE;                 // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffer;                                      // G-Buffers: color + depth
  nvvkhl_shaders::ProceduralSkyShaderParameters m_skyParams{};

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;  // Buffer of the vertices
    nvvk::Buffer indices;   // Buffer of the indices
  };
  std::vector<PrimitiveMeshVk> m_bMeshes;
  nvvk::Buffer                 m_bFrameInfo;
  nvvk::Buffer                 m_bPrimInfo;
  nvvk::Buffer                 m_bSceneDesc;  // SceneDescription
  nvvk::Buffer                 m_bInstInfoBuffer;
  nvvk::Buffer                 m_bMaterials;
  nvvk::Buffer                 m_bSkyParams;

  std::vector<VkSampler> m_samplers;

  // Data and setting
  struct Material
  {
    nvmath::vec4f color{1.F};
  };
  std::vector<nvh::PrimitiveMesh> m_meshes;
  std::vector<nvh::Node>          m_nodes;
  std::vector<Material>           m_materials;

  // Pipeline
  shaders::PushConstant m_pushConst{};                        // Information sent to the shader
  VkPipelineLayout      m_pipelineLayout   = VK_NULL_HANDLE;  // The description of the pipeline
  VkPipeline            m_graphicsPipeline = VK_NULL_HANDLE;  // The graphic pipeline to render
  int                   m_frame{0};

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::SBTWrapper           m_sbt;  // Shader binding table wrapper
  nvvk::RaytracingBuilderKHR m_rtBuilder;
  PipelineContainer          m_rtPipe;
};

}  // namespace
//////////////////////////////////////////////////////////////////////////
///
///
///
auto main(int argc, char** argv) -> int
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name             = PROJECT_NAME " Example";
  spec.vSync            = false;
  spec.vkSetup          = nvvk::ContextCreateInfo(false);  // #MICROMESH cannot have validation layers (crash)
  spec.vkSetup.apiMajor = 1;
  spec.vkSetup.apiMinor = 3;

  spec.vkSetup.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accel_feature);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rt_pipeline_feature);  // To use vkCmdTraceRaysKHR
  spec.vkSetup.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
  spec.vkSetup.addDeviceExtension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);

  // #MICROMESH
  static VkPhysicalDeviceOpacityMicromapFeaturesEXT mm_opacity_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT};
  static VkPhysicalDeviceDisplacementMicromapFeaturesNV mm_displacement_features = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_FEATURES_NV};
  spec.vkSetup.addDeviceExtension(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME, true, &mm_opacity_features);
  spec.vkSetup.addDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME, true, &mm_displacement_features);
  // Disable error messages introduced by micromesh
  spec.ignoreDbgMessages.push_back(0x901f59ec);  // Unknown extension
  spec.ignoreDbgMessages.push_back(0xdd73dbcf);  // Unknown structure
  spec.ignoreDbgMessages.push_back(0xba164058);  // Unknown flag  vkGetAccelerationStructureBuildSizesKHR:
  spec.ignoreDbgMessages.push_back(0x22d5bbdc);  // Unknown flag  vkCreateRayTracingPipelinesKHR
  spec.ignoreDbgMessages.push_back(0x27112e51);  // Unknown flag  vkCreateBuffer
  spec.ignoreDbgMessages.push_back(0x79de34d4);  // Unknown VK_NV_displacement_micromesh, VK_NV_opacity_micromesh

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // #MICROMESH
  if(!app->getContext()->hasDeviceExtension(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME))
  {
    LOGE("ERROR: Micro-Mesh not supported");
    exit(1);
  }

  if(!app->getContext()->hasDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME))
  {
    LOGE("ERROR: Micro-Mesh displacement not supported");
    exit(1);
  }

  load_VK_EXT_opacity_micromap_prototypes(app->getDevice(), vkGetDeviceProcAddr);

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementTesting>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());         // Menu / Quit
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>());  // Window title info
  app->addElement(std::make_shared<Raytracing>());


  app->run();

  vkDeviceWaitIdle(app->getContext()->m_device);
  app.reset();

  return test->errorCode();
}
