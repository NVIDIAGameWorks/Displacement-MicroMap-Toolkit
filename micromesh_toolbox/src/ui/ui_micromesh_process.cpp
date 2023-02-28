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

//#define DONTUSEMT // to avoid using multithreading

#include <thread>

#include "ui_micromesh_process.hpp"
#include "nvh/timesampler.hpp"
#include "toolbox_viewer.hpp"
#include "ui_micromesh_tools.hpp"


namespace fs = std::filesystem;
using PE     = ImGuiH::PropertyEditor;


// Global subdivision level slider, shared by all tools
void globalSubdivLevel(ViewerSettings&                         settings,
                       tool_bake::ToolBakeArgs&                bake_args,
                       tool_remesh::ToolRemeshArgs&            remesh_args,
                       tool_tessellate::ToolPreTessellateArgs& pretess_args)
{
  PE::begin();
  PE::entry("Bake Subdiv Level",
            [&] { return ImGui::SliderInt("##maxSubdivLevel", (int*)&settings.tools.subdivLevel, 0, 5); });
  PE::end();

  // Update all values driven by the global bake subdivision level
  bake_args.level              = settings.tools.subdivLevel;
  remesh_args.maxSubdivLevel   = static_cast<uint32_t>(settings.tools.subdivLevel);
  pretess_args.subdivLevelBias = settings.tools.pretessellateBias - settings.tools.subdivLevel;
  if(settings.tools.decimateRateFromSubdivLevel == 1)
  {
    remesh_args.decimationRatio = 1.f / (static_cast<float>(1 << (2 * settings.tools.subdivLevel)));
  }
}

bool UiMicromeshProcessPipeline::onUI()
{
  static bool       tool_running = false;
  static bool       tool_error   = false;
  static bool       use_pretess  = false;
  static bool       use_remesher = false;
  static bool       use_baker    = false;
  static bool       use_displace = false;
  static std::mutex toolMutex;

  static tool_remesh::ToolRemeshArgs                  remesh_args;
  static tool_bake::ToolBakeArgs                      bake_args{};
  static tool_tessellate::ToolPreTessellateArgs       pretess_args{};
  static tool_tessellate::ToolDisplacedTessellateArgs displace_args{};

  static ImVec4 colActive   = ImVec4(0.1F, 0.8F, 0.1F, 1.0F);
  static ImVec4 colInactive = ImVec4(0.8F, 0.1F, 0.1F, 1.0F);

  // Setting menu
  if(ImGui::Begin("Micromesh Pipeline"))
  {
    // Viewer data access
    ViewerSettings&                               settings    = m_toolboxViewer->m_settings;
    std::unique_ptr<micromesh_tool::ToolContext>& toolContext = m_toolboxViewer->m_toolContext;
    VkDevice                                      device      = m_toolboxViewer->m_device;
    GLFWwindow*                                   win_handel  = m_toolboxViewer->m_app->getWindowHandle();
    std::unique_ptr<ToolboxScene>&                scene_ref   = m_toolboxViewer->m_scenes[SceneVersion::eReference];
    std::unique_ptr<ToolboxScene>&                scene_base  = m_toolboxViewer->m_scenes[SceneVersion::eBase];


    ImVec2 arrow_size(ImGui::GetFrameHeight() * 0.5F, ImGui::GetFrameHeight() * 0.5F);

    // Validation
    if(!scene_ref->valid())
    {
      use_pretess  = false;
      use_remesher = false;
      use_displace = false;
    }
    if(!scene_base->valid() && !use_pretess && !use_remesher)
    {
      use_baker = false;
    }

    // ----- REFERENCE -----
    loadLine("Reference", ViewerSettings::eReference);

    globalSubdivLevel(settings, bake_args, remesh_args, pretess_args);


    // If the reference has bary, nothing can be done
    ImGui::BeginDisabled(scene_ref->hasBary());

    // ----- PRE_TESSELLATOR -----
    if(toolHeader("Pre-tessellator", use_pretess))
    {
      ImGui::BeginDisabled(!use_pretess);
      uiPretesselator(pretess_args, settings.tools, win_handel);
      ImGui::EndDisabled();
    }
    // ----- DISPLACE_TESSELLATOR -----
    if(toolHeader("Displaced Tessellate", use_displace))
    {
      ImGui::BeginDisabled(!use_displace);
      uiDisplaceTessalate(displace_args, win_handel);
      ImGui::EndDisabled();
    }
    ImGuiH::DownArrow(arrow_size);


    // ----- REMESHER -----
    if(toolHeader("Remesher", use_remesher))
    {
      ImGui::BeginDisabled(!use_remesher);
      uiRemesher(settings.tools, remesh_args);
      ImGui::EndDisabled();
    }
    ImGuiH::DownArrow(arrow_size);

    // ----- BASE -----
    loadLine("Base", ViewerSettings::eBase);

    if(settings.showAdvancedUI)
    {
      attributesOperations(scene_base);
    }


    ImGuiH::DownArrow(arrow_size);

    // ----- BAKER -----
    if(toolHeader("Baker", use_baker))
    {
      ImGui::BeginDisabled(!use_baker);
      uiBaker(bake_args, settings.tools, win_handel);
      ImGui::EndDisabled();
    }
    ImGuiH::DownArrow(arrow_size);

    // ----- RUN -----
    ImGui::Separator();
    bool run_pressed{};
    {
      std::lock_guard<std::mutex> lock(toolMutex);
      ImGuiH::PushButtonColor(tool_running ? ImGuiHCol_ButtonRed : ImGuiHCol_ButtonGreen);
      bool can_run = ((use_pretess || use_remesher || use_displace) && scene_ref->valid()) || (use_baker && scene_base->valid());
      ImGui::BeginDisabled(!can_run);
      run_pressed = ImGui::Button("RUN", ImVec2(ImGui::GetContentRegionAvail().x, 0)) && !tool_running;
      ImGui::EndDisabled();
      ImGuiH::PopButtonColor();
    }
    if(run_pressed)
    {
      settings.activtyStatus.activate("Tool running");
      tool_running               = true;
      bool copy_result           = true;
      settings.geometryView.slot = ViewerSettings::eReference;  // In case something went wrong, render the reference

      // Those are the two scenes to work with.
      std::unique_ptr<micromesh_tool::ToolScene>& reference = scene_ref->getToolScene();
      std::unique_ptr<micromesh_tool::ToolScene>& base      = scene_base->getToolScene();

      // Remove everything on the modified scene, this will be re-created from the reference
      if(use_displace || use_pretess || use_remesher)
      {
        nvh::Stopwatch st;
        vkDeviceWaitIdle(m_toolboxViewer->m_device);
        scene_base->destroy();

        // Copy the reference to Base to be the Scene to use
        copy_result = (micromesh::Result::eSuccess == base->create(reference));
        LOGI("Copy Reference to Base: %.3f\n", st.elapsed());
      }

      // Tools are done on a separated thread
      auto executeOp = [&, copy_result] {
        bool first_step_result  = copy_result;
        bool second_step_result = false;
        bool any_error          = false;

        if(first_step_result && use_pretess)
        {
          nvh::Stopwatch st;
          settings.activtyStatus.activate("Pre-Tessellation");
          first_step_result = tool_tessellate::toolPreTessellate(*m_toolboxViewer->m_toolContext, pretess_args, base);
          any_error         = any_error || !first_step_result;
          if(!first_step_result)
            LOGE("Error in: Pre - Tessellation\n");
          LOGI("Pre-Tessellation: %.3f\n", st.elapsed());
        }

        if(first_step_result && use_displace)
        {
          nvh::Stopwatch st;
          settings.activtyStatus.activate("Displace Tessellate");
          first_step_result = tool_tessellate::toolDisplacedTessellate(*toolContext, displace_args, base);
          any_error         = any_error || !first_step_result;
          if(!first_step_result)
            LOGE("Error in: Displace Tessellate\n");
          LOGI("Displace Tessellate: %.3f\n", st.elapsed());
        }

        if(first_step_result && use_remesher)
        {
          nvh::Stopwatch st;
          settings.activtyStatus.activate("Remesher");
          first_step_result = tool_remesh::toolRemesh(*toolContext, remesh_args, base);
          any_error         = any_error || !first_step_result;
          if(!first_step_result)
            LOGE("Error in: Remesher\n");
          LOGI("Remesher: %.3f\n", st.elapsed());
        }

        if(first_step_result == true)
        {
          // Things went well, showing the modified version
          settings.geometryView.slot  = ViewerSettings::eBase;
          settings.geometryView.baked = false;
        }


        // --- Second Step ---

        if(use_baker)
        {
          nvh::Stopwatch st;
          settings.activtyStatus.activate("Baker");

          fs::path bary_filename = scene_ref->getPathName().filename();
          bary_filename.replace_extension(".bary");
          bake_args.baryFilename = bary_filename.string();

          if(!reference->valid() && base->valid())
          {
            second_step_result = tool_bake::toolBake(*toolContext, bake_args, base);
            any_error          = any_error || !second_step_result;
          }
          else if(reference->valid() && base->valid())
          {
            second_step_result = tool_bake::toolBake(*toolContext, bake_args, *reference, base);
            any_error          = any_error || !second_step_result;
          }
          LOGI("Baker: %.3f\n", st.elapsed());
          m_toolboxViewer->setAllDirty(SceneDirtyFlags::eDeviceMesh);
        }

        if(second_step_result == true)
        {
          // Things went well, showing the Bake version
          settings.geometryView.baked = true;
        }

        // Tool state changed: done
        {
          std::lock_guard<std::mutex> lock(toolMutex);
          tool_running = false;
          tool_error   = any_error;
        }
        settings.activtyStatus.stop();
      };
#ifdef DONTUSEMT
      executeOp();
#else
      std::thread(executeOp).detach();    // THREAD : Tools are done on a separated thread
#endif
    }


    ImGui::EndDisabled();  // END of Reference->hasBary()


    // ----- BAKED MESH -----
    ImGui::Separator();
    {
      ViewerSettings::RenderViewSlot viewbake  = ViewerSettings::RenderViewSlot::eBase;
      ToolboxScene*                  toolScene = scene_base.get();
      if(scene_ref->hasBary())
      {
        viewbake  = ViewerSettings::RenderViewSlot::eReference;
        toolScene = scene_ref.get();
      }

      ImGui::BeginDisabled(!toolScene->hasBary());

      bool changed = false;

      ImGuiH::PushButtonColor(
          toolScene->hasBary() ? ImGuiHCol_ButtonGreen : ImGuiHCol_ButtonRed,
          (((settings.geometryView.slot == viewbake) && (settings.geometryView.baked)) || !toolScene->hasBary()) ? 1.0F : 0.F);

      float button_width = ImGui::GetColumnWidth() - ImGui::GetScrollX() - 2 * ImGui::GetStyle().ItemSpacing.x
                           - ImGui::CalcTextSize("Save").x;

      changed |= ImGui::Button("Baked Mesh", ImVec2(button_width, 0));
      ImGuiH::PopButtonColor();

      ImGui::BeginDisabled(viewbake == ViewerSettings::eReference);
      ImGui::SameLine();
      if(ImGui::Button("Save"))
      {
        std::string filename =
            NVPSystem::windowSaveFileDialog(win_handel, "Save glTF", "glTF(.gltf, .glb)|*.gltf;*.glb;");
        if(!filename.empty())
          m_toolboxViewer->saveScene(filename);
      }

      if(changed)
      {
        settings.geometryView.slot  = viewbake;
        settings.geometryView.baked = true;
        toolScene->setDirty(SceneDirtyFlags::eRasterRecord);
        toolScene->setDirty(SceneDirtyFlags::eRtxAccelerations);
        m_toolboxViewer->resetFrame();
      }

      ImGui::EndDisabled();  // viewbake == ViewerSettings::eReference
      ImGui::EndDisabled();  // toolScene->hasBary()
    }
  }
  ImGui::End();  // Micromesh

  // Notify the user when there was an error processing, otherwise it isn't
  // always obvious.
  {
    std::lock_guard<std::mutex> lock(toolMutex);
    if(tool_error)
    {
      tool_error = false;
      ImGui::OpenPopup("Error");
    }

    // Always center this window when appearing
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

    if(ImGui::BeginPopupModal("Error", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
      ImGui::Text("The operation did not complete.\nCheck the log for details\n\n");
      ImGui::Separator();
      if(ImGui::Button("OK", ImVec2(120, 0)))
      {
        ImGui::CloseCurrentPopup();
      }
      ImGui::SetItemDefaultFocus();
      ImGui::EndPopup();
    }
  }

  return false;
}

//-------------------------------------------------------------------------------------------------
// Expose the attributes on the base scene, and allow to clear them
//
void UiMicromeshProcessPipeline::attributesOperations(std::unique_ptr<ToolboxScene>& scene_base)
{
  if(scene_base->valid())
  {
    if(ImGui::TreeNode("Attributes"))
    {
      bool has_direction{false};
      bool has_bound{false};
      bool has_importance{false};
      bool has_subdiv{false};
      bool has_primflags{false};

      for(auto& m : scene_base->getToolScene()->meshes())
      {
        has_direction |= m->view().hasMeshAttributeFlags(meshops::eMeshAttributeVertexDirectionBit);
        has_bound |= m->view().hasMeshAttributeFlags(meshops::eMeshAttributeVertexDirectionBoundsBit);
        has_importance |= m->view().hasMeshAttributeFlags(meshops::eMeshAttributeVertexImportanceBit);
        has_subdiv |= m->view().hasMeshAttributeFlags(meshops::eMeshAttributeTriangleSubdivLevelsBit);
        has_primflags |= m->view().hasMeshAttributeFlags(meshops::eMeshAttributeTrianglePrimitiveFlagsBit);
      }

      auto attribFct = [&](bool has, std::string name, meshops::MeshAttributeFlags flags) {
        PE::entry(name, [&] {
          ImGui::BeginDisabled(!has);
          if(ImGui::SmallButton("clear"))
          {
            for(auto& m : scene_base->getToolScene()->meshes())
              m->view().resize(flags, 0, 0);
          }
          ImGui::EndDisabled();
          return false;
        });
      };


      PE::begin();
      attribFct(has_direction, "Directions", meshops::eMeshAttributeVertexDirectionBit);
      attribFct(has_bound, "Direction Bounds", meshops::eMeshAttributeVertexDirectionBoundsBit);
      attribFct(has_importance, "Importance", meshops::eMeshAttributeVertexImportanceBit);
      attribFct(has_subdiv, "Triangle Subdiv Level", meshops::eMeshAttributeTriangleSubdivLevelsBit);
      attribFct(has_primflags, "Triangle Primitive Flags", meshops::eMeshAttributeTrianglePrimitiveFlagsBit);
      PE::end();

      ImGui::TreePop();
    }
  }
}

//-------------------------------------------------------------------------------------------------
// Display the Reference or Base UI line with the "Load" and "Delete" button
//
void UiMicromeshProcessPipeline::loadLine(std::string name, ViewerSettings::RenderViewSlot view)
{
  ViewerSettings& settings   = m_toolboxViewer->m_settings;
  GLFWwindow*     win_handel = m_toolboxViewer->m_app->getWindowHandle();


  ImGui::PushID(name.c_str());
  bool changed = false;

  ImGui::BeginDisabled(!m_toolboxViewer->getScene(view)->valid());

  ImGuiH::PushButtonColor(m_toolboxViewer->getScene(view)->valid() ? ImGuiHCol_ButtonGreen : ImGuiHCol_ButtonRed,
                          (((settings.geometryView.slot == view) && settings.geometryView.baked == false)
                           || !m_toolboxViewer->getScene(view)->valid()) ?
                              1.0F :
                              0.F);
  // The follow is the width of the panel, minus the 2 buttons
  ImVec2 large_button_size = ImVec2(ImGui::GetColumnWidth() - ImGui::GetScrollX() - 4 * ImGui::GetStyle().ItemSpacing.x
                                        - ImGui::CalcTextSize("Load").x - ImGui::CalcTextSize("Delete").x,
                                    0);
  changed |= ImGui::Button(name.c_str(), large_button_size);
  ImGuiH::PopButtonColor();
  bool oldSettingsBaked = settings.geometryView.baked;

  ImGui::SameLine();
  if(ImGui::Button("Delete"))
  {
    vkDeviceWaitIdle(m_toolboxViewer->m_device);
    m_toolboxViewer->getScene(view)->destroy();
  }
  ImGui::EndDisabled();

  ImGui::SameLine();
  if(ImGui::Button("Load"))
  {
    changed                    = true;
    settings.geometryView.slot = view;
    std::string filename =
        NVPSystem::windowOpenFileDialog(win_handel, "Load Scene", "glTF(.gltf, .glb), OBJ(.obj)|*.gltf;*.glb;*.obj");
    if(!filename.empty())
    {
      ViewerSettings& view_settings                 = m_toolboxViewer->m_settings;
      m_toolboxViewer->m_settings.geometryView.slot = view;
      view_settings.activtyStatus.activate("Loading Scene");
      vkDeviceWaitIdle(m_toolboxViewer->m_device);
      auto executeOp = [&, filename, view] {
        m_toolboxViewer->createScene(filename, view == ViewerSettings::eReference ? SceneVersion::eReference : SceneVersion::eBase);
        view_settings.activtyStatus.stop();
      };
#ifdef DONTUSEMT
      executeOp();
#else
      std::thread(executeOp).detach();    // THREAD : Tools are done on a separated thread
#endif
    }
  }

  if(changed)
  {
    settings.geometryView.slot  = view;
    settings.geometryView.baked = false;
    m_toolboxViewer->setAllDirty(SceneDirtyFlags::eRasterRecord);
    if(oldSettingsBaked != settings.geometryView.baked)
    {
      m_toolboxViewer->setAllDirty(SceneDirtyFlags::eRtxAccelerations);
    }
    m_toolboxViewer->resetFrame();
  }
  ImGui::PopID();
};

//-------------------------------------------------------------------------------------------------
// Tool Header
// Display the name and the toggle button
//
bool UiMicromeshProcessPipeline::toolHeader(const char* name, bool& use)
{
  bool open = ImGui::CollapsingHeader(name, ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_AllowItemOverlap);
  ImGui::SameLine();
  ImGuiH::ToggleButton(name, &use);
  ImGui::TableNextColumn();
  return open;
};

//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
// NON PIPELINE APPROACH
//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------

bool UiMicromeshProcess::onUI()
{
  static bool tool_running = false;

  static tool_remesh::ToolRemeshArgs                  remesh_args;
  static tool_bake::ToolBakeArgs                      bake_args{};
  static tool_tessellate::ToolPreTessellateArgs       pretess_args{};
  static tool_tessellate::ToolDisplacedTessellateArgs displace_args{};

  static ImVec4 colActive   = ImVec4(0.1F, 0.8F, 0.1F, 1.0F);
  static ImVec4 colInactive = ImVec4(0.8F, 0.1F, 0.1F, 1.0F);

  static const char* slotNames[]     = {"Reference", "Base", "Scratch"};
  static int         item_source_idx = 0;
  static int         item_dest_idx   = 0;
  static int         hires_mesh      = 0;  // Baking : reference by default;
  static int         lores_mesh      = 1;  // Baking : base by default;

  // Setting menu
  if(ImGui::Begin("Micromesh Operations"))
  {
    // Viewer data access
    ViewerSettings&                               settings      = m_toolboxViewer->m_settings;
    std::unique_ptr<micromesh_tool::ToolContext>& toolContext   = m_toolboxViewer->m_toolContext;
    VkDevice                                      m_device      = m_toolboxViewer->m_device;
    GLFWwindow*                                   win_handel    = m_toolboxViewer->m_app->getWindowHandle();
    std::unique_ptr<ToolboxScene>&                scene_ref     = m_toolboxViewer->m_scenes[SceneVersion::eReference];
    std::unique_ptr<ToolboxScene>&                scene_base    = m_toolboxViewer->m_scenes[SceneVersion::eBase];
    std::unique_ptr<ToolboxScene>&                scene_scratch = m_toolboxViewer->m_scenes[SceneVersion::eScratch];

//
// Macro to setup some source and destination info for operations
//
#define SETUPSRCANDDEST
    assert(item_source_idx >= 0 && item_source_idx < 3);
    std::unique_ptr<micromesh_tool::ToolScene>& scene_src = m_toolboxViewer->m_scenes[item_source_idx]->getToolScene();
    std::unique_ptr<micromesh_tool::ToolScene>& scene_dst = m_toolboxViewer->m_scenes[item_dest_idx]->getToolScene();
    std::unique_ptr<ToolboxScene>&              toolBoxScene_dest = m_toolboxViewer->m_scenes[item_dest_idx];
    ViewerSettings::RenderViewSlot sourceGeometryView = (ViewerSettings::RenderViewSlot)(item_source_idx + 1);
    ViewerSettings::RenderViewSlot destGeometryView   = (ViewerSettings::RenderViewSlot)(item_dest_idx + 1);

    //
    // ----> LAMBDA to Add the source + Target and Run Button
    //
    auto runSourceToTarget = [&](const char* runName) {
      {
        bool validsrc[] = {scene_ref->valid(), scene_base->valid(), scene_scratch->valid()};
        ImGuiH::PushButtonColor(tool_running ? ImGuiHCol_ButtonRed : ImGuiHCol_ButtonGreen);
        ImGui::BeginDisabled(!validsrc[item_source_idx]);
        bool run_pressed = ImGui::Button(runName, ImVec2(ImGui::GetContentRegionAvail().x, 0));
        ImGui::EndDisabled();
        ImGuiH::PopButtonColor();
        ImGui::Separator();
        ImGui::Separator();
        ImGui::Text("");
        return run_pressed;
      }
    };
    //
    // ----> LAMBDA for copy of a scene
    //
    auto copyScene = [&](int item_source_idx, int item_dest_idx, std::unique_ptr<micromesh_tool::ToolScene>& scene_src,
                         std::unique_ptr<micromesh_tool::ToolScene>& scene_dst, std::unique_ptr<ToolboxScene>& toolBoxScene_dest) {
      nvh::Stopwatch st;
      bool           copy_result = true;
      if(item_source_idx != item_dest_idx)
      {
        scene_dst->destroy();
        copy_result = (micromesh::Result::eSuccess == scene_dst->create(scene_src));
        LOGI("Copy %s to %s: %.3f\n", slotNames[item_source_idx], slotNames[item_dest_idx], st.elapsed());
        if(!copy_result)
        {
          LOGE("Error during Copy of %s to %s: %.3f\n", slotNames[item_source_idx], slotNames[item_dest_idx], st.elapsed());
        }
        toolBoxScene_dest->setDirty(SceneDirtyFlags::eDeviceMesh);  // ask for rebuilding resources
        //toolBoxScene_dest->setDirty(SceneDirtyFlags::eRasterPipeline);  // ask for rebuilding resources
        toolBoxScene_dest->setDirty(SceneDirtyFlags::eRasterRecord);  // ask for rebuilding resources
      }
      else
      {
        LOGI("Source and Destination are the same. No copy necessary\n");
      }
      return copy_result;
    };
    //
    // ----- MESH SLOTS -----
    //
    int         selected     = 0;
    static bool displbaked[] = {false, false, false};  // to remember if we want to see baked or not
    selected |= loadSaveDelLine("Reference", scene_ref, ViewerSettings::eReference, displbaked[0]);
    selected |= loadSaveDelLine("Base", scene_base, ViewerSettings::eBase, displbaked[1]);
    selected |= loadSaveDelLine("Scratch", scene_scratch, ViewerSettings::eScratch, displbaked[2]);

    // Global subdivision level slider, shared by all tools
    globalSubdivLevel(settings, bake_args, remesh_args, pretess_args);

    if(selected)
    {
      switch(settings.geometryView.slot)
      {
        case ViewerSettings::RenderViewSlot::eReference:
          //if (item_source_idx != 0) item_dest_idx = item_source_idx;
          item_dest_idx   = 0;
          item_source_idx = 0;
          hires_mesh      = 0;
          lores_mesh      = 0;
          break;
        case ViewerSettings::RenderViewSlot::eBase:
          //if (item_source_idx != 1) item_dest_idx = item_source_idx;
          item_dest_idx   = 1;
          item_source_idx = 1;
          hires_mesh      = 1;
          lores_mesh      = 1;
          break;
        case ViewerSettings::RenderViewSlot::eScratch:
          //if (item_source_idx != 2) item_dest_idx = item_source_idx;
          item_dest_idx   = 2;
          item_source_idx = 2;
          hires_mesh      = 2;
          lores_mesh      = 2;
          break;
      }
    }
    ImGui::Text("Operators :");
    //
    // ----> source and target settings for all the operators
    //
    int        numitems    = NUMSCENES;
    const bool table_valid = ImGui::BeginTable("split", 2, ImGuiTableFlags_Resizable);
    ImGui::TableNextColumn();
    ImGui::Text("Source");  //Colored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), "Pink");
    if(ImGui::BeginListBox("##Source", ImVec2(-FLT_MIN, 4.f + float(numitems) * ImGui::GetTextLineHeightWithSpacing())))
    {
      for(int n = 0; n < IM_ARRAYSIZE(slotNames); n++)
      {
        const bool is_selected = (item_source_idx == n);
        if(ImGui::Selectable(slotNames[n], is_selected))
          item_source_idx = n;

        // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
        if(is_selected)
          ImGui::SetItemDefaultFocus();
      }
      ImGui::EndListBox();
    }
    ImGui::TableNextColumn();
    ImGui::Text("Destination");  //Colored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), "Pink");
    if(ImGui::BeginListBox("##Dest", ImVec2(-FLT_MIN, 4.f + float(numitems) * ImGui::GetTextLineHeightWithSpacing())))
    {
      for(int n = 0; n < IM_ARRAYSIZE(slotNames); n++)
      {
        const bool is_selected = (item_dest_idx == n);
        if(ImGui::Selectable(slotNames[n], is_selected))
          item_dest_idx = n;

        // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
        if(is_selected)
          ImGui::SetItemDefaultFocus();
      }
      ImGui::EndListBox();
    }
    ImGui::EndTable();


    //
    // ----- COPY -----
    //
    if(ImGui::CollapsingHeader("Simple Copy", ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_AllowItemOverlap))
    {
      if(runSourceToTarget("RUN Copy"))
      {
        settings.activtyStatus.activate("Tool running");
        tool_running     = true;
        bool copy_result = false;

        SETUPSRCANDDEST
        if(copyScene(item_source_idx, item_dest_idx, scene_src, scene_dst, toolBoxScene_dest))
        {
          settings.geometryView.slot = item_dest_idx == 0 ? ViewerSettings::eReference : ViewerSettings::eBase;
        }
        settings.activtyStatus.stop();
        tool_running = false;
      }
    }
    //
    // ----- PRE_TESSELLATOR -----
    //
    if(ImGui::CollapsingHeader("Pre-tessellator", ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_AllowItemOverlap))
    {
      uiPretesselator(pretess_args, settings.tools, win_handel);
      if(runSourceToTarget("RUN Pre-tessellation"))
      {
        settings.activtyStatus.activate("Tool running");
        tool_running     = true;
        bool copy_result = false;

        SETUPSRCANDDEST
        copy_result = copyScene(item_source_idx, item_dest_idx, scene_src, scene_dst, toolBoxScene_dest);

        settings.activtyStatus.activate("Pre-Tessellation");
        vkDeviceWaitIdle(m_toolboxViewer->m_device);  // Make sure nothing is on the fly

        //
        //----> Lambda for multithreading (or not)
        //
        auto executeOp = [&, copy_result, sourceGeometryView, destGeometryView] {
          bool result = copy_result;

          if(result)
          {
            nvh::Stopwatch st;
            result = tool_tessellate::toolPreTessellate(*m_toolboxViewer->m_toolContext, pretess_args, scene_dst);
            toolBoxScene_dest->setDirty(SceneDirtyFlags::eDeviceMesh);  // ask for rebuilding resources
            if(!result)
            {
              settings.geometryView.slot = sourceGeometryView;
              LOGE("Error in: Pre-Tessellation\n");
            }
            else
            {
              LOGI("Pre-Tessellation on %s: %.3f\n", slotNames[item_dest_idx], st.elapsed());
              settings.geometryView.slot = destGeometryView;
            }
          }
          // Tool state changed: done
          tool_running = false;
          settings.activtyStatus.stop();
        };
#ifdef DONTUSEMT
        executeOp();
#else
        std::thread(executeOp).detach();  // THREAD : Tools are done on a separated thread
#endif
      }
    }
    //
    // ----- DISPLACE_TESSELLATOR -----
    //
    if(ImGui::CollapsingHeader("Displaced Tessellate", ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_AllowItemOverlap))
    {
      uiDisplaceTessalate(displace_args, win_handel);
      if(runSourceToTarget("RUN - Tessellate Displaced"))
      {
        settings.activtyStatus.activate("Tool running");
        tool_running     = true;
        bool copy_result = false;

        SETUPSRCANDDEST
        copy_result = copyScene(item_source_idx, item_dest_idx, scene_src, scene_dst, toolBoxScene_dest);

        settings.activtyStatus.activate("Tessellating Displaced");
        vkDeviceWaitIdle(m_toolboxViewer->m_device);  // Make sure nothing is on the fly

        //
        //----> Lambda for multithreading (or not)
        //
        auto executeOp = [&, copy_result, sourceGeometryView, destGeometryView] {
          bool result = copy_result;

          if(result)
          {
            nvh::Stopwatch st;
            result = tool_tessellate::toolDisplacedTessellate(*toolContext, displace_args, scene_dst);
            if(!result)
            {
              settings.geometryView.slot = sourceGeometryView;
              LOGE("Error in: Displaced Tessellate\n");
            }
            else
            {
              LOGI("Displaced Tessellate on %s: %.3f\n", slotNames[item_dest_idx], st.elapsed());
              settings.geometryView.slot = destGeometryView;
            }
          }
          // Tool state changed: done
          tool_running = false;
          settings.activtyStatus.stop();
          toolBoxScene_dest->setDirty(SceneDirtyFlags::eDeviceMesh);  // ask for rebuilding resources
        };
#ifdef DONTUSEMT
        executeOp();
#else
        std::thread(executeOp).detach();  // THREAD : Tools are done on a separated thread
#endif
      }
    }
    // ----- REMESHER -----
    if(ImGui::CollapsingHeader("Remesher", ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_AllowItemOverlap))
    {
      uiRemesher(settings.tools, remesh_args);
      if(runSourceToTarget("RUN Remesher"))
      {
        settings.activtyStatus.activate("Tool running");
        tool_running     = true;
        bool copy_result = false;

        SETUPSRCANDDEST
        copy_result = copyScene(item_source_idx, item_dest_idx, scene_src, scene_dst, toolBoxScene_dest);
        //
        //----> Lambda for multithreading (or not)
        //
        settings.activtyStatus.activate("Remesher");
        vkDeviceWaitIdle(m_toolboxViewer->m_device);  // Make sure nothing is on the fly

        auto executeOp = [&, copy_result, sourceGeometryView, destGeometryView] {
          bool result = copy_result;

          if(result)
          {
            settings.geometryView.baked = false;

            nvh::Stopwatch st;
            result = tool_remesh::toolRemesh(*toolContext, remesh_args, scene_dst);
            if(!result)
            {
              settings.geometryView.slot = sourceGeometryView;
              LOGE("Error in: Remesher\n");
            }
            else
            {
              LOGI("Remesher on %s: %.3f\n", slotNames[item_dest_idx], st.elapsed());
              settings.geometryView.slot = destGeometryView;
            }
          }
          // Tool state changed: done
          tool_running = false;
          settings.activtyStatus.stop();
          toolBoxScene_dest->setDirty(SceneDirtyFlags::eDeviceMesh);  // ask for rebuilding resources
        };
#ifdef DONTUSEMT
        executeOp();
#else
        std::thread(executeOp).detach();  // THREAD : Tools are done on a separated thread
#endif
      }
    }
    // ----- BAKER -----
    if(ImGui::CollapsingHeader("Baker", ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_AllowItemOverlap))
    {
      uiBaker(bake_args, settings.tools, win_handel);
      //
      // ----> source and target settings for all the operators
      //
      int        numitems    = NUMSCENES;
      const bool table_valid = ImGui::BeginTable("split", 2, ImGuiTableFlags_Resizable);
      ImGui::TableNextColumn();
      ImGui::Text("High-res Mesh");  //Colored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), "Pink");
      if(ImGui::BeginListBox("##REF", ImVec2(-FLT_MIN, 4 + float(numitems) * ImGui::GetTextLineHeightWithSpacing())))
      {
        for(int n = 0; n < IM_ARRAYSIZE(slotNames); n++)
        {
          const bool is_selected = (hires_mesh == n);
          if(ImGui::Selectable(slotNames[n], is_selected))
            hires_mesh = n;

          // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
          if(is_selected)
            ImGui::SetItemDefaultFocus();
        }
        ImGui::EndListBox();
      }
      ImGui::TableNextColumn();
      ImGui::Text("Low-res Mesh (Target)");  //Colored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), "Pink");
      if(ImGui::BeginListBox("##BASE", ImVec2(-FLT_MIN, 4 + float(numitems) * ImGui::GetTextLineHeightWithSpacing())))
      {
        for(int n = 0; n < IM_ARRAYSIZE(slotNames); n++)
        {
          const bool is_selected = (lores_mesh == n);
          if(ImGui::Selectable(slotNames[n], is_selected))
            lores_mesh = n;

          // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
          if(is_selected)
            ImGui::SetItemDefaultFocus();
        }
        ImGui::EndListBox();
      }
      ImGui::EndTable();
      // ----- RUN BUTTON -----
      ImGuiH::PushButtonColor(tool_running ? ImGuiHCol_ButtonRed : ImGuiHCol_ButtonGreen);
      bool can_run = m_toolboxViewer->m_scenes[lores_mesh]->valid();
      ImGui::BeginDisabled(!can_run);
      bool run_pressed = ImGui::Button("RUN (REFERENCE + BASE -> BAKED)", ImVec2(ImGui::GetContentRegionAvail().x, 0));
      ImGui::EndDisabled();
      ImGuiH::PopButtonColor();
      if(run_pressed)
      {
        bool result = false;
        settings.activtyStatus.activate("Baker");
        vkDeviceWaitIdle(m_toolboxViewer->m_device);  // Make sure nothing is on the fly

        //
        //----> Lambda for multithreading (or not)
        //
        auto executeOp = [&] {
          nvh::Stopwatch st;
          bool           result = false;

          // The Baker doesn't give the choice of source and destination
          std::unique_ptr<micromesh_tool::ToolScene>& hires_scene = m_toolboxViewer->m_scenes[hires_mesh]->getToolScene();
          std::unique_ptr<micromesh_tool::ToolScene>& lores_scene = m_toolboxViewer->m_scenes[lores_mesh]->getToolScene();
          fs::path bary_filename = m_toolboxViewer->m_scenes[hires_mesh]->getPathName().filename();
          bary_filename.replace_extension(".bary");
          bake_args.baryFilename = bary_filename.string();

          if(!hires_scene->valid() && lores_scene->valid())
          {
            result = tool_bake::toolBake(*toolContext, bake_args, lores_scene);
          }
          else if(hires_scene->valid() && lores_scene->valid())
          {
            result = tool_bake::toolBake(*toolContext, bake_args, *hires_scene, lores_scene);
          }
          LOGI("Baker: %.3f\n", st.elapsed());
          if(result == true)
          {
            ViewerSettings::RenderViewSlot s;
            switch(lores_mesh)
            {
              case 0:
                s = ViewerSettings::eReference;
                break;
              case 1:
                s = ViewerSettings::eBase;
                break;
              case 2:
                s = ViewerSettings::eScratch;
                break;
              default:
                s = ViewerSettings::eNone;
            }
            settings.geometryView.slot  = s;
            settings.geometryView.baked = true;
            m_toolboxViewer->setAllDirty(SceneDirtyFlags::eDeviceMesh);
          }
          // Tool state changed: done
          tool_running = false;
          settings.activtyStatus.stop();
        };
#ifdef DONTUSEMT
        executeOp();
#else
        std::thread(executeOp).detach();  // THREAD : Tools are done on a separated thread
#endif
      }
    }
  }
  ImGui::End();  // Micromesh

  return false;
}

//-------------------------------------------------------------------------------------------------
// Expose the attributes on the base scene, and allow to clear them
//
void UiMicromeshProcess::attributesOperations(std::unique_ptr<ToolboxScene>& scene)
{
  if(scene->valid())
  {
    if(ImGui::TreeNode("Attributes"))
    {
      bool has_direction{false};
      bool has_bound{false};
      bool has_importance{false};
      bool has_subdiv{false};
      bool has_primflags{false};

      for(auto& m : scene->getToolScene()->meshes())
      {
        has_direction |= m->view().hasMeshAttributeFlags(meshops::eMeshAttributeVertexDirectionBit);
        has_bound |= m->view().hasMeshAttributeFlags(meshops::eMeshAttributeVertexDirectionBoundsBit);
        has_importance |= m->view().hasMeshAttributeFlags(meshops::eMeshAttributeVertexImportanceBit);
        has_subdiv |= m->view().hasMeshAttributeFlags(meshops::eMeshAttributeTriangleSubdivLevelsBit);
        has_primflags |= m->view().hasMeshAttributeFlags(meshops::eMeshAttributeTrianglePrimitiveFlagsBit);
      }

      auto attribFct = [&](bool has, std::string name, meshops::MeshAttributeFlags flags) {
        PE::entry(name, [&] {
          ImGui::BeginDisabled(!has);
          if(ImGui::SmallButton("clear"))
          {
            for(auto& m : scene->getToolScene()->meshes())
              m->view().resize(flags, 0, 0);
          }
          ImGui::EndDisabled();
          return false;
        });
      };


      PE::begin();
      attribFct(has_direction, "Directions", meshops::eMeshAttributeVertexDirectionBit);
      attribFct(has_bound, "Direction Bounds", meshops::eMeshAttributeVertexDirectionBoundsBit);
      attribFct(has_importance, "Importance", meshops::eMeshAttributeVertexImportanceBit);
      attribFct(has_subdiv, "Triangle Subdiv Level", meshops::eMeshAttributeTriangleSubdivLevelsBit);
      attribFct(has_primflags, "Triangle Primitive Flags", meshops::eMeshAttributeTrianglePrimitiveFlagsBit);
      PE::end();

      ImGui::TreePop();
    }
  }
}

//-------------------------------------------------------------------------------------------------
// Display the Reference or Base UI line with the "Load" and "Delete" button
//
int UiMicromeshProcess::loadSaveDelLine(std::string                    name,
                                        std::unique_ptr<ToolboxScene>& tboxscene,
                                        ViewerSettings::RenderViewSlot view,
                                        bool&                          dispbaked)
{
  bool            selected   = false;
  ViewerSettings& settings   = m_toolboxViewer->m_settings;
  GLFWwindow*     win_handel = m_toolboxViewer->m_app->getWindowHandle();
  bool            changed    = false;


  ImGui::PushID(name.c_str());

  float sizeAdjust = ImGui::CalcTextSize("Load").x + ImGui::CalcTextSize("Delete").x + 4 * ImGui::GetStyle().ItemSpacing.x;

  ImGui::BeginDisabled(!m_toolboxViewer->getScene(view)->valid());
  ImGuiH::PushButtonColor(m_toolboxViewer->getScene(view)->valid() ? ImGuiHCol_ButtonGreen : ImGuiHCol_ButtonRed,
                          ((view == settings.geometryView.slot) || !m_toolboxViewer->getScene(view)->valid()) ? 1.0F : 0.F);
  // The follow is the width of the panel, minus the 2 buttons
  ImVec2 large_button_size = ImVec2(ImGui::GetColumnWidth() - ImGui::GetScrollX() - 2 * ImGui::GetStyle().ItemSpacing.x
                                        - sizeAdjust - ImGui::CalcTextSize("Save").x - 2 * ImGui::CalcTextSize("µMesh").x,
                                    0);
  large_button_size.x      = std::max(ImGui::CalcTextSize("Release").x, large_button_size.x);
  bool oldSettingsBaked    = settings.geometryView.baked;

  if(ImGui::Button(name.c_str(), large_button_size))
  {
    changed                     = true;
    selected                    = true;
    settings.geometryView.slot  = view;
    settings.geometryView.baked = false;

    // Note: this code will change the overlay automatically
    if(settings.overlayView.slot != ViewerSettings::RenderViewSlot::eNone)
    {
      settings.overlayView.slot  = view;
      settings.overlayView.baked = false;
    }
    if(settings.shellView.slot != ViewerSettings::RenderViewSlot::eNone)
    {
      settings.shellView.slot  = view;
      settings.shellView.baked = false;
    }
  }
  ImGuiH::PopButtonColor();
  ImGui::EndDisabled();

  ImGui::SameLine();


  ImGui::BeginDisabled(!tboxscene->hasBary());
  dispbaked = tboxscene->hasBary() && (view == settings.geometryView.slot) && settings.geometryView.baked;
  if(ImGui::Checkbox("µMesh", &dispbaked))
  {
    changed                     = true;
    selected                    = true;
    settings.geometryView.slot  = view;
    settings.geometryView.baked = dispbaked;
    // Note: this code will change the overlay automatically
    if(settings.overlayView.slot != ViewerSettings::RenderViewSlot::eNone)
    {
      settings.overlayView.slot  = view;
      settings.overlayView.baked = dispbaked;
    }
    if(settings.shellView.slot != ViewerSettings::RenderViewSlot::eNone)
    {
      settings.shellView.slot  = view;
      settings.shellView.baked = dispbaked;
    }
  }
  ImGui::EndDisabled();

  ImGui::SameLine();
  ImGui::BeginDisabled(!m_toolboxViewer->getScene(view)->valid());
  if(ImGui::Button("Delete"))
  {
    vkDeviceWaitIdle(m_toolboxViewer->m_device);
    changed = true;
    m_toolboxViewer->getScene(view)->destroy();
  }
  ImGui::EndDisabled();

  ImGui::SameLine();
  if(ImGui::Button("Load"))
  {
    changed = true;
    std::string filename =
        NVPSystem::windowOpenFileDialog(win_handel, "Load Scene", "glTF(.gltf, .glb), OBJ(.obj)|*.gltf;*.glb;*.obj");
    if(!filename.empty())
    {
      ViewerSettings& view_settings                 = m_toolboxViewer->m_settings;
      m_toolboxViewer->m_settings.geometryView.slot = view;
      view_settings.activtyStatus.activate("Loading Scene");
      vkDeviceWaitIdle(m_toolboxViewer->m_device);
      auto executeOp = [&, filename, view] {
        SceneVersion v = eReference;
        switch(view)
        {
          case ViewerSettings::eReference:
            v = SceneVersion::eReference;
            break;
          case ViewerSettings::eBase:
            v = SceneVersion::eBase;
            break;
          case ViewerSettings::eScratch:
            v = SceneVersion::eScratch;
            break;
        }
        m_toolboxViewer->createScene(filename, v);
        dispbaked                   = m_toolboxViewer->getScene(v)->hasBary();
        settings.geometryView.baked = dispbaked;
        // Note: this code will change the overlay automatically
        if(settings.overlayView.slot != ViewerSettings::RenderViewSlot::eNone)
        {
          settings.overlayView.slot  = view;
          settings.overlayView.baked = dispbaked;
        }
        if(settings.shellView.slot != ViewerSettings::RenderViewSlot::eNone)
        {
          settings.shellView.slot  = view;
          settings.shellView.baked = dispbaked;
        }
        view_settings.activtyStatus.stop();
      };
#ifdef DONTUSEMT
      executeOp();
#else
      std::thread(executeOp).detach();    // THREAD : Tools are done on a separated thread
#endif
    }
  }
  ImGui::SameLine();
  ImGui::BeginDisabled(!m_toolboxViewer->getScene(view)->valid());
  if(ImGui::Button("Save"))
  {
    changed               = true;
    std::string  filename = NVPSystem::windowSaveFileDialog(win_handel, "Save glTF", "glTF(.gltf, .glb)|*.gltf;*.glb;");
    SceneVersion v        = eReference;
    switch(view)
    {
      case ViewerSettings::eReference:
        v = SceneVersion::eReference;
        break;
      case ViewerSettings::eBase:
        v = SceneVersion::eBase;
        break;
      case ViewerSettings::eScratch:
        v = SceneVersion::eScratch;
        break;
    }

    if(!filename.empty())
      m_toolboxViewer->saveScene(filename, v);
  }
  ImGui::EndDisabled();

  if(settings.showAdvancedUI)
  {
    attributesOperations(tboxscene);
  }

  if(changed)
  {
    m_toolboxViewer->setAllDirty(SceneDirtyFlags::eRasterRecord);
    if(oldSettingsBaked != settings.geometryView.baked)
    {
      m_toolboxViewer->setAllDirty(SceneDirtyFlags::eRtxAccelerations);
    }
    m_toolboxViewer->resetFrame();
  }
  ImGui::PopID();
  return selected;
};
