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

#include "ui_rendering.hpp"
#include "imgui_helper.h"
#include "nvvk/context_vk.hpp"
#include "toolbox_viewer.hpp"

UiRendering::UiRendering(ToolboxViewer* v)
    : _v(v)
{
}

bool UiRendering::onUI(ViewerSettings& settings)
{
  using PE = ImGuiH::PropertyEditor;
  ImGui::PushID("UiRendering");


  bool changed = false;
  changed |= ImGui::RadioButton("RTX", reinterpret_cast<int*>(&settings.renderSystem), ViewerSettings::ePathtracer);
  ImGui::SameLine();
  changed |= ImGui::RadioButton("Raster", reinterpret_cast<int*>(&settings.renderSystem), ViewerSettings::eRaster);
  ImGui::SameLine();
  ImGui::TextDisabled("(R) Toggle render");

  // Warn if the driver is out of date
  nvvk::Context& ctx = *_v->m_app->getContext();
  if(ctx.m_physicalInfo.properties10.driverVersion == 2227896320 || ctx.m_physicalInfo.properties10.driverVersion == 2202780544)
  {
    ImGui::TextColored(ImVec4(0.8F, 0.5F, 0.5F, 1.0F), "Driver update required");
    ImGuiH::tooltip("This driver has a known issue with micromesh on pre-Ada GPUs", true);
  }

  PE::begin();

  // Geometry
  {
    static const std::array<const char*, 1 + 2 * NUMSCENES> view_items = {
        "None", "Reference", "Base", "Scratch", "Reference+µMesh", "Base+µMesh", "Scratch+µMesh"};
    static const std::array<const char*, 5> view_items_simple = {"None", "Reference", "Base", "Baked (Reference+µMesh)",
                                                                 "Baked (Base+µMesh)"};

    // Adjusting combo with µMesh
    int geom = settings.geometryView.slot + (!settings.geometryView.baked ? 0 : (settings.nonpipelineUI ? NUMSCENES : 2));
    bool oldSettingsBaked = settings.geometryView.baked;

    changed |= PE::entry(
        "Geometry",
        [&] {
          if(settings.nonpipelineUI)
            return ImGui::Combo("##GeoMode", (int*)&geom, view_items.data(), static_cast<int>(view_items.size()));
          else
            return ImGui::Combo("##GeoMode", (int*)&geom, view_items_simple.data(), static_cast<int>(view_items_simple.size()));
        },
        "Shortcut: F1, F2, F3");
    changed |= ImGuiH::hoverScrolling(
        (int&)geom, 0, static_cast<int>(settings.nonpipelineUI ? view_items.size() : view_items_simple.size()) - 1, -1);
    if(changed)
    {
      if(settings.nonpipelineUI)
      {
        settings.geometryView.slot  = (ViewerSettings::RenderViewSlot)(geom > NUMSCENES ? geom - NUMSCENES : geom);
        settings.geometryView.baked = geom > NUMSCENES ? true : false;
      }
      else  // simplified case where we only have ref, base, baked
      {
        settings.geometryView.slot  = (ViewerSettings::RenderViewSlot)(geom > 2 ? geom - 2 : geom);
        settings.geometryView.baked = geom > 2 ? true : false;
      }
    }

    if(changed)
    {
      _v->setAllDirty(SceneDirtyFlags::eRasterRecord);
      if(oldSettingsBaked != settings.geometryView.baked)
      {
        _v->setAllDirty(SceneDirtyFlags::eRtxAccelerations);
      }
    }
  }


  if(PE::entry(
         "Double Sided", [&]() { return ImGui::Checkbox("##2", &settings.forceDoubleSided); },
         "Forcing the material to be two-sided"))
  {
    changed = true;
    _v->setAllDirty(SceneDirtyFlags::eRasterPipeline);
    _v->setAllDirty(SceneDirtyFlags::eRtxPipeline);
  }


  PE::entry("Show Axis", [&] { return ImGui::Checkbox("##4", &settings.showAxis); });


  PE::end();

  // Display warnings related to the currently displayed scene
  sceneWarnings(settings.geometryView);

  ImGui::PopID();
  return changed;
}

void UiRendering::sceneWarnings(const ViewerSettings::RenderView& view)
{
  const ImVec4  warningColor(0.8F, 0.5F, 0.5F, 1.0F);
  ToolboxScene* scene = _v->getScene(view.slot);
  if(!scene || !scene->valid())
  {
    return;
  }

  if(scene->stats() && scene->stats()->heightmaps)
  {
    ImGui::TextColored(warningColor, "Some meshes have heightmap displacement");
    ImGuiH::tooltip(
        "When using rasterization, meshes with glTF KHR_materials_displacement will be subdivided and displaced. See "
        "Raster -> Heightmaps for settings.",
        true);
  }

  if(view.baked && scene->getToolSceneRtx() && !scene->getToolSceneVK()->barys().empty() && !scene->getToolSceneVK()->hasRtxMicromesh())
  {
    ImGui::TextColored(warningColor, "Missing " VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME " to raytrace micromap");
  }

  if(_v->m_driverMaxSubdivLevel > 0U && scene->stats() && scene->stats()->maxBarySubdivLevel > _v->m_driverMaxSubdivLevel)
  {
    ImGui::TextColored(warningColor, "Warning: .bary subdiv %u exceeds driver's supported %u",
                       scene->stats()->maxBarySubdivLevel, _v->m_driverMaxSubdivLevel);
  }
  else if(_v->m_driverMaxSubdivLevel == 0 && scene->stats() && scene->stats()->maxBarySubdivLevel > 5)
  {
    ImGui::TextColored(warningColor, "Warning: .bary subdiv %u exceeds commonly supported 5", scene->stats()->maxBarySubdivLevel);
  }

  if(scene->stats() && scene->stats()->normalmapsMissingTangents)
  {
    ImGui::TextColored(warningColor, "Missing tangents for normal maps");
  }
}
