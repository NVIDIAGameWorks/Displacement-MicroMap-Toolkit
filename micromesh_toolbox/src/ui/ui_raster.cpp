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

#include "ui_raster.hpp"
#include "imgui_helper.h"
#include "toolbox_viewer.hpp"
#include "ui_color_picker.hpp"

using PE = ImGuiH::PropertyEditor;

UiRaster::UiRaster(ToolboxViewer* v)
    : _v(v)
{
}

bool UiRaster::onUI(ViewerSettings& settings)
{
  ImGui::PushID("UiRaster");

  PE::begin();
  bool          changed       = false;
  bool          redo_pipeline = false;
  bool          redo_record   = false;
  HbaoSettings& hbao          = settings.hbao;

  // All combo choices
  static const std::array<const char*, 1 + 2 * NUMSCENES> view_items = {
      "None", "Reference", "Base", "Scratch", "Reference+µMesh", "Base+µMesh", "Scratch+µMesh"};
  static const std::array<const char*, 5>  view_items_simple = {"None", "Reference", "Base", "Baked (Reference+µMesh)",
                                                                "Baked (Base+µMesh)"};
  static const std::array<const char*, 11> shading_items     = {"Default",
                                                                "Faceted",
                                                                "Phong",
                                                                "Anisotropy",
                                                                "Min/Max",
                                                                "Subdiv Level",
                                                                "Base Triangle Index",
                                                                "Compression Format",
                                                                "Heightmap Texel Frequency",
                                                                "Opposing Directions",
                                                                "Shared Position"};
  static const std::array<const char*, 8>  dbg_items         = {"None",       "Metallic", "Roughness",  "Normal",
                                                                "Base Color", "Emissive", "Txt Coords", "Direction"};
  static const std::array<const char*, 7>  colormap_items    = {"Temperature", "Viridis", "Plasma", "Magma",
                                                                "Inferno",     "Turbo",   "Batlow"};

  // Shading
  {
    redo_pipeline |= PE::entry("Shading", [&] {
      return ImGui::Combo("##Shading", (int*)&settings.shading, shading_items.data(), static_cast<int>(shading_items.size()));
    });
    redo_pipeline |= ImGuiH::hoverScrolling((int&)settings.shading, 0, static_cast<int>(shading_items.size()) - 1, -1);
  }

  // Overlay
  int overlay = settings.overlayView.slot + (!settings.overlayView.baked ? 0 : (settings.nonpipelineUI ? NUMSCENES : 2));

  {
    redo_record |= PE::entry("Overlay", [&] {
      if(settings.nonpipelineUI)
        return ImGui::Combo("##Overlay", (int*)&overlay, view_items.data(), static_cast<int>(view_items.size()));
      else
        return ImGui::Combo("##Overlay", (int*)&overlay, view_items_simple.data(), static_cast<int>(view_items_simple.size()));
    });
    redo_record |= ImGuiH::hoverScrolling(
        (int&)overlay, 0, static_cast<int>(settings.nonpipelineUI ? view_items.size() : view_items_simple.size()) - 1, -1);
    if(redo_record)
    {
      if(settings.nonpipelineUI)
      {
        settings.overlayView.slot = (ViewerSettings::RenderViewSlot)(overlay > NUMSCENES ? overlay - NUMSCENES : overlay);
        settings.overlayView.baked = overlay > NUMSCENES ? true : false;
      }
      else  // simplified case where we only have ref, base, baked
      {
        settings.overlayView.slot  = (ViewerSettings::RenderViewSlot)(overlay > 2 ? overlay - 2 : overlay);
        settings.overlayView.baked = overlay > 2 ? true : false;
      }
    }
  }

  // Shell
  int shell = settings.shellView.slot + (!settings.shellView.baked ? 0 : (settings.nonpipelineUI ? NUMSCENES : 2));

  {
    int nb_items = settings.nonpipelineUI ? NUMSCENES + 1 : static_cast<int>(view_items_simple.size()) - 1;  // We don't render "Baked" shells
    redo_record |= PE::entry("Shell", [&] {
      if(settings.nonpipelineUI)
        return ImGui::Combo("##Shell", (int*)&shell, view_items.data(), nb_items);
      else
        return ImGui::Combo("##Shell", (int*)&shell, view_items_simple.data(), nb_items);
    });
    redo_record |= ImGuiH::hoverScrolling((int&)shell, 0, nb_items - 1, -1);
    if(redo_record)
    {
      settings.shellView.slot  = (ViewerSettings::RenderViewSlot)(shell > NUMSCENES ? shell - NUMSCENES : shell);
      settings.shellView.baked = shell > NUMSCENES ? true : false;
    }
  }

  // Ambient occlusiong [x]
  PE::entry(
      "Ambient Occlusion", [&] { return ImGui::Checkbox("##4", &hbao.active); },
      "Screen-Space Ambient Occlusion (hbao)");

  if(PE::treeNode("Extra"))
  {
    // Colormap
    {
      changed |= PE::entry("Colormap", [&] {
        return ImGui::Combo("##DebugMode", (int*)&settings.colormap, colormap_items.data(),
                            static_cast<int>(colormap_items.size()));
      });
      changed |= ImGuiH::hoverScrolling((int&)settings.colormap, 0, static_cast<int>(colormap_items.size()) - 1, -1);
    }

    changed |= PE::entry("Overlay", [&] {
      bool open_popup = ImGui::ColorButton("Overlay##MyColor", settings.overlayColor, 0);
      open_popup |= openColorPicker(open_popup, settings.overlayColor, ImGuiColorEditFlags_NoAlpha);
      return open_popup;
    });


    // settings.shading == eFaceted
    changed |= PE::entry("Metallic", [&] { return ImGui::SliderFloat("#metallic", &settings.metallic, 0.F, 1.F); });
    changed |= PE::entry("Roughness", [&] { return ImGui::SliderFloat("#metallic", &settings.roughness, 0.001F, 1.F); });

    // Debug
    {
      redo_pipeline |= PE::entry("Debug Method", [&] {
        return ImGui::Combo("##DebugMode", (int*)&settings.debugMethod, dbg_items.data(), static_cast<int>(dbg_items.size()));
      });
      redo_pipeline |= ImGuiH::hoverScrolling((int&)settings.debugMethod, 0, static_cast<int>(dbg_items.size()) - 1, -1);
    }

    changed |= PE::entry(
        "Vector Length",
        [&] {
          return ImGui::SliderFloat("#vector", &settings.vectorLength, 0.001F, 1.F, "%.3f", ImGuiSliderFlags_Logarithmic);
        },
        "The visual length for normal and direction vectors");

    // Ambient occlusion low-level settings
    if(PE::treeNode("HBAO settings"))
    {
      changed |= PE::entry("Scene radius", [&]() { return ImGui::SliderFloat("Scene radius", &hbao.radius, 0.0F, 1.0F); });
      changed |=
          PE::entry("Intensity", [&]() { return ImGui::SliderFloat("intensity", &hbao.settings.intensity, 0.0F, 3.0F); });
      changed |= PE::entry("Radius", [&]() { return ImGui::SliderFloat("radius", &hbao.settings.radius, 0.001F, 3.0F); });
      changed |= PE::entry("Bias", [&]() { return ImGui::SliderFloat("bias", &hbao.settings.bias, -1.0F, 1.0F); });
      changed |= PE::entry("Blur Sharpness", [&]() {
        return ImGui::SliderFloat("blurSharpness", &hbao.settings.blurSharpness, 0.0F, 50.0F);
      });
      PE::treePop();
    }

    PE::treePop();
  }

  PE::end();

  if(redo_pipeline)
  {
    _v->setAllDirty(SceneDirtyFlags::eRasterPipeline);
    changed = true;
  }

  if(redo_record)
  {
    _v->setAllDirty(SceneDirtyFlags::eRasterRecord);
    changed = true;
  }

  ImGui::PopID();
  return changed;
}
