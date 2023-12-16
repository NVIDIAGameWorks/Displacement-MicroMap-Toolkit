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

class ToolboxViewer;
struct ViewerSettings;

class UiRaytracing
{
  using PE = ImGuiH::PropertyEditor;

  ToolboxViewer* _v;

public:
  UiRaytracing(ToolboxViewer* v)
      : _v(v){};

  bool onUI(ViewerSettings& settings)
  {
    ImGui::PushID("UiRaytracing");

    bool changed = false;
    bool specializationChanged = false;
    PE::begin();
    changed |= PE::entry("Depth", [&] { return ImGui::SliderInt("#1", &settings.maxDepth, 1, 10); });
    changed |= PE::entry("Samples", [&] { return ImGui::SliderInt("#2", &settings.maxSamples, 1, 100); });
    changed |= PE::entry("Frames", [&] { return ImGui::DragInt("#3", &settings.maxFrames, 5.0F, 1, 1000000); });
    {
      static const std::array<const char*, 2> shading_items = {"Default", "Faceted"};
      specializationChanged |= PE::entry("Shading", [&] {
        return ImGui::Combo("##Shading", (int*)&settings.shading, shading_items.data(), static_cast<int>(shading_items.size()));
      });
      specializationChanged |=
          ImGuiH::hoverScrolling((int&)settings.shading, 0, static_cast<int>(shading_items.size()) - 1, -1);
      settings.shading = static_cast<shaders::RenderShading>(
          std::min(static_cast<int>(settings.shading), static_cast<int>(shading_items.size())));
    }
    PE::end();
    ImGui::PopID();

    bool heightmapChanged = false;
    PE::begin();
    if(PE::treeNode("Heightmaps"))
    {
      ToolboxScene* currentScene = _v->getScene(settings.geometryView.slot);
      bool hasHeightmaps = currentScene && currentScene->valid() && currentScene->stats() && currentScene->stats()->heightmaps;
      ImGui::BeginDisabled(!hasHeightmaps);
      heightmapChanged |= PE::entry("Heightmap Subdiv", [&]() {
        return ImGui::SliderInt("Heightmap Subdiv", &settings.heightmapRTXSubdivLevel, 0, 5);
      });
      heightmapChanged |= PE::entry("Heightmap Scale", [&]() {
        return ImGui::InputFloat("Heightmap Scale", &settings.heightmapScale, 0.01F, 0.1F, "%0.3f");
      });
      heightmapChanged |= PE::entry("Heightmap Offset", [&]() {
        return ImGui::InputFloat("Heightmap Offset", &settings.heightmapOffset, 0.01F, 0.1F, "%0.3f");
      });
      ImGui::EndDisabled();
      PE::treePop();
    }
    PE::end();

    if(heightmapChanged)
    {
      _v->setAllDirty(SceneDirtyFlags::eRtxAccelerations);
      changed = true;
    }
    if(specializationChanged)
    {
      _v->setAllDirty(SceneDirtyFlags::eRtxPipeline);
      changed = true;
    }

    return changed;
  }
};
