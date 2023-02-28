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
    PE::begin();
    changed |= PE::entry("Depth", [&] { return ImGui::SliderInt("#1", &settings.maxDepth, 1, 10); });
    changed |= PE::entry("Samples", [&] { return ImGui::SliderInt("#2", &settings.maxSamples, 1, 100); });
    changed |= PE::entry("Frames", [&] { return ImGui::DragInt("#3", &settings.maxFrames, 5.0F, 1, 1000000); });
    PE::end();
    ImGui::PopID();

    return changed;
  }
};
