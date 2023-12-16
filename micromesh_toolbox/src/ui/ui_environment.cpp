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

#include "ui_environment.hpp"
#include "imgui/imgui_helper.h"
#include "nvvkhl/hdr_env.hpp"
#include "nvvkhl/sky.hpp"
#include "toolbox_viewer.hpp"


UiEnvironment::UiEnvironment(ToolboxViewer* v)
    : _v(v)
{
}

bool UiEnvironment::onUI(ViewerSettings& settings)
{
  ImGui::PushID("UiEnvironment");

  using PE   = ImGuiH::PropertyEditor;
  bool reset = false;

  const bool sky_only = !(_v->m_hdrEnv && _v->m_hdrEnv->isValid());
  reset |= ImGui::RadioButton("Sky", reinterpret_cast<int*>(&settings.envSystem), ViewerSettings::eSky);
  ImGui::SameLine();
  ImGui::BeginDisabled(sky_only);
  reset |= ImGui::RadioButton("Hdr", reinterpret_cast<int*>(&settings.envSystem), ViewerSettings::eHdr);
  ImGui::EndDisabled();
  PE::begin();
  if(PE::treeNode("Sky"))
  {
    reset |= _v->m_sky->onUI();
    PE::treePop();
  }
  ImGui::BeginDisabled(sky_only);
  if(PE::treeNode("Hdr"))
  {
    reset |= PE::entry(
        "Color", [&] { return ImGui::ColorEdit3("##Color", &settings.envColor.x, ImGuiColorEditFlags_Float); },
        "Color multiplier");

    reset |= PE::entry(
        "Rotation", [&] { return ImGui::SliderAngle("Rotation", &settings.envRotation); }, "Rotating the environment");
    PE::treePop();
  }
  ImGui::EndDisabled();
  PE::end();

  ImGui::PopID();
  return reset;
}
