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

#include "imgui/imgui_helper.h"
#include "ui_statistics.hpp"
#include "tool_scene.hpp"


bool UiStatistics::onUI(const micromesh_tool::ToolScene* scene)
{
  using PE = ImGuiH::PropertyEditor;

  if(scene == nullptr)
  {
    ImGui::Text("No current scene");
    return false;
  }

  uint64_t num_tri = 0;
  for(const auto& r : scene->meshes())
  {
    num_tri += r->view().triangleCount();
  }

  const bool clipboard = ImGui::Button("Copy to Clipboard");
  if(clipboard)
    ImGui::LogToClipboard();

  ImGui::PushID("Stat_Val");
  const tinygltf::Model& tiny = scene->model();
  PE::begin();
  PE::entry("Instances", std::to_string(tiny.nodes.size()));
  PE::entry("Mesh", std::to_string(tiny.meshes.size()));
  PE::entry("Materials", std::to_string(tiny.materials.size()));
  PE::entry("Triangles", std::to_string(num_tri));
  //PE::entry("Lights", std::to_string(gltf.m_lights.size()));
  PE::entry("Textures", std::to_string(tiny.textures.size()));
  PE::entry("Images", std::to_string(tiny.images.size()));
  PE::end();
  ImGui::PopID();

  if(clipboard)
    ImGui::LogFinish();

  return false;
}
