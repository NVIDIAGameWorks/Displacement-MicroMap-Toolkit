/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "ui_about.hpp"
#include "imgui.h"
#include "toolbox_version.h"

void uiAbout(bool* p_open)
{
  if(!*p_open)
    return;

  // Always center this window when appearing
  const ImVec2 win_size(500, 300);
  ImGui::SetNextWindowSize(win_size, ImGuiCond_Appearing);
  ImVec2 center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

  if(ImGui::Begin("About##NVIDIA", p_open, ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoCollapse))
  {
    ImGui::Text("NVIDIA Micro-Mesh ToolBox v" MICROMESH_TOOLBOX_VERSION_STRING);
    ImGui::Spacing();
    ImGui::Text("https://github.com/NVIDIAGameWorks/Displacement-MicroMap-Toolkit");
    ImGui::Spacing();
    ImGui::TextWrapped(
        "micromesh_toolbox is a graphical workbench that allows inspecting micromeshes as well as interacting with "
        "some of the tools. It relies on VK_NV_mesh_shader to allow rasterized display of micromeshes. "
        "VK_KHR_acceleration_structure is required for baking micromaps. If available, VK_NV_displacement_micromap is "
        "used to render micromeshes with raytracing when choosing Rendering -> RTX. VK_NV_displacement_micromap was "
        "introduced with the RTX 40 Series Ada Lovelace architecture based GPUs. Previous RTX cards have support, but "
        "performance will be better with Ada. If you see a message about the missing extension, update to the latest "
        "driver (note: beta drivers are available at https://developer.nvidia.com/vulkan-driver).");
    ImGui::End();
  }
}
