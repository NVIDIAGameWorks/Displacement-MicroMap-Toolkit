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

#include "ui_micromesh_tools.hpp"


void uiPretesselator(tool_tessellate::ToolPreTessellateArgs& args, ViewerSettings::GlobalToolSettings& toolSettings, GLFWwindow* glfWin)
{
  if(ImGui::SmallButton("Reset##Pretess"))
    args = {};
  ImGuiH::tooltip("Reset values to default");

  using PE = ImGuiH::PropertyEditor;
  PE::begin();
  PE::entry("Max Subdiv Level", [&] { return ImGui::SliderInt("##maxSubdivLevel", (int*)&args.maxSubdivLevel, 0, 15); });
  ImGuiH::tooltip("A value of zero will be replaced with the internal maximum");
  PE::entry("Bake Subdiv Bias",
            [&] { return ImGui::SliderInt("##subdivLevelBias", &toolSettings.pretessellateBias, -10, 10); });
  ImGuiH::tooltip(
      "Use negative values to limit the tessellation. Visualize results with Rendering -> Shading -> Heightmap Texel "
      "Frequency.");
  PE::entry(
      "Pre-tessellate Bias",
      [&] {
        ImGui::Text("%i", args.subdivLevelBias);
        return false;
      },
      "Offset from matching heightmap resolution. Typically negative. Driven by Bake Subdiv Level");
  PE::entry("Match UV Area", [&] { return ImGui::Checkbox("##matchUVArea", &args.matchUVArea); });
  PE::entry("Heightmap Width", [&] { return ImGui::InputInt("##heightmapWidth", (int*)&args.heightmapWidth); });
  PE::entry("Heightmap Height", [&] { return ImGui::InputInt("##heightmapWidth", (int*)&args.heightmapHeight); });

  PE::end();
}
