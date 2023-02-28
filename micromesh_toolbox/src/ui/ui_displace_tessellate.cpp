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


void uiDisplaceTessalate(tool_tessellate::ToolDisplacedTessellateArgs& args, GLFWwindow* glfWin)
{
  static const char* ImageFilter = "Images|*.jpg;*.png;*.tga;*.bmp;*.psd;*.gif;*.hdr;*.pic;*;pnm;*.exr";

  static const std::array<const char*, 3> reduce_names = {"Linear", "Normalized Linear", "Tangent"};

  using PE = ImGuiH::PropertyEditor;

  if(ImGui::SmallButton("Reset##Displace"))
    args = {};
  ImGuiH::tooltip("Reset values to default");

  PE::begin();

  std::string basePath;  // base path for any heightmaps

  // #TODO: Should we expose the parameter to the user?
  //{
  //  PE::entry(
  //      "Base Path", [&]() { return ImGuiH::InputText("##T1", &args.basePath, ImGuiInputTextFlags_None); },
  //      "Input of the heightmap");
  //  ImGuiH::LoadFileButtons(glfWin, args.basePath, "Load Heightmap", ImageFilter);
  //}

  // #TODO: This one seems unused
  //  PE::entry("Heightmaps", [&] { return ImGui::Checkbox("##heightmaps", &args.heightmaps); });
  PE::entry("Tessellation Bias", [&] { return ImGui::InputInt("##TessBias", (int*)&args.heightmapTessBias); });
  PE::entry("Generate Directions", [&] { return ImGui::Checkbox("##DirectionsGen", &args.heightmapDirectionsGen); });
  ImGui::BeginDisabled(!args.heightmapDirectionsGen);
  PE::entry("Direction Type", [&] {
    return ImGui::Combo("##Op", (int*)&args.heightmapDirectionsOp, reduce_names.data(), static_cast<int>(reduce_names.size()));
  });
  ImGui::EndDisabled();
  PE::entry("Bias", [&] { return ImGui::SliderFloat("##heightmapBias", &args.heightmapBias, 0.0F, 1.0F); });
  PE::entry("Scale", [&] { return ImGui::SliderFloat("##heightmapScale", &args.heightmapScale, 0.0F, 1.0F); });
  PE::entry("PN Triangles", [&] { return ImGui::Checkbox("##heightmapPNtriangles", &args.heightmapPNtriangles); });
  PE::end();
}
