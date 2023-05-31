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


void uiOptimizer(bool& use_optimizer, tool_optimize::ToolOptimizeArgs& args)
{
  using PE = ImGuiH::PropertyEditor;

  bool open_optim = ImGui::CollapsingHeader("Optimizer", ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_AllowItemOverlap);
  ImGui::SameLine();
  ImGui::TableNextColumn();
  PE::begin();
  ImGuiH::ToggleButton("Optimizer", &use_optimizer);
  if(open_optim)
  {
    ImGui::BeginDisabled(!use_optimizer);

    PE::entry(
        "trimSubdiv", [&]() { return ImGui::InputInt("trimSubdiv", &args.trimSubdiv); },
        "Reduces the subdivision level of each triangle to at most this number. Removes unused subdivision levels - "
        "like "
        "reducing the resolution of an image. (Default: 4)");

    PE::entry(
        "Min PNSR", [&]() { return ImGui::InputFloat("minPNSR", &args.psnr); },
        "Minimum Peak Signal-to-Noise Ratio in decibels for lossy compression. 20 is very low quality; 30 is low "
        "quality; 40 is normal quality; 50 is high quality. (Default: 40)");

    PE::entry(
        "Validate Edges", [&]() { return ImGui::Checkbox("validateEdges", &args.validateEdges); },
        "Validates that the input and output displacements are watertight. (Default: false)");

    ImGui::EndDisabled();
  }
  PE::end();
}
