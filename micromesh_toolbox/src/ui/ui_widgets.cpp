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

#include "ui_widgets.hpp"


void ImGuiH::ToggleButton(const char* str_id, bool* v)
{
  float height = ImGui::GetFrameHeight();
  float width  = height * 1.55F;
  float radius = height * 0.50F;

  // Right-align the button
  ImGui::SetCursorPosX(ImGui::GetWindowContentRegionMax().x - width);

  ImVec2      p         = ImGui::GetCursorScreenPos();
  ImDrawList* draw_list = ImGui::GetWindowDrawList();

  ImGui::InvisibleButton(str_id, ImVec2(width, height));
  if(ImGui::IsItemClicked())
    *v = !*v;
  ImU32 col_bg;
  if(ImGui::IsItemHovered())
    col_bg = *v ? IM_COL32(145 + 20, 211, 68 + 20, 255) : IM_COL32(218 - 20, 145 - 20, 145 - 20, 255);
  else
    col_bg = *v ? IM_COL32(145, 211, 68, 255) : IM_COL32(218, 145, 145, 255);

  draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height), col_bg, height * 0.5F);
  draw_list->AddCircleFilled(ImVec2(*v ? (p.x + width - radius) : (p.x + radius), p.y + radius), radius - 1.5F,
                             IM_COL32(255, 255, 255, 255));
}

void ImGuiH::DownArrow(ImVec2 size)
{
  float width  = size.x;
  float height = size.y;

  ImGui::SetCursorPosX((ImGui::GetWindowContentRegionMax().x - width) / 2.0F);

  ImVec2      p         = ImGui::GetCursorScreenPos();
  ImDrawList* draw_list = ImGui::GetWindowDrawList();

  // Advance the cursor
  ImGui::InvisibleButton("DownArrow", ImVec2(width, height * 2.0F));

  ImU32 col_bg = IM_COL32(200, 200, 200, 255);
  draw_list->AddRectFilled(p, ImVec2(p.x + width, p.y + height / 2.0F), col_bg, 0.0F);
  draw_list->AddTriangleFilled(ImVec2(p.x - width / 2.0F, p.y + height / 2.0F),
                               ImVec2(p.x + width * 3.0F / 2.0F, p.y + height / 2.0F),
                               ImVec2(p.x + width / 2.0F, p.y + height * 1.5F), col_bg);
}
