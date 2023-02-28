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


//--------------------------------------------------------------------------------------------------
// This is opening a color picker with a custom color palette
// - Origin: imgui_demo
//
bool openColorPicker(bool openPopup, ImVec4& color, ImGuiColorEditFlags misc_flags = 0)
{
  // Generate a default palette. The palette will persist and can be edited.
  static bool   saved_palette_init = true;
  static ImVec4 saved_palette[32]  = {};
  if(saved_palette_init)
  {
    for(int n = 0; n < IM_ARRAYSIZE(saved_palette); n++)
    {
      ImGui::ColorConvertHSVtoRGB(static_cast<float>(n) / 31.0f, 0.8f, 0.8f, saved_palette[n].x, saved_palette[n].y,
                                  saved_palette[n].z);
      saved_palette[n].w = 1.0f;  // Alpha
    }
    saved_palette_init = false;
  }

  static ImVec4 backup_color;

  if(openPopup)
  {
    ImGui::OpenPopup("myColorPicker");
    backup_color = color;
  }

  if(ImGui::BeginPopup("myColorPicker"))
  {
    ImGui::ColorPicker4("##picker", (float*)&color, misc_flags | ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_NoSmallPreview);
    ImGui::SameLine();

    ImGui::BeginGroup();  // Lock X position
    ImGui::Text("Current");
    ImGui::ColorButton("##current", color, ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_AlphaPreviewHalf, ImVec2(60, 40));
    ImGui::SameLine();
    if(ImGui::Button("Invert"))
    {
      color.x = 1 - color.x;
      color.y = 1 - color.y;
      color.z = 1 - color.z;
    }
    ImGui::Text("Previous");
    if(ImGui::ColorButton("##previous", backup_color,
                          ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_AlphaPreviewHalf, ImVec2(60, 40)))
      color = backup_color;
    ImGui::Separator();
    ImGui::Text("Palette");
    for(int n = 0; n < IM_ARRAYSIZE(saved_palette); n++)
    {
      ImGui::PushID(n);
      if((n % 8) != 0)
        ImGui::SameLine(0.0f, ImGui::GetStyle().ItemSpacing.y);

      ImGuiColorEditFlags palette_button_flags =
          ImGuiColorEditFlags_NoAlpha | ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_NoTooltip;
      if(ImGui::ColorButton("##palette", saved_palette[n], palette_button_flags, ImVec2(20, 20)))
        color = ImVec4(saved_palette[n].x, saved_palette[n].y, saved_palette[n].z, color.w);  // Preserve alpha!

      // Allow user to drop colors into each palette entry. Note that ColorButton() is already a
      // drag source by default, unless specifying the ImGuiColorEditFlags_NoDragDrop flag.
      if(ImGui::BeginDragDropTarget())
      {
        if(const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(IMGUI_PAYLOAD_TYPE_COLOR_3F))
          memcpy((float*)&saved_palette[n], payload->Data, sizeof(float) * 3);
        if(const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(IMGUI_PAYLOAD_TYPE_COLOR_4F))
          memcpy((float*)&saved_palette[n], payload->Data, sizeof(float) * 4);
        ImGui::EndDragDropTarget();
      }

      ImGui::PopID();
    }
    ImGui::EndGroup();
    ImGui::EndPopup();
  }

  // If the color changed, return true
  return backup_color.x != color.x || backup_color.y != color.y || backup_color.z != color.z || backup_color.w != color.w;
}