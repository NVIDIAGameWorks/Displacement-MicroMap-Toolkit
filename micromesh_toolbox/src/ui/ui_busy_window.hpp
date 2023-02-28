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

#include <string>
#include "imgui.h"


//--------------------------------------------------------------------------------------------------
// This function place a Popup window in the middle of the screen, blocking all inputs and is
// use to show the application is busy doing something
//
inline void showBusyWindow(const std::string& busyReasonText)
{
  static bool state_open = false;

  // Display a modal window when loading assets or other long operation on separated thread
  if(!state_open && !busyReasonText.empty())
  {
    ImGui::OpenPopup("Busy Info");
    state_open = true;
  }

  // Position in the center of the main window when appearing
  const ImVec2 win_size(300, 75);
  ImGui::SetNextWindowSize(win_size);
  const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5F, 0.5F));

  // Window without any decoration
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 15.0);
  if(ImGui::BeginPopupModal("Busy Info", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDecoration))
  {
    if(busyReasonText.empty())
    {
      ImGui::CloseCurrentPopup();
      state_open = false;
    }
    else
    {

      // Center text in window
      const ImVec2 available = ImGui::GetContentRegionAvail();
      const ImVec2 text_size = ImGui::CalcTextSize(busyReasonText.c_str(), nullptr, false, available.x);

      ImVec2 pos;
      pos.x = (available.x - text_size.x) * 0.5F;
      pos.y = (available.y - text_size.y) * 0.2F;

      ImGui::SetCursorPosX(pos.x);
      ImGui::Text("%s", busyReasonText.c_str());

      // Add animation \ | / -
      ImGui::SetCursorPosX(available.x * 0.5F);
      ImGui::Text("%c", "|/-\\"[static_cast<int>(ImGui::GetTime() / 0.25F) & 3]);
    }
    ImGui::EndPopup();
  }
  ImGui::PopStyleVar();
}
