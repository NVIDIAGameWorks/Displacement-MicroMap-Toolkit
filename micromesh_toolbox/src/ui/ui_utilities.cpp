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

#include "ui_utilities.hpp"
#include "nvpsystem.hpp"


namespace ImGuiH {

static std::string s_lastErrorMessage;
static bool        s_openErrorPopup = false;

void PushButtonColor(ImGuiHCol c, float s /*= 1.0F*/, float v /*= 1.0F*/)
{
  ImGui::PushStyleColor(ImGuiCol_Button, static_cast<ImVec4>(ImColor::HSV(float(c) / 7.0F, 0.6F * s, 0.6F * v)));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, static_cast<ImVec4>(ImColor::HSV(float(c) / 7.0F, 0.7F * s, 0.7F * v)));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, static_cast<ImVec4>(ImColor::HSV(float(c) / 7.0F, 0.8F * s, 0.8F * v)));
}

void PopButtonColor()
{
  ImGui::PopStyleColor(3);
}


struct InputTextCallback_UserData
{
  std::string*           Str;
  ImGuiInputTextCallback ChainCallback;
  void*                  ChainCallbackUserData;
};

static int inputTextCallback(ImGuiInputTextCallbackData* data)
{
  auto* user_data = static_cast<InputTextCallback_UserData*>(data->UserData);
  if(data->EventFlag == ImGuiInputTextFlags_CallbackResize)
  {
    // Resize string callback
    // If for some reason we refuse the new length (BufTextLen) and/or capacity (BufSize) we need to set them back to what we want.
    std::string* str = user_data->Str;
    IM_ASSERT(data->Buf == str->c_str());
    str->resize(data->BufTextLen);
    data->Buf = const_cast<char*>(str->c_str());
  }
  else if(user_data->ChainCallback != nullptr)
  {
    // Forward to user callback, if any
    data->UserData = user_data->ChainCallbackUserData;
    return user_data->ChainCallback(data);
  }
  return 0;
}

bool InputText(const char* label, std::string* str, ImGuiInputTextFlags flags, ImGuiInputTextCallback callback, void* user_data)
{
  IM_ASSERT((flags & ImGuiInputTextFlags_CallbackResize) == 0);
  flags |= ImGuiInputTextFlags_CallbackResize;

  InputTextCallback_UserData cb_user_data{};
  cb_user_data.Str                   = str;
  cb_user_data.ChainCallback         = callback;
  cb_user_data.ChainCallbackUserData = user_data;
  return ImGui::InputText(label, const_cast<char*>(str->c_str()), str->capacity() + 1, flags, inputTextCallback, &cb_user_data);
}


//--------------------------------------------------------------------------------------------------
// This is adding the buttons [...] [A] [B]
// It returns true if one of the buttons was clicked.
// The result string will contain the scene path
//
bool LoadFileButtons(GLFWwindow* glfwin, std::string& result, const char* title, const char* exts)
{
  ImGui::PushID(title);
  bool button_pressed{false};
  if(ImGui::SmallButton("...##1"))
  {
    button_pressed = true;
    result         = NVPSystem::windowOpenFileDialog(glfwin, title, exts);
  }
  ImGui::PopID();
  return button_pressed;
}

void ErrorMessageShow(const char* message)
{
  s_lastErrorMessage = message;
  s_openErrorPopup   = true;
}

void ErrorMessageRender()
{
  if(s_openErrorPopup)
  {
    s_openErrorPopup = false;
    ImGui::OpenPopup("Error");
  }

  // Always center this window when appearing
  ImVec2 center = ImGui::GetMainViewport()->GetCenter();
  ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

  if(ImGui::BeginPopupModal("Error", NULL, ImGuiWindowFlags_AlwaysAutoResize))
  {
    ImGui::Text("%s", s_lastErrorMessage.c_str());
    ImGui::Separator();
    if(ImGui::Button("OK", ImVec2(120, 0)))
    {
      ImGui::CloseCurrentPopup();
    }
    ImGui::SetItemDefaultFocus();
    ImGui::EndPopup();
  }
}

}  // namespace ImGuiH