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

struct GLFWwindow;

// This file contains ImGUI utility functions added for the toolbox project


typedef int ImGuiHCol;  // -> enum ImGuiHCol_             // Enum: A color identifier for styling

enum ImGuiHCol_Button
{
  ImGuiHCol_ButtonRed,
  ImGuiHCol_ButtonYellow,
  ImGuiHCol_ButtonGreen,
  ImGuiHCol_ButtonTurquoise,
  ImGuiHCol_ButtonBlue,
  ImGuiHCol_ButtonPurple,
  ImGuiHCol_ButtonPink,
};

namespace ImGuiH {
void PushButtonColor(ImGuiHCol c, float s = 1.0F, float v = 1.0F);
void PopButtonColor();

bool InputText(const char* label, std::string* str, ImGuiInputTextFlags flags, ImGuiInputTextCallback callback = nullptr, void* user_data = nullptr);

bool LoadFileButtons(GLFWwindow* glfwin, std::string& result, const char* title, const char* exts);

void ErrorMessageShow(const char* message);
void ErrorMessageRender();

}  // namespace ImGuiH
