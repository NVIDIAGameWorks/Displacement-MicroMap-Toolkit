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
#include "imgui/imgui_helper.h"
#include "ui_widgets.hpp"
#include "ui_utilities.hpp"

#include "tool_bake.hpp"
#include "tool_displacedtessellate.hpp"
#include "tool_optimize.hpp"
#include "tool_pretessellate.hpp"
#include "tool_remesh.hpp"
#include "settings.hpp"


void uiBaker(tool_bake::ToolBakeArgs& bake_args, ViewerSettings::GlobalToolSettings& toolSettings, GLFWwindow* glfwWindow);
void uiDisplaceTessalate(tool_tessellate::ToolDisplacedTessellateArgs& args, GLFWwindow* glfWin);
void uiOptimizer(bool& use_optimizer, tool_optimize::ToolOptimizeArgs& args);
void uiPretesselator(tool_tessellate::ToolPreTessellateArgs& args, ViewerSettings::GlobalToolSettings& toolSettings, GLFWwindow* glfWin);
void uiRemesher(ViewerSettings::GlobalToolSettings& toolSettings, tool_remesh::ToolRemeshArgs& remesh_args);
