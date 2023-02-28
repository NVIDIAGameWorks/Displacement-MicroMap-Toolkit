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

#include <iostream>
#include <memory>
#include <tool_context.hpp>
#include <tool_scene.hpp>

namespace tool_optimize {
struct ToolOptimizeArgs
{
  // Reduces the subdivision level of each triangle to at most this number.
  int trimSubdiv = 5;
  // Minimum Peak Signal-to-Noise Ratio in decibels for lossy compression.
  float psnr = 40.0;
  // validateEdges is true by default only in debug mode.
  bool validateEdges =
#ifdef NDEBUG
      false;
#else
      true;
#endif
};

bool toolOptimizeParse(int argc, char** argv, ToolOptimizeArgs& args, std::ostream& os = std::cerr);
bool toolOptimize(micromesh_tool::ToolContext& context, const ToolOptimizeArgs& args, std::unique_ptr<micromesh_tool::ToolScene>& scene);

void toolOptimizeAddRequirements(meshops::ContextConfig& contextConfig);
}  // namespace tool_optimize
