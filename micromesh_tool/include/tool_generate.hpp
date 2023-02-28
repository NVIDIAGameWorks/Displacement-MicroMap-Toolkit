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

#include <inttypes.h>
#include <string>
#include <tool_context.hpp>
#include <tool_scene.hpp>

namespace tool_generate {

struct ToolGenerateArgs
{
  enum Geometry
  {
    GEOMETRY_PLANE,
    GEOMETRY_CUBE,
    GEOMETRY_TERRAIN,
    GEOMETRY_SPHERE,
    GEOMETRY_ROCK,
  };

  Geometry    geometry   = GEOMETRY_CUBE;
  uint32_t    resolution = 128;
};

bool toolGenerateParse(int argc, char** argv, ToolGenerateArgs& args);
void toolGenerateAddRequirements(meshops::ContextConfig& contextConfig);
bool toolGenerate(micromesh_tool::ToolContext& context, const ToolGenerateArgs& args, std::unique_ptr<micromesh_tool::ToolScene>& scene);

}  // namespace tool_generate