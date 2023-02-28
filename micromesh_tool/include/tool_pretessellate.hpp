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
#include <string>
#include <tool_context.hpp>
#include <tool_scene.hpp>

namespace tool_tessellate {

struct ToolPreTessellateArgs
{
  uint32_t    maxSubdivLevel  = 0;  // 0 == baryutils maximum
  int32_t     subdivLevelBias = -5;
  bool        matchUVArea     = true;
  uint32_t    heightmapWidth  = 0;
  uint32_t    heightmapHeight = 0;
};

bool toolPreTessellateParse(int argc, char** argv, ToolPreTessellateArgs& args, std::ostream& os = std::cerr);
bool toolPreTessellate(micromesh_tool::ToolContext&                context,
                       const ToolPreTessellateArgs&                args,
                       std::unique_ptr<micromesh_tool::ToolScene>& scene);

void toolPretessellateAddRequirements(meshops::ContextConfig& contextConfig);
}  // namespace tool_tessellate
