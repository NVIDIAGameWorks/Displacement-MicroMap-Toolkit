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

// TODO: use public meshops API for attrib generation
#include <meshops_internal/umesh_util.hpp>

namespace tinygltf {
class Model;
}

namespace tool_tessellate {

struct ToolDisplacedTessellateArgs
{
  std::string    basePath;  // base path for .bary files and heightmaps
  bool           heightmaps{true};
  int            heightmapTessBias{0};
  bool           heightmapDirectionsGen = false;
  NormalReduceOp heightmapDirectionsOp  = NormalReduceOp::eNormalReduceNormalizedLinear;
  float          heightmapBias{0.0f};
  float          heightmapScale{1.0f};
  bool           heightmapPNtriangles = false;
};

bool toolDisplacedTessellateParse(int argc, char** argv, ToolDisplacedTessellateArgs& args, std::ostream& os = std::cerr);
bool toolDisplacedTessellate(micromesh_tool::ToolContext&                context,
                             const ToolDisplacedTessellateArgs&          args,
                             std::unique_ptr<micromesh_tool::ToolScene>& scene);

void toolDisplacedTessellateAddRequirements(meshops::ContextConfig& contextConfig);
}  // namespace tool_tessellate
