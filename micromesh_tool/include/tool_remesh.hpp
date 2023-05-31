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

namespace tool_remesh {

struct ToolRemeshArgs
{
  float       errorThreshold{100.f};
  float       curvaturePower{1.f};
  float       importanceWeight{200.f};
  float       curvatureMaxDist{0.05f};
  float       directionBoundsFactor{1.02f};
  std::string curvatureMaxDistMode{"scenefraction"};
  bool        fitToOriginalSurface{true};
  uint32_t    maxSubdivLevel{5};
  glm::ivec2  heightmapResolution{-1, -1};
  uint32_t    heightmapTexcoord{0};  // Texture coordinates used by the displacement map

  std::string importanceMap;          // Input filename of the optional importance map
  uint32_t    importanceTexcoord{0};  // Texture coordinates to use with the importance map

  float    decimationRatio{0.1f};
  uint32_t maxVertexValence{20};
  float    importanceThreshold{1.f};
  bool     ignoreTexCoords{false};
  bool     ignoreNormals{false};
  bool     ignoreTangents{false};
  bool     ignoreDisplacementDirections{false};
  bool     disableMicromeshData{false};
  uint32_t remeshMinTriangles{0};  // Only remesh meshes with at least this many triangles
};

bool toolRemeshParse(int argc, char** argv, ToolRemeshArgs& args, std::ostream& os = std::cerr);
void toolRemeshSanitizeArgs(ToolRemeshArgs& args);
void toolRemeshAddRequirements(meshops::ContextConfig& contextConfig);
bool toolRemesh(micromesh_tool::ToolContext& context, const ToolRemeshArgs& args, std::unique_ptr<micromesh_tool::ToolScene>& scene);

}  // namespace tool_remesh
