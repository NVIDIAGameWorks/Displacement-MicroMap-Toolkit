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
#include <vector>
#include <meshops/meshops_types.h>
#include <tool_context.hpp>
#include <tool_scene.hpp>

// TODO: use public meshops API for attrib generation
#include <meshops_internal/umesh_util.hpp>

namespace nvvk {
class Context;
}

struct VmaAllocator_T;

namespace tinygltf {
class Model;
}

namespace tool_bake {

// Choose the set of textures to resample/bake from the high-res file to the low-res file.
enum class TexturesToResample
{
  eNone,     // Don't resample any textures
  eNormals,  // Only resample normal maps
  eAll,      // Resample all textures
};

// A resampling operation transfers data from a hi-res mesh to a lo-res mesh.
// This struct describes an extra texture input - something that might not be
// included in a glTF material, but should go through the same resampling
// transformation. See MICROSDK-240.
struct ResampleExtraTexture
{
  int meshIdx = 0;
  // Must be either an absolute path or a path relative to the hi-res glTF file.
  std::string inURI;
  // Must be either an absolute path or a path relative to the output glTF file.
  // Automatically generated if empty.
  std::string outURI;
  bool        isNormalMap = false;
};

struct ToolBakeArgs
{
  enum BakingMethod : int
  {
    // Subdiv levels from file, if any
    eCustomOrUniform,

    // Use the target subdiv level
    eUniform,

    // Generate subdiv levels
    eAdaptive3D,
    eAdaptiveUV,

    // Use subdiv levels from the file, error out if missing
    eCustom,
  };

  std::string        outputTextureStem;  // output filename stem for generated textures
  std::string        highFilename;
  std::string        baryFilename;
  BakingMethod       method                  = eCustomOrUniform;
  int                level                   = 3;
  float              adaptiveFactor          = 1.0f;
  bool               compressed              = true;
  bool               compressedRasterData    = false;
  float              minPSNR                 = 50.0f;
  float              maxDisplacement         = 5.f;
  bool               overrideDirectionLength = false;  // true: don't use direction vector length, but maxDisplacement
  bool               uniDirectional          = false;
  bool               writeIntermediateMeshes = false;
  bool               heightmapDirectionsGen  = false;
  NormalReduceOp     heightmapDirectionsOp   = NormalReduceOp::eNormalReduceNormalizedLinear;
  TexturesToResample texturesToResample      = TexturesToResample::eNone;
  int                resampleResolution      = 0;
  meshops::TangentSpaceAlgorithm tangentAlgorithm     = meshops::TangentSpaceAlgorithm::eDefault;
  bool                           fitDirectionBounds   = true;
  bool                           heightmapPNtriangles = false;
  bool                           discardDirectionBounds = true;
#if 0
  bool               separateBaryFiles       = false;
#endif
  int highTessBias = 0;  // Target subdivision level offset for highres heightmap tessellation

  // Factor applied to the maximum tracing distance, useful when the displacement bounds define a tight
  // shell around the original geometry, where floating-point approximations may create false misses.
  // A value of 1.02 typically provides satisfying results without resulting in performance/accuracy loss.
  float maxDistanceFactor = 1.0f;


  float heightmapScale{1.0f};
  float heightmapBias{0.0f};
  // 0 == no limit. Note that the command-line has a different default!
  int                               memLimitMb = 0;
  std::vector<ResampleExtraTexture> resampleExtraTextures;
  std::string                       quaternionTexturesStem;
  std::string                       offsetTexturesStem;
  std::string                       heightTexturesStem;
  std::vector<std::string>          heightmaps;  // Per-mesh heightmaps, overriding any in gltf materials
};

// Bakes dispalcement from the base scene onto the scene specified by
// ToolBakeArgs::highFilename, or a copy of the base scene. The base scene is
// modified in-place, adding a ToolBary that contains the displacement.
bool toolBake(micromesh_tool::ToolContext& context, const ToolBakeArgs& args, std::unique_ptr<micromesh_tool::ToolScene>& base);

// Overload to provide the reference mesh directly. Ignores
// ToolBakeArgs::highFilename.
bool toolBake(micromesh_tool::ToolContext&                context,
              const ToolBakeArgs&                         args,
              const micromesh_tool::ToolScene&            reference,
              std::unique_ptr<micromesh_tool::ToolScene>& base);

bool toolBakeParse(int argc, char** argv, ToolBakeArgs& args, std::ostream& os = std::cerr);

void toolBakeAddRequirements(meshops::ContextConfig& contextConfig);

}  // namespace tool_bake
