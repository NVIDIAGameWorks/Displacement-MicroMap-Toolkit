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

#include <memory>
#include <tool_pretessellate.hpp>
#include "meshops/meshops_mesh_view.h"
#include <microutils/microutils.hpp>
#include "micromesh/micromesh_types.h"
#include "tool_meshops_objects.hpp"
#include "tool_scene.hpp"
#include <gltf.hpp>
#include <string>
#include <filesystem>
#include <inputparser.hpp>
#include <nvh/fileoperations.hpp>
#include <nvh/nvprint.hpp>
#include <imageio/imageio.hpp>
#include <meshops/bias_scale.hpp>

// TODO: use public meshops API for attrib generation
#include <meshops_internal/umesh_util.hpp>

namespace fs = std::filesystem;

namespace tool_tessellate {

bool toolPreTessellateParse(int argc, char** argv, ToolPreTessellateArgs& args, std::ostream& os)
{
  bool              printHelp       = false;
  bool              edgeLengthBased = false;
  CommandLineParser parser(
      "pretessellate: Tessellates a mesh to match the heightmap resolution plus a --subdivLevelBias. Useful when a "
      "mesh is too coarse for baking.");
  parser.addArgument({"--help"}, &printHelp, "Print Help");
  parser.addArgument({"--subdivLevelBias"}, &args.subdivLevelBias,
                     "Target subdivision level offset. This is typically negative and used to subtract the target bake "
                     "level. Default is -5.");
  parser.addArgument({"--maxlevel"}, &args.maxSubdivLevel,
                     "Maximum subdivision level to tessellate to. Use 0 for the baryutil SDK maximum. Default is 0.");
  parser.addArgument({"--dispmapresx"}, &args.heightmapWidth, "Optional heightmap");
  parser.addArgument({"--dispmapresy"}, &args.heightmapHeight, "Output mesh.");
  parser.addArgument({"--edgeLengthBased"}, &edgeLengthBased,
                     "Ignore heightmaps and tessellate based on triangle edge lengths. The longest edge will "
                     "tessellate to --maxlevel. --maxlevel must be explicitly set.");

  if(!parser.parse(argc, argv, os) || printHelp)
  {
    parser.printHelp(printHelp ? std::cout : os);
    return false;
  }

  if(edgeLengthBased && args.maxSubdivLevel == 0)
  {
    os << "Error: must choose non-zero --maxlevel when using --edgeLengthBased" << std::endl;
    return false;
  }

  args.matchUVArea = !edgeLengthBased;

  return true;
}

bool toolPreTessellate(micromesh_tool::ToolContext&                context,
                       const ToolPreTessellateArgs&                args,
                       std::unique_ptr<micromesh_tool::ToolScene>& scene)
{
  if(scene->meshes().size() == 0)
  {
    LOGE("Error: pretessellate input input has no meshes\n");
    return false;
  }

  int maxSubdivLevel = args.maxSubdivLevel;
  if(maxSubdivLevel == 0)
  {
    // TODO: make this a meshops value?
    maxSubdivLevel = baryutils::BaryLevelsMap::MAX_LEVEL;
  }

  size_t totalTriangles    = 0;
  size_t totalNewTriangles = 0;
  for(size_t meshIndex = 0; meshIndex < scene->meshes().size(); ++meshIndex)
  {
    LOGI("Mesh %zu/%zu\n", meshIndex + 1, scene->meshes().size());
    auto& mesh     = scene->meshes()[meshIndex];
    auto& meshView = mesh->view();

    // Track new triangle counts. Assume the mesh will be skipped, update later.
    size_t initialTriangleCount = meshView.triangleCount();
    totalTriangles += initialTriangleCount;
    totalNewTriangles += initialTriangleCount;

    if(meshView.vertexTexcoords0.empty())
    {
      LOGW("Warning: did not pre-tessellate mesh %zu (no texture coordinates)\n", meshIndex);
      continue;
    }

    uint32_t heightmapWidth  = args.heightmapWidth;
    uint32_t heightmapHeight = args.heightmapHeight;

    BiasScalef biasScale;
    int        heightmapImageIndex;
    if(scene->getHeightmap(mesh->relations().material, biasScale.bias, biasScale.scale, heightmapImageIndex))
    {
      auto& image = scene->images()[heightmapImageIndex];
      if(image->info().valid())
      {
        if(heightmapWidth == 0)
        {
          heightmapWidth = static_cast<uint32_t>(image->info().width);
        }
        if(heightmapHeight == 0)
        {
          heightmapHeight = static_cast<uint32_t>(image->info().height);
        }
      }
      else
      {
        LOGE("Failed to read heightmap info on scene\n");
      }
    }

    if(heightmapWidth == 0 || heightmapHeight == 0)
    {
      LOGW("Warning: did not pre-tessellate mesh %zu (missing heightmap resolution)\n", meshIndex);
      continue;
    }

    meshops::MeshTopologyData meshTopology;
    if(micromesh_tool::buildTopologyData(context.meshopsContext(), meshView, meshTopology) != micromesh::Result::eSuccess)
    {
      LOGE("Error: failed to build mesh topology\n");
      return false;
    }

    // Generate subdivision levels and edge flags
    meshops::OpGenerateSubdivisionLevel_input baseSubdivSettings;
    baseSubdivSettings.maxSubdivLevel  = maxSubdivLevel;
    baseSubdivSettings.useTextureArea  = args.matchUVArea;
    baseSubdivSettings.subdivLevelBias = args.subdivLevelBias;
    baseSubdivSettings.textureWidth    = heightmapWidth;
    baseSubdivSettings.textureHeight   = heightmapHeight;
    uint32_t          maxGeneratedSubdivLevel;
    micromesh::Result result =
        generateMeshAttributes(context.meshopsContext(),
                               meshops::MeshAttributeFlagBits::eMeshAttributeTriangleSubdivLevelsBit
                                   | meshops::MeshAttributeFlagBits::eMeshAttributeTrianglePrimitiveFlagsBit,
                               &baseSubdivSettings, meshTopology, meshView, maxGeneratedSubdivLevel,
                               NormalReduceOp::eNormalReduceNormalizedLinear, meshops::TangentSpaceAlgorithm::eDefault);
    if(result != micromesh::Result::eSuccess)
    {
      LOGE("Error: generating attributes for base mesh %zu failed\n", meshIndex);
      return false;
    }

    // Tessellate based on the generated subdivision levels
    auto tessellatedMesh = std::make_unique<micromesh_tool::ToolMesh>(*mesh, meshops::MeshData());
    {
      meshops::OpPreTessellate_input input{};
      input.maxSubdivLevel = maxGeneratedSubdivLevel;
      input.meshView       = meshView;
      meshops::OpPreTessellate_output output{};
      output.meshView = &tessellatedMesh->view();

      result = meshops::meshopsOpPreTessellate(context.meshopsContext(), 1, &input, &output);
      if(result != micromesh::Result::eSuccess)
      {
        LOGE("Error: failed to tessellate mesh %zu\n", meshIndex);
        return false;
      }
    }

    // Subdiv levels were generated for tessellation input but should not be saved
    assert(tessellatedMesh->view().triangleSubdivisionLevels.empty());
    assert(tessellatedMesh->view().trianglePrimitiveFlags.empty());

    // Update stats. New triangles already includes initialTriangleCount.
    totalNewTriangles += tessellatedMesh->view().triangleCount() - initialTriangleCount;
    LOGI("  Triangles: %zu -> %zu\n", meshView.triangleCount(), tessellatedMesh->view().triangleCount());

    // Replace the current mesh in the scene
    // NOTE: this replaces the local mesh and meshView references!!!
    scene->setMesh(meshIndex, std::move(tessellatedMesh));
  }
  LOGI("New triangle amount: %.1f%%\n",
       ((totalTriangles == 0) ? std::numeric_limits<double>::infinity() :
                                (100.0 * static_cast<double>(totalNewTriangles) / static_cast<double>(totalTriangles))));

  return true;
}

void toolPretessellateAddRequirements(meshops::ContextConfig& contextConfig)
{
  // Nothing special needed
}

}  // namespace tool_tessellate
