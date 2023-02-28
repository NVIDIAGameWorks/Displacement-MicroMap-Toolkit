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

#include <tool_displacedtessellate.hpp>
#include <meshops_internal/heightmap.hpp>
#include "meshops/meshops_mesh_view.h"
#include <microutils/microutils.hpp>
#include <gltf/micromesh_util.hpp>
#include "micromesh/micromesh_types.h"
#include "tiny_gltf.h"
#include "tool_meshops_objects.hpp"
#include <tool_scene.hpp>
#include <gltf.hpp>
#include <string>
#include <filesystem>
#include <inputparser.hpp>
#include <nvh/fileoperations.hpp>
#include <nvh/nvprint.hpp>
#include <baryutils/baryutils.h>
#include <stb_image.h>
#include <filesystem>
#include <vector>
#include <list>
#include <unordered_map>
#include <meshops/bias_scale.hpp>

namespace fs = std::filesystem;

namespace tool_tessellate {

bool toolDisplacedTessellateParse(int argc, char** argv, ToolDisplacedTessellateArgs& args, std::ostream& os)
{
  bool              printHelp       = false;
  bool              edgeLengthBased = false;
  std::string       heightmapDirections;
  CommandLineParser parser("displacedtessellate: tessellates and displaces a mesh with bary or heightmap displacement");
  parser.addArgument({"--help"}, &printHelp, "Print Help");
  parser.addArgument({"--tessellate-heightmaps"}, &args.heightmaps,
                     "Tessellate meshes displaced with a heightmap, matching texel frequency. Default: true");
  parser.addArgument({"--heightmap-tess-bias"}, &args.heightmapTessBias,
                     "Heightmap tessellation factor bias e.g. -1 to reduce detail. Default: 0");
  parser.addArgument({"--heightmap-scale"}, &args.heightmapScale,
                     "HEIGHTMAP: Override scaling value from glTF extension. default=1.0");
  parser.addArgument({"--heightmap-bias"}, &args.heightmapBias,
                     "HEIGHTMAP: Override offset value from glTF extension. default=0.0");
  parser.addArgument({"--heightmapDirections"}, &heightmapDirections,
                     "HEIGHTMAP: Use raw or filtered normals for displacement directions: <normals, average, round, "
                     "sharp>. default=normals");
  parser.addArgument({"--PNtriangles"}, &args.heightmapPNtriangles, "HEIGHTMAP: Use PN Triangles");

  if(!parser.parse(argc, argv, os) || printHelp)
  {
    parser.printHelp(printHelp ? std::cout : os);
    return false;
  }

  if(heightmapDirections == "normals")
  {
    args.heightmapDirectionsGen = false;
  }
  else if(heightmapDirections == "average")
  {
    args.heightmapDirectionsGen = true;
    args.heightmapDirectionsOp  = NormalReduceOp::eNormalReduceLinear;
  }
  else if(heightmapDirections == "round")
  {
    args.heightmapDirectionsGen = true;
    args.heightmapDirectionsOp  = NormalReduceOp::eNormalReduceNormalizedLinear;
  }
  else if(heightmapDirections == "sharp")
  {
    args.heightmapDirectionsGen = true;
    args.heightmapDirectionsOp  = NormalReduceOp::eNormalReduceTangent;
  }
  else if(!heightmapDirections.empty())
  {
    os << "Error: unknown --heightmapDirections '" << heightmapDirections << "'." << std::endl;
    return false;
  }

  return true;
}

static bool tessellateBary(micromesh_tool::ToolContext&      context,
                           const meshops::ResizableMeshView& meshView,
                           const bary::BasicView&            basicView,
                           int32_t                           groupIndex,
                           int32_t                           mapOffset,
                           meshops::ResizableMeshView&       tessellatedMesh)
{
  meshops::OpDisplacedTessellate_input input{};
  input.meshView                   = meshView;
  input.baryDisplacement           = &basicView;
  input.baryDisplacementGroupIndex = groupIndex;
  input.baryDisplacementMapOffset  = mapOffset;

  if(!meshView.hasMeshAttributeFlags(meshops::eMeshAttributeVertexDirectionBit))
  {
    LOGW("Warning: missing direction vectors. Using normals instead; there may be cracks.\n");
    input.meshView.vertexDirections = meshView.vertexNormals;
  }

  meshops::OpDisplacedTessellate_output output{};
  output.meshView = &tessellatedMesh;

  micromesh::Result result = meshops::meshopsOpDisplacedTessellate(context.meshopsContext(), 1, &input, &output);
  if(result != micromesh::Result::eSuccess)
  {
    return false;
  }

  return true;
}

static bool tessellateHeightmap(micromesh_tool::ToolContext&       context,
                                meshops::ResizableMeshView&        meshView,
                                const HeightMap&                   heightmap,
                                const ToolDisplacedTessellateArgs& args,
                                const BiasScalef&                  biasScale,
                                bool                               pnTriangles,
                                meshops::ResizableMeshView&        tessellatedMesh)
{
  // Create a meshops texture from the heightmap
  meshops::TextureConfig heightmapConfig{};
  heightmapConfig.width            = heightmap.width;
  heightmapConfig.height           = heightmap.height;
  heightmapConfig.mips             = 1;
  heightmapConfig.baseFormat       = micromesh::Format::eR32_sfloat;
  heightmapConfig.internalFormatVk = VK_FORMAT_R32_SFLOAT;
  micromesh_tool::MeshopsTexture heightmapTexture(context.meshopsContext(), meshops::eTextureUsageBakerHeightmapSource,
                                                  heightmapConfig, heightmap.width * heightmap.height * sizeof(float),
                                                  reinterpret_cast<const micromesh::MicromapValue*>(heightmap.raw()));
  if(!heightmapTexture.valid())
  {
    LOGE("Error: meshopsTextureCreateFromData() failed to import the heightmap texture\n");
    return false;
  }

  // Tessellation requires the max subdiv level. This will likely be generated.
  uint32_t maxSubdivLevel{0};
  if(!meshView.triangleSubdivisionLevels.empty())
  {
    maxSubdivLevel = *std::max_element(meshView.triangleSubdivisionLevels.begin(), meshView.triangleSubdivisionLevels.end());
  }

  meshops::MeshTopologyData topology;
  if(micromesh_tool::buildTopologyData(context.meshopsContext(), meshView, topology) != micromesh::Result::eSuccess)
  {
    LOGE("Error: failed to build mesh topology\n");
    return false;
  }

  // Generate any missing attributes
  {
    meshops::MeshAttributeFlags requiredAttribs = meshops::MeshAttributeFlagBits::eMeshAttributeTriangleSubdivLevelsBit
                                                  | meshops::MeshAttributeFlagBits::eMeshAttributeTrianglePrimitiveFlagsBit;
    if(args.heightmapDirectionsGen)
    {
      requiredAttribs |= meshops::MeshAttributeFlagBits::eMeshAttributeVertexDirectionBit;
    }

    meshops::OpDisplacedTessellate_properties properties;
    meshops::meshopsOpDisplacedGetProperties(context.meshopsContext(), properties);

    // Updates heightmapDesc.maxSubdivLevel if subdiv levels are generated (it is unlikely to already have them)
    meshops::OpGenerateSubdivisionLevel_input referenceSubdivSettings;
    referenceSubdivSettings.maxSubdivLevel  = properties.maxHeightmapTessellateLevel;
    referenceSubdivSettings.subdivLevelBias = args.heightmapTessBias;
    referenceSubdivSettings.textureWidth    = heightmapConfig.width;
    referenceSubdivSettings.textureHeight   = heightmapConfig.height;
    referenceSubdivSettings.useTextureArea  = true;
    micromesh::Result result =
        generateMeshAttributes(context.meshopsContext(), requiredAttribs, &referenceSubdivSettings, topology, meshView,
                               maxSubdivLevel, args.heightmapDirectionsOp, meshops::TangentSpaceAlgorithm::eInvalid);
    if(result != micromesh::Result::eSuccess)
    {
      LOGE("Error: generating attributes for reference mesh failed\n");
      return false;
    }
  }

  meshops::OpDisplacedTessellate_input input{};
  input.meshView                                = meshView;
  input.heightmap.texture                       = heightmapTexture;
  input.heightmap.bias                          = biasScale.bias;
  input.heightmap.scale                         = biasScale.scale;
  input.heightmap.maxSubdivLevel                = maxSubdivLevel;
  input.heightmap.normalizeDirections           = true;
  input.heightmap.usesVertexNormalsAsDirections = meshView.vertexDirections.empty();
  input.heightmap.pnTriangles                   = pnTriangles;
  input.meshTopology                            = topology;

  meshops::OpDisplacedTessellate_output output{};
  output.meshView = &tessellatedMesh;

  micromesh::Result result = meshops::meshopsOpDisplacedTessellate(context.meshopsContext(), 1, &input, &output);
  if(result != micromesh::Result::eSuccess)
  {
    return false;
  }

  return true;
}

bool toolDisplacedTessellate(micromesh_tool::ToolContext&                context,
                             const ToolDisplacedTessellateArgs&          args,
                             std::unique_ptr<micromesh_tool::ToolScene>& scene)
{
  if(scene->meshes().size() == 0)
  {
    LOGE("Error: displacedtessellate input input has no meshes\n");
    return false;
  }

  size_t totalTriangles    = 0;
  size_t totalNewTriangles = 0;

  for(size_t meshIndex = 0; meshIndex < scene->meshes().size(); ++meshIndex)
  {
    LOGI("Mesh %zu/%zu\n", meshIndex + 1, scene->meshes().size());
    const std::unique_ptr<micromesh_tool::ToolMesh>& mesh     = scene->meshes()[meshIndex];
    auto&                                            meshView = mesh->view();

    auto tessellatedMesh = std::make_unique<micromesh_tool::ToolMesh>(*mesh, meshops::MeshData());

    // Get attribute & displacement maps

    BiasScalef biasScale;
    int        heightmapImageIndex;

    if(mesh->relations().bary != -1)
    {
      size_t                 micromapIndex = mesh->relations().bary;
      int32_t                groupIndex    = mesh->relations().group;
      int32_t                mapOffset     = mesh->relations().mapOffset;
      const bary::BasicView& basicView     = scene->barys()[micromapIndex]->groups()[groupIndex].basic;
      if(!meshView.hasMeshAttributeFlags(meshops::eMeshAttributeTriangleVerticesBit | meshops::eMeshAttributeVertexPositionBit)
         || !(meshView.hasMeshAttributeFlags(meshops::eMeshAttributeVertexDirectionBit)
              || meshView.hasMeshAttributeFlags(meshops::eMeshAttributeVertexNormalBit)))
      {
        LOGE("Error: required attributes for base mesh %zu not provided\n", meshIndex);
        return false;
      }

      LOGI("Applying bary displacement for prim mesh %zu (bary group %i)\n", meshIndex, groupIndex);

      if(!tessellateBary(context, meshView, basicView, groupIndex, mapOffset, tessellatedMesh->view()))
      {
        LOGE("Error: failed to tessellate mesh %zu\n", meshIndex);
        return false;
      }
    }
    else if(scene->getHeightmap(mesh->relations().material, biasScale.bias, biasScale.scale, heightmapImageIndex))
    {
      // Add global scale/bias from args
      biasScale = BiasScalef(args.heightmapBias, args.heightmapScale) * biasScale;
      auto&                       image     = scene->images()[heightmapImageIndex];
      std::unique_ptr<HeightMap>& heightmap = image->heigtmap();
      if(!heightmap)
      {
        LOGE("Error: failed to load heightmap for mesh %zu\n", meshIndex);
        continue;
      }

      LOGI("Heightmap %s\n", image->relativePath().string().c_str());

      if(!tessellateHeightmap(context, meshView, *heightmap, args, biasScale, args.heightmapPNtriangles, tessellatedMesh->view()))
      {
        LOGE("Error: failed to tessellate mesh %zu\n", meshIndex);
        return false;
      }
    }

    totalTriangles += meshView.triangleCount();
    if(tessellatedMesh->view().triangleCount())
    {
      // Subdiv levels were generated for tessellation input but should not be saved
      assert(tessellatedMesh->view().triangleSubdivisionLevels.empty());
      assert(tessellatedMesh->view().trianglePrimitiveFlags.empty());

      // Update stats
      totalNewTriangles += tessellatedMesh->view().triangleCount();
      LOGI("  Triangles: %zu -> %zu\n", meshView.triangleCount(), tessellatedMesh->view().triangleCount());

      // Replace the current mesh in the scene
      // NOTE: this replaces the local mesh and meshView references!!!
      scene->setMesh(meshIndex, std::move(tessellatedMesh));
    }
    else
    {
      totalNewTriangles += meshView.triangleCount();
    }
  }

  // Remove all bary or heightmap displacement reference that have just been
  // applied to the geometry
  scene->clearBarys();
  for(auto& material : scene->materials())
  {
    material.extensions.erase("KHR_materials_displacement");
  }

  LOGI("New triangle amount: %.1f%%\n",
       ((totalTriangles == 0) ? std::numeric_limits<double>::infinity() :
                                (100.0 * static_cast<double>(totalNewTriangles) / static_cast<double>(totalTriangles))));
  return true;
}

void toolDisplacedTessellateAddRequirements(meshops::ContextConfig& contextConfig)
{
  // Nothing special needed
}

}  // namespace tool_tessellate
