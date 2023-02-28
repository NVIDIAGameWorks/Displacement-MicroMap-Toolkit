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
#include <tool_remesh.hpp>
#include "meshops/meshops_mesh_view.h"
#include <microutils/microutils.hpp>
#include "micromesh/micromesh_types.h"
#include "meshops_internal/meshops_vertexattribs.h"
#include "tool_meshops_objects.hpp"
#include "tool_scene.hpp"
#include <gltf.hpp>
#include <string>
#include <filesystem>
#include <inputparser.hpp>
#include <nvh/fileoperations.hpp>
#include <nvh/nvprint.hpp>
#include <imageio/imageio.hpp>
#include <nvh/parallel_work.hpp>
#include "nvh/timesampler.hpp"

namespace fs = std::filesystem;

namespace tool_remesh {

bool toolRemeshParse(int argc, char** argv, ToolRemeshArgs& args, std::ostream& os)
{
  bool              printHelp       = false;
  bool              edgeLengthBased = false;
  CommandLineParser parser("remesh: decimates a triangle mesh, optimizing for micromap friendly geometry");
  parser.addArgument({"--help"}, &printHelp, "Print Help");


  parser.addArgument({"--importancemap"}, &args.importanceMap, "Optional importance map filename.");
  parser.addArgument({"--importancetexcoord"}, &args.importanceTexcoord,
                     "Texture coordinates used by the optional importance map (-1 if not used). default=-1 if no "
                     "importance map is provided, 0 otherwise");

  parser.addArgument({"--errorthreshold"}, &args.errorThreshold, "Error threshold for decimation. default=100.0");
  parser.addArgument({"--decimationratio"}, &args.decimationRatio,
                     "Minimum decimation ratio achieved by the remesher. Supersedes errorthreshold if "
                     "decimationratio>0.0. default=0.1");
  parser.addArgument({"--maxvertexvalence"}, &args.maxVertexValence,
                     "Maximum vertex valence resulting from decimation operations. default=20");
  parser.addArgument({"--maxverteximportance"}, &args.importanceThreshold,
                     "Maximum importance vertex of the vertices involved in edge collapse operations. default=1.0");

  parser.addArgument({"--curvaturepower"}, &args.curvaturePower,
                     "Exponent applied to the curvature values. default=1.0");
  parser.addArgument({"--verteximportanceweight"}, &args.importanceWeight,
                     "Weight of the per-vertex importance (curvature) in the decimation error estimate. default=100.0");
  parser.addArgument({"--curvaturemaxdist"}, &args.curvatureMaxDist,
                     "Maximum raytracing distance for curvature estimation, fraction of scene size. default=0.05");
  parser.addArgument({"--curvaturemaxdistmode"}, &args.curvatureMaxDistMode,
                     "Scale of the maximum raytracing distance for curvature estimation, either scenefraction or "
                     "worldspace. default=scenefraction");


  parser.addArgument({"--fittosurface"}, &args.fitToOriginalSurface,
                     "If enabled, adjust the decimated vertices to stick to the original surface. default=true");
  parser.addArgument({"--directionboundsfactor"}, &args.directionBoundsFactor,
                     "Additional scale to the direction bounds to guarantee they contain the surface. default=1.02");
  parser.addArgument({"--maxsubdivlevel"}, &args.maxSubdivLevel,
                     "If not -1, controls the maximum subdivision level generated during remeshing: "
                     " a triangle may not be further collapsed if its implicit "
                     "subdivision level reaches 4^level. That is based on the greater of"
                     "either the heightmap resolution of its area, "
                     "or     the number of source triangles that are represented by the output triangle. default=5");
  parser.addArgument({"--heightmapresx"}, &args.heightmapResolution.x,
                     "Optional height map horizontal resolution (-1 if not used). default=-1");
  parser.addArgument({"--heightmapresy"}, &args.heightmapResolution.y,
                     "Optional height map vertical resolution (-1 if not used). default=-1");
  parser.addArgument({"--heightmaptexcoord"}, &args.heightmapTexcoord,
                     "Optional texture coordinates used by the height map (-1 if not used). default=-1 if no "
                     "height map resolution is provided, 0 otherwise");

  parser.addArgument({"--ignoretexcoords"}, &args.ignoreTexCoords,
                     "Ignore the texture coordinate discontinuities. default=false");
  parser.addArgument({"--ignorenormals"}, &args.ignoreNormals, "Ignore the normals discontinuities. default=false");
  parser.addArgument({"--ignoretangents"}, &args.ignoreTangents, "Ignore the tangents discontinuities. default=false");
  parser.addArgument({"--ignoredisplacementdirections"}, &args.ignoreDisplacementDirections,
                     "Ignore the displacement directions discontinuities. default=false");

  parser.addArgument({"--disablemicromeshdata"}, &args.disableMicromeshData,
                     "Disable the generation of micromesh-related metadata. default=false");


  if(!parser.parse(argc, argv, os) || printHelp)
  {
    parser.printHelp(printHelp ? std::cout : os);
    return false;
  }

  toolRemeshSanitizeArgs(args);

  return true;
}

static std::string toLower(std::string str)
{
  std::string s;
  s.resize(str.size());
  for(size_t i = 0; i < str.size(); i++)
  {
    s[i] = static_cast<char>(tolower(str[i]));
  }
  return s;
}

template <typename T>
void sanitizeArgMinMax(const std::string& argName, T& arg, const T& minValue, const T& maxValue)
{
  if(arg < minValue || arg > maxValue)
  {
    std::stringstream sstr;
    T                 newValue = std::min(maxValue, std::max(minValue, arg));
    sstr << "Remesher: " << argName << " = " << arg << " out of range ( " << minValue << ", " << maxValue
         << " ) - Clamping " << argName << " = " << newValue << std::endl;
    LOGE("%s", sstr.str().c_str());
    arg = newValue;
    return;
  }
}

#define SANITIZE_ARG_MIN_MAX(_x, _min, _max) sanitizeArgMinMax(toLower(#_x), args._x, _min, _max)

void toolRemeshSanitizeArgs(ToolRemeshArgs& args)
{
  if(args.curvatureMaxDistMode != "scenefraction" && args.curvatureMaxDistMode != "worldspace")
  {
    LOGE(
        "curvaturemaxdistmode is %s, allowed values are scenefraction and worldspace - Setting default "
        "curvaturemaxdistmode = scenefraction",
        args.curvatureMaxDistMode.c_str());
    args.curvatureMaxDistMode = "scenefraction";
  }


  SANITIZE_ARG_MIN_MAX(errorThreshold, 0.f, FLT_MAX);
  SANITIZE_ARG_MIN_MAX(importanceWeight, 0.f, FLT_MAX);

  if(args.curvatureMaxDistMode == "scenefraction")
  {
    SANITIZE_ARG_MIN_MAX(curvatureMaxDist, 0.f, 1.f);
  }
  else
  {
    // World space, absolute positive value
    SANITIZE_ARG_MIN_MAX(curvatureMaxDist, 0.f, FLT_MAX);
  }
  SANITIZE_ARG_MIN_MAX(decimationRatio, 0.f, 1.f);
  SANITIZE_ARG_MIN_MAX(importanceThreshold, 0.f, 1.f);
  SANITIZE_ARG_MIN_MAX(directionBoundsFactor, 0.f, FLT_MAX);

  SANITIZE_ARG_MIN_MAX(maxVertexValence, 4u, ~0u);
}

void toolRemeshAddRequirements(meshops::ContextConfig& contextConfig)
{
  contextConfig.requiresDeviceContext = true;
}

bool toolRemesh(micromesh_tool::ToolContext& context, const ToolRemeshArgs& args, std::unique_ptr<micromesh_tool::ToolScene>& scene)
{
  if(scene->meshes().size() == 0)
  {
    LOGE("Error: remesh input has no meshes\n");
    return false;
  }

  size_t totalTriangles    = 0;
  size_t totalNewTriangles = 0;

  micromesh_tool::GenerateImportanceOperator generateImportanceOperator(context.meshopsContext());

  micromesh_tool::RemeshingOperator remeshingOperator(context.meshopsContext());

  if(!generateImportanceOperator.valid())
  {
    LOGE("Error: failed to create vertex importance operator\n");
    return false;
  }

  meshops::MeshAttributeFlags requiredMeshAttributes =
      meshops::eMeshAttributeTriangleVerticesBit | meshops::eMeshAttributeTriangleSubdivLevelsBit
      | meshops::eMeshAttributeTrianglePrimitiveFlagsBit | meshops::eMeshAttributeVertexPositionBit
      | meshops::eMeshAttributeVertexNormalBit | meshops::eMeshAttributeVertexTangentBit
      | meshops::eMeshAttributeVertexDirectionBit | meshops::eMeshAttributeVertexDirectionBoundsBit
      | meshops::eMeshAttributeVertexImportanceBit | meshops::eMeshAttributeVertexTexcoordBit;

  for(size_t meshIndex = 0; meshIndex < scene->meshes().size(); ++meshIndex)
  {

    LOGI("Mesh %zu/%zu\n", meshIndex + 1, scene->meshes().size());
    auto& mesh     = scene->meshes()[meshIndex];
    auto& meshView = mesh->view();

    // Allocate storage for output attributes, if missing
    const meshops::MeshAttributeFlags combinedMeshAttributes = (~meshView.getMeshAttributeFlags()) & requiredMeshAttributes;
    bool hadDirections = ((meshView.getMeshAttributeFlags() & meshops::eMeshAttributeVertexDirectionBit)
                          == meshops::eMeshAttributeVertexDirectionBit);
    meshView.resize(combinedMeshAttributes, meshView.triangleCount(), meshView.vertexCount());
    {
      nvh::ScopedTimer timer("Generating per-vertex directions - ");
      if(meshopsGenerateVertexDirections(context.meshopsContext(), meshView) != micromesh::Result::eSuccess)
      {
        LOGW(
            "Could not generate consistent per-vertex directions (maybe the input mesh contains non-manifold "
            "geometry?) - Remeshing may produce undefined results\n");
      }
    }

    uint64_t originalTriangleCount = meshView.triangleCount();

    meshops::DeviceMeshSettings deviceMeshSettings;
    deviceMeshSettings.usageFlags  = meshops::DeviceMeshUsageBlasBit;
    deviceMeshSettings.attribFlags = requiredMeshAttributes;
    meshops::DeviceMesh deviceMesh;
    micromesh::Result   result = micromesh::Result::eSuccess;
    {
      nvh::ScopedTimer timer("Uploading mesh to device - ");
      result = meshopsDeviceMeshCreate(context.meshopsContext(), meshView, deviceMeshSettings, &deviceMesh);
      if(result != micromesh::Result::eSuccess)
      {
        LOGE("Remesh tool: cannot create device mesh (Error %s)\n", micromesh::micromeshResultGetName(result));
        return false;
      }
    }

    meshops::OpGenerateImportance_modified importanceParameters{};
    importanceParameters.deviceMesh             = deviceMesh;
    importanceParameters.meshView               = meshView;
    importanceParameters.importanceTextureCoord = ~0u;
    importanceParameters.importancePower        = args.curvaturePower;

    meshops::Texture importanceMap;
    if(!args.importanceMap.empty())
    {
      size_t               width = 0, height = 0, components = 0;
      size_t               requiredComponents = 1;
      imageio::ImageIOData importanceData =
          imageio::loadGeneral(args.importanceMap.c_str(), &width, &height, &components, requiredComponents, 8);

      if(width == 0 || height == 0 || components == 0)
      {
        LOGE("Remesh tool: cannot load importance map %s\n", args.importanceMap.c_str());
        return false;
      }

      meshops::TextureConfig config{};
      config.width            = static_cast<uint32_t>(width);
      config.height           = static_cast<uint32_t>(height);
      config.baseFormat       = micromesh::Format::eR8_unorm;
      config.internalFormatVk = VK_FORMAT_R8_UNORM;


      result = meshops::meshopsTextureCreateFromData(context.meshopsContext(),
                                                     meshops::TextureUsageFlagBit::eTextureUsageRemesherImportanceSource,
                                                     config, width * height, importanceData, &importanceMap);
      if(result != micromesh::Result::eSuccess)
      {
        LOGE("Remesh tool: cannot create meshops texture (Error %s)\n", micromesh::micromeshResultGetName(result));
        return false;
      }
      importanceParameters.importanceTexture      = importanceMap;
      importanceParameters.importanceTextureCoord = ((args.importanceTexcoord == ~0u) ? 0 : args.importanceTexcoord);
    }

    if(args.curvatureMaxDistMode == "worldspace")
    {
      importanceParameters.rayTracingDistance = args.curvatureMaxDist;
    }
    if(args.curvatureMaxDistMode == "scenefraction")
    {
      meshops::ContextConfig contextConfig;
      result = meshops::meshopsContextGetConfig(context.meshopsContext(), &contextConfig);
      if(result != micromesh::Result::eSuccess)
      {
        LOGE("Remesh tool: cannot get meshops config (Error %s)\n", micromesh::micromeshResultGetName(result));
        return false;
      }


      float scale                             = meshopsComputeMeshViewExtent(context.meshopsContext(), meshView);
      importanceParameters.rayTracingDistance = args.curvatureMaxDist * scale;
    }
    {
      nvh::ScopedTimer timer("Generating per-vertex importance - ");
      result = meshops::meshopsOpGenerateImportance(context.meshopsContext(), generateImportanceOperator, 1, &importanceParameters);
      if(result != micromesh::Result::eSuccess)
      {
        LOGE("Remesh tool: cannot generate vertex importance (Error %s)\n", micromesh::micromeshResultGetName(result));
        return false;
      }
    }
    if(!args.importanceMap.empty())
    {
      meshops::meshopsTextureDestroy(context.meshopsContext(), importanceMap);
    }


    meshops::OpRemesh_input input{};
    input.errorThreshold        = args.errorThreshold;
    input.generateMicromeshInfo = !args.disableMicromeshData;
    input.fitToOriginalSurface  = args.fitToOriginalSurface;

    if(args.heightmapResolution.x > 0 && args.heightmapResolution.y > 0)
    {
      input.heightmapTextureCoord = (args.heightmapTexcoord != ~0u) ? args.heightmapTexcoord : 0;
    }
    else
    {
      input.heightmapTextureCoord = 0;
    }

    input.heightmapTextureWidth  = args.heightmapResolution.x;
    input.heightmapTextureHeight = args.heightmapResolution.y;
    input.importanceThreshold    = args.importanceThreshold;
    input.importanceWeight       = args.importanceWeight;
    if(args.decimationRatio > 0.f && args.decimationRatio < 1.f)
    {
      input.maxOutputTriangleCount = static_cast<uint32_t>(static_cast<float>(meshView.triangleCount()) * args.decimationRatio);
    }
    else
    {
      LOGW("Invalid decimation ratio %f ( valid range ]0, 1[ ) - reverting to error threshold %f", args.decimationRatio,
           args.errorThreshold);
      input.maxOutputTriangleCount = ~0u;
    }
    input.maxSubdivLevel                = args.maxSubdivLevel;
    input.maxVertexValence              = args.maxVertexValence;
    input.progressiveRemeshing          = false;
    input.preservedVertexAttributeFlags = 0u;

    if(!args.ignoreDisplacementDirections)
    {
      input.preservedVertexAttributeFlags |= meshops::eMeshAttributeVertexDirectionBit;
    }
    if(!args.ignoreNormals)
    {
      input.preservedVertexAttributeFlags |= meshops::eMeshAttributeVertexNormalBit;
    }
    if(!args.ignoreTangents)
    {
      input.preservedVertexAttributeFlags |= meshops::eMeshAttributeVertexTangentBit;
    }
    if(!args.ignoreTexCoords)
    {
      input.preservedVertexAttributeFlags |= meshops::eMeshAttributeVertexTexcoordBit;
    }
    input.directionBoundsFactor = args.directionBoundsFactor;
    meshops::OpRemesh_modified modified{};
    modified.deviceMesh = deviceMesh;
    modified.meshView   = &meshView;

    result = meshops::meshopsOpRemesh(context.meshopsContext(), remeshingOperator, 1, &input, &modified);
    if(result != micromesh::Result::eSuccess)
    {
      LOGE("Remesh tool: cannot remesh (Error %s)\n", micromesh::micromeshResultGetName(result));
      return false;
    }

    meshops::OpBuildTopology_input topoInput{};
    topoInput.meshView = meshView;

    meshops::MeshTopologyData       topo;
    meshops::OpBuildTopology_output topoOutput{};
    topoOutput.meshTopology = &topo;
    result                  = meshops::meshopsOpBuildTopology(context.meshopsContext(), 1, &topoInput, &topoOutput);


    meshops::OpSanitizeSubdivisionLevel_input sanitizerInput{};
    sanitizerInput.maxSubdivLevel = args.maxSubdivLevel;
    sanitizerInput.meshTopology   = &topoOutput.meshTopology->topology;
    meshops::OpSanitizeSubdivisionLevel_modified sanitizedModified{};
    sanitizedModified.meshView = meshView;


    result = meshopsOpSanitizeSubdivisionLevel(context.meshopsContext(), 1, &sanitizerInput, &sanitizedModified);

    if(result != micromesh::Result::eSuccess)
    {
      LOGE("Remesh tool: cannot sanitize subd levels (Error %s)\n", micromesh::micromeshResultGetName(result));
      return false;
    }

    meshops::meshopsDeviceMeshDestroy(context.meshopsContext(), deviceMesh);

    uint64_t finalTriangleCount = meshView.triangleCount();
    LOGI("  Triangles: %zu -> %zu\n", originalTriangleCount, finalTriangleCount);
  }

  return true;
}

}  // namespace tool_remesh
