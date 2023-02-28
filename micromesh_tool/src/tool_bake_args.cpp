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
#include <tool_bake.hpp>
#include <string>
#include <filesystem>
#include "inputparser.hpp"
#include "nvh/fileoperations.hpp"
#include "nvh/nvprint.hpp"
#include "json.hpp"

namespace tool_bake {

// Parses the --resample-extra-textures command line; returns true on success.
bool parseResampleExtraTextures(std::string argument, std::vector<ResampleExtraTexture>& parsed)
{
  if(argument.empty())
    return true;

  nlohmann::json root;
  try
  {
    root = nlohmann::json::parse(argument);
  }
  catch(const std::exception& e1)
  {
    // Could it be a file?
    if(std::filesystem::exists(argument))
    {
      const std::string content = nvh::loadFile(argument, false);
      try
      {
        root = nlohmann::json::parse(content);
      }
      catch(const std::exception& e2)
      {
        LOGE("Could not parse the content of the --resample-extra-textures file (%s). Exception text: %s\n",
             argument.c_str(), e2.what());
        return false;
      }
    }
    else
    {
      LOGE("Could not parse the --resample-extra-textures argument %s as a valid JSON structure. Exception text: %s\n",
           argument.c_str(), e1.what());
      return false;
    }
  }

  if(!root.is_array())
  {
    LOGE(
        "The content of the --resample-extra-textures argument %s was not a JSON array. Does it start and end with "
        "square brackets []?\n",
        argument.c_str());
    return false;
  }

  for(const nlohmann::json obj : root)
  {
    ResampleExtraTexture tex;
    tex.meshIdx     = obj.value("mesh", 0);
    tex.inURI       = obj.value("in", "");
    tex.outURI      = obj.value("out", "");
    tex.isNormalMap = obj.value("normal_map", false);
    parsed.push_back(tex);
  }
  return true;
}

std::vector<std::string> tokenize(std::string& str, char delim)
{
  std::istringstream       ss(str);
  std::string              item;
  std::vector<std::string> result;
  while(getline(ss, item, delim))
  {
    result.push_back(item);
  }
  return result;
};

bool toolBakeParse(int argc, char** argv, ToolBakeArgs& args, std::ostream& os)
{
  bool printHelp = false;

  namespace fs        = std::filesystem;
  std::string texturesToResample;
  std::string resampleExtraTexturesStr;
  std::string adaptiveSubdivisionMode;
  std::string heightmaps;
  std::string tangentAlgorithmName;
  std::string heightmapDirections;

  CommandLineParser parser(
      "bake: creates an Nvidia displacement micromap. Takes a base triangle mesh and computes distances needed to "
      "tessellate and displace it to match a reference mesh (--high). The result is written to a .bary file, "
      "referenced by the .gltf scene.");
  parser.addArgument({"--help", "-h"}, &printHelp, "Print Help");

  bool bakeHeightMap = false;
  bool bakeHighLow   = false;
  parser.addArgument({"--bakeHeightMap"}, &bakeHeightMap, "Legacy option. Ignored.");
  parser.addArgument({"--bakeHighLow"}, &bakeHighLow, "Legacy option. Ignored.");

  // Baking: high-low
  parser.addArgument({"--high"}, &args.highFilename,
                     "Optional high-res reference mesh. Input mesh is used if this is not given. Heightmaps, if "
                     "provided, are applied to this mesh.");
  parser.addArgument({"--resample"}, &texturesToResample,
                     "Selects textures to resample/re-bake from the high to the low level mesh: <none, normals, all>. "
                     "default=none");
  parser.addArgument({"--resample-resolution"}, &args.resampleResolution,
                     "When resampling, the resolution in pixels of each side of each of the output textures. 0 to "
                     "match "
                     "high level mesh. default=0");
  parser.addArgument({"--resample-extra-textures"}, &resampleExtraTexturesStr,
                     "Specifies extra textures, other than those in the .gltf files, to resample from the hi-res to "
                     "the "
                     "output mesh. This can be a string containing JSON, or a path to a JSON file. This must be a JSON "
                     "array of structs; each struct must specify \"in\", the path to the input image (as an absolute "
                     "path, or relative to the hi-res glTF file). Optionally, it can also specify \"out\", the path to "
                     "the output image (as an absolute path, or relative to the output glTF file; automatically "
                     "generated if not given), and/or \"mesh\", the index of the high-res and lo-res glTF primitive{} "
                     "to use for resampling (defaults to 0). Add \"normal_map\":true to mark the image as a normal "
                     "map. "
                     "Here's an example:\n    "
                     "--resample-extra-textures=\"[{\\\"in\\\":\\\"0.png\\\"},{\\\"mesh\\\":1,\\\"in\\\":\\\"1."
                     "png\\\",\\\"out\\\":\\\"1-resampled.png\\\",\\\"normal_map\\\":true}]\"");
  parser.addArgument({"--quaternion-textures-stem"}, &args.quaternionTexturesStem,
                     "Generates a quaternion texture named {argument}.{mesh index}.png for each mesh.");
  parser.addArgument({"--offset-textures-stem"}, &args.offsetTexturesStem,
                     "Generates an offset texture named {argument}.{mesh index}.png for each mesh.");
  parser.addArgument({"--height-textures-stem"}, &args.heightTexturesStem,
                     "Generates a heightmap texture named {argument}.{mesh index}.png for each mesh. Note that values "
                     "are relative to the direction vectors with direction bounds, not normals! Best used with "
                     "--subdivmode uniform and --fit-direction-bounds false.");
  parser.addArgument({"--memLimitMb"}, &args.memLimitMb,
                     "Attempt to keep memory usage below this threshold. Default is 4096. 0 to disable.");
  parser.addArgument({"--tangents"}, &tangentAlgorithmName,
                     "Tangent generation algorithm. Options: \"liani\" (default; used in Omniverse), \"lengyel\" "
                     "(commonly used algorithm, as listed in Foundations of Game Engine Development, Volume 2), "
                     "\"mikktspace\" (mikktspace.com; used in Blender and glTF)");
  parser.addArgument({"--fit-direction-bounds"}, &args.fitDirectionBounds,
                     "Compute direction vector bounds for tighter BVH. default=true");
  parser.addArgument({"--discard-direction-bounds"}, &args.discardDirectionBounds,
                     "Discards any input direction vector bounds. They will be re-created if --fit-direction-bounds is "
                     "enabled. default=true");
  parser.addArgument({"--heightmaps"}, &heightmaps,
                     "Height map filenames. One per mesh and separated with ';'. Empty names are supported. "
                     "default=glTF KHR_materials_displacement extension.");

  // Outputs
  parser.addArgument({"--bary"}, &args.baryFilename,
                     "OUTPUT: Optionally override the bary filename. default=<gltfFile>.bary");
  parser.addArgument({"--write-intermediate-meshes"}, &args.writeIntermediateMeshes,
                     "DEBUG: write heightmap displaced geometry from --bakeHighLow to ./highres_*.gltf");
#if 0
  parser.addArgument({"--split"}, &args.separateBaryFiles, "OUTPUT: Splitting in multiple bary files. default=false");
#endif

  // Generations
  parser.addArgument({"--level"}, &args.level, "Subdivision level <0-5>. default=3");
  parser.addArgument({"--compressed"}, &args.compressed, "Use compression. default=true");
  parser.addArgument({"--compressedRasterData"}, &args.compressedRasterData,
                     "If compressed add mip data for rasterization. default=false");
  parser.addArgument({"--minPSNR"}, &args.minPSNR, "Compression level. default=50.0f");

  parser.addArgument({"--maxDisplacement"}, &args.maxDisplacement,
                     "HIGH-LOW: Max lookup displacement distance, in percent of scene radius");
  parser.addArgument({"--maxDistanceFactor"}, &args.maxDistanceFactor,
                     "HIGH-LOW: Factor applied to the maximum tracing distance, useful when the displacement "
                     "bounds define a tight shell around the original geometry. default=1.0");

  parser.addArgument({"--scale"}, &args.heightmapScale,
                     "HEIGHTMAP: Override scaling value from glTF extension. default=1.0");
  parser.addArgument({"--bias"}, &args.heightmapBias,
                     "HEIGHTMAP: Override offset value from glTF extension. default=0.0");
  parser.addArgument({"--heightmapDirections"}, &heightmapDirections,
                     "HEIGHTMAP: Use raw or filtered normals for displacement directions: <normals, average, round, "
                     "sharp>. default=normals");
  parser.addArgument({"--overrideDirectionLength"}, &args.overrideDirectionLength,
                     "DISPLACEMENT: Override length of direction vector");
  parser.addArgument({"--uniDirectional"}, &args.uniDirectional,
                     "DISPLACEMENT: Only trace forwards. Default also traces backwards from the low surface");
  parser.addArgument({"--subdivmode"}, &adaptiveSubdivisionMode,
                     "DISPLACEMENT: Subdivision mode: <uniform, adaptive3d, adaptiveUV, custom>. default=custom if "
                     "NV_micromap_tooling::subdivisionLevels is provided; otherwise, uniform");
  parser.addArgument({"--subdivadaptivefactor"}, &args.adaptiveFactor,
                     "DISPLACEMENT: Subdivision adaptive factor: <0..1>. default=1");
  parser.addArgument({"--highTessBias"}, &args.highTessBias,
                     "High level mesh tessellation bias in subdivisiion levels. The high level mesh is tessellated to "
                     "match its heightmap resolution, if any. Use negative numbers to reduse the intermediate geometry "
                     "generated and improve baking performance. default=0");
  parser.addArgument({"--PNtriangles"}, &args.heightmapPNtriangles, "HEIGHTMAP: Use PN Triangles");

  if(!parser.parse(argc, argv, os) || printHelp)
  {
    parser.printHelp(printHelp ? std::cout : os);
    return false;
  }

  if(bakeHeightMap)
    os << "Ignoring unused --bakeHeightMap" << std::endl;
  if(bakeHighLow)
    os << "Ignoring unused --bakeHighLow" << std::endl;

  if(!parseResampleExtraTextures(resampleExtraTexturesStr, args.resampleExtraTextures))
  {
    return false;
  }

  args.heightmaps = tokenize(heightmaps, ';');

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

  if(adaptiveSubdivisionMode == "uniform")
    args.method = ToolBakeArgs::eUniform;
  else if(adaptiveSubdivisionMode == "adaptive3d")
    args.method = ToolBakeArgs::eAdaptive3D;
  else if(adaptiveSubdivisionMode == "adaptiveUV")
    args.method = ToolBakeArgs::eAdaptiveUV;
  else if(adaptiveSubdivisionMode == "custom")
    args.method = ToolBakeArgs::eCustom;
  else if(!adaptiveSubdivisionMode.empty())
  {
    os << "Error: unknown --subdivmode '" << adaptiveSubdivisionMode << "'." << std::endl;
    return false;
  }

  args.texturesToResample = TexturesToResample::eNone;
  if(texturesToResample == "normals")
  {
    args.texturesToResample = TexturesToResample::eNormals;
  }
  else if(texturesToResample == "all")
  {
    args.texturesToResample = TexturesToResample::eAll;
  }

  if(!tangentAlgorithmName.empty())
  {
    args.tangentAlgorithm = meshops::tangentAlgorithmFromName(tangentAlgorithmName.c_str());
    if(args.tangentAlgorithm == meshops::TangentSpaceAlgorithm::eInvalid)
    {
      os << "Error: Unrecognized --tangent algorithm name \"" << tangentAlgorithmName << "\"" << std::endl;
      return false;
    }
  }

  return true;
}

}  // namespace tool_bake
