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

#include <cstdlib>
#include <memory>
#include <gltf/micromesh_util.hpp>
#include <microutils/microutils.hpp>
#include <tool_bake.hpp>
#include <tool_generate.hpp>
#include <tool_merge_args.hpp>
#include <tool_merge.hpp>
#include <tool_optimize.hpp>
#include <tool_pretessellate.hpp>
#include <tool_displacedtessellate.hpp>
#include <tool_remesh.hpp>
#include <tool_scene.hpp>
#include <nvh/nvprint.hpp>
#include <nvh/timesampler.hpp>
#include <inputparser.hpp>
#include <debug_util.h>
#include <filesystem>
#include <functional>
#include <map>
#include <any>
#include <thread>
#include <tool_version.h>

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
#ifdef _WIN32
  fixAbortOnWindows();
#endif

  try
  {
    std::string   exeName     = fs::path(argv[0]).filename().string();
    bool          printHelp   = false;
    bool          parseResult = true;
    bool          verbose     = false;
    std::ostream& parseError  = std::cerr;
    std::string   inputFilename;
    std::string   outputFilename;

    // Create a top level parser to take global input and output filenames, then
    // add individual tools that can be joined together in one long pipeline.
    MultiCommandLineParser parser(
        exeName + " (v" MICROMESH_TOOL_VERSION_STRING "): Tool for processing and baking micromeshes");
    parser.global().addArgument({"--help", "-h"}, &printHelp,
                                "Prints this command help text. May be passed to sub-commands.");
    parser.global().addArgument({"--input"}, &inputFilename, "Input scene (*.gltf|*.glb|*.obj)");
    parser.global().addArgument({"--output"}, &outputFilename, "Output scene (*.gltf)");
    parser.global().addArgument({"--verbose"}, &verbose, "Show log level info, not just errors and warnings.");
    parser.addSubcommand("generate",
                         "Creates test meshes with textures. Use displacedtessellate to create real geometry from "
                         "meshes with heightmaps");
    parser.addSubcommand("pretessellate",
                         "Tessellates a mesh to match the heightmap resolution plus a --subdivLevelBias. Useful when a "
                         "mesh is too coarse for baking");
    parser.addSubcommand("bake",
                         "Creates an Nvidia displacement micromap. Takes a base triangle mesh and computes distances "
                         "needed to tessellate and displace it to match a reference mesh (--high). The result is "
                         "written to a .bary file, referenced by the .gltf scene");
    parser.addSubcommand("displacedtessellate", "Tessellates and displaces a mesh with bary or heightmap displacement");
    parser.addSubcommand("remesh", "Decimates a triangle mesh, optimizing for micromap friendly geometry");
    parser.addSubcommand("merge", "Merges multiple glTF files into one, with support for micromesh extensions.");
    parser.addSubcommand("print", "Prints mesh data preview at this point in the pipeline.");
    parser.addSubcommand("optimize", "Trims and compresses displacement data to save space and improve performance.");

    // Parse the top level commands. Child commands are stored in MultiCommandLineParser::SubcommandArgs objects.
    if(!parser.parse(argc, argv, parseError))
    {
      parseResult = false;
      verbose     = true;
    }

    if(printHelp)
    {
      parser.printHelp(std::cout);
      std::cout << std::endl << "Examples" << std::endl << std::endl;
      std::cout << exeName << " --input reefcrab/reefcrab.gltf --output reefcrab_with_micromap.gltf {remesh --decimationratio 0.1} {bake --level 5}"
                << std::endl
                << std::endl;
      std::cout << exeName << " --input wall/wall_quad/wall.gltf --output wall_with_micromap.gltf {pretessellate} {bake --level 5}"
                << std::endl
                << std::endl;
      std::cout << exeName
                << " --input wall/wall_geometry/quad_pretess.gltf --output resampled_wall_with_micromap.gltf {bake "
                   "--high wall/wall_geometry/wall.gltf --level 5 --resample all}"
                << std::endl
                << std::endl;
      std::cout << exeName
                << " --input wall/wall_geometry/quad_pretess.gltf --output resampled_wall_with_micromap.gltf {bake "
                   "--high wall/wall_decimated/wall.gltf --level 5 --resample all}"
                << std::endl
                << std::endl;
      return EXIT_SUCCESS;
    }

    if(!verbose)
    {
      nvprintSetConsoleLogging(false, LOGBITS_ALL);
      nvprintSetConsoleLogging(true, LOGBITS_WARNINGS);
    }

    // Generate a name for the output glTF if it's empty and we're only baking (legacy feature)
    if(outputFilename.empty() && parser.subcommands().size() == 1 && parser.subcommands().front().first == "bake")
    {
      auto fpath = fs::path(inputFilename);
      auto fname = fpath.stem();
      fname      = fname.string() + std::string("_B.gltf");
      fpath.replace_filename(fname);
      outputFilename = fpath.string();
    }

    if(!parser.subcommands().empty() && parser.subcommands()[0].first == "generate")
    {
      if(!inputFilename.empty())
      {
        std::cerr << "Error: --input given but first operation is 'generate'." << std::endl;
        parseResult = false;
      }
      if(outputFilename.empty())
      {
        std::cerr << "Error: output filename is required." << std::endl;
        parseResult = false;
      }
    }
    else
    {
      if(inputFilename.empty() || outputFilename.empty())
      {
        std::cerr << "Error: input and output filenames are required." << std::endl;
        parseResult = false;
      }
    }

    if(parser.subcommands().size() == 0)
    {
      parseError << "Missing subcommand" << std::endl;
      parseResult = false;
    }

    meshops::ContextConfig meshopsContextConfig;
    meshopsContextConfig.messageCallback = microutils::makeDefaultMessageCallback();
    meshopsContextConfig.threadCount     = std::thread::hardware_concurrency();
    meshopsContextConfig.verbosityLevel  = 999;

    // Parse subcommands for each tool_*.
    std::vector<std::any> subcommandArgs;
    for(auto& subcommand : parser.subcommands())
    {
      subcommandArgs.emplace_back();
      auto& args = subcommandArgs.back();
      if(subcommand.first == "generate")
      {
        args = tool_generate::ToolGenerateArgs{};
        if(!tool_generate::toolGenerateParse(subcommand.second.count(), subcommand.second.argv(),
                                             std::any_cast<tool_generate::ToolGenerateArgs&>(args)))
        {
          parseError << std::endl;
          parseResult = false;
        }

        tool_generate::toolGenerateAddRequirements(meshopsContextConfig);
      }
      else if(subcommand.first == "pretessellate")
      {
        args = tool_tessellate::ToolPreTessellateArgs{};
        if(!tool_tessellate::toolPreTessellateParse(subcommand.second.count(), subcommand.second.argv(),
                                                    std::any_cast<tool_tessellate::ToolPreTessellateArgs&>(args), parseError))
        {
          parseError << std::endl;

          parseResult = false;
        }

        tool_tessellate::toolPretessellateAddRequirements(meshopsContextConfig);
      }
      else if(subcommand.first == "bake")
      {
        args = tool_bake::ToolBakeArgs{};
        if(!tool_bake::toolBakeParse(subcommand.second.count(), subcommand.second.argv(),
                                     std::any_cast<tool_bake::ToolBakeArgs&>(args), parseError))
        {
          parseError << std::endl;
          parseResult = false;
        }

        tool_bake::toolBakeAddRequirements(meshopsContextConfig);
      }
      else if(subcommand.first == "displacedtessellate")
      {
        args = tool_tessellate::ToolDisplacedTessellateArgs{};
        if(!tool_tessellate::toolDisplacedTessellateParse(subcommand.second.count(), subcommand.second.argv(),
                                                          std::any_cast<tool_tessellate::ToolDisplacedTessellateArgs&>(args), parseError))
        {
          parseError << std::endl;
          parseResult = false;
        }

        tool_tessellate::toolDisplacedTessellateAddRequirements(meshopsContextConfig);
      }
      else if(subcommand.first == "remesh")
      {
        args = tool_remesh::ToolRemeshArgs{};
        if(!tool_remesh::toolRemeshParse(subcommand.second.count(), subcommand.second.argv(),
                                         std::any_cast<tool_remesh::ToolRemeshArgs&>(args), parseError))
        {
          parseError << std::endl;
          parseResult = false;
        }

        tool_remesh::toolRemeshAddRequirements(meshopsContextConfig);
      }
      else if(subcommand.first == "merge")
      {
        args = tool_merge::ToolMergeArgs{};
        if(!tool_merge::toolMergeParse(subcommand.second.count(), subcommand.second.argv(),
                                       std::any_cast<tool_merge::ToolMergeArgs&>(args), parseError))
        {
          parseError << std::endl;
          parseResult = false;
        }
      }
      else if(subcommand.first == "print")
      {
        // Make sure only the exe location exists in the argument array
        if(subcommand.second.count() > 1)
        {
          parseError << "Error: subcommand print takes no arguments" << std::endl;
          parseResult = false;
        }
      }
      else if(subcommand.first == "optimize")
      {
        args = tool_optimize::ToolOptimizeArgs{};
        if(!tool_optimize::toolOptimizeParse(subcommand.second.count(), subcommand.second.argv(),
                                             std::any_cast<tool_optimize::ToolOptimizeArgs&>(args), parseError))
        {
          parseError << std::endl;
          parseResult = false;
        }

        tool_optimize::toolOptimizeAddRequirements(meshopsContextConfig);
      }
    }

    if(!parseResult)
    {
      parser.printHelp();
      return EXIT_FAILURE;
    }

    micromesh_tool::ToolContext context(meshopsContextConfig);
    if(!context.valid())
    {
      return EXIT_FAILURE;
    }

    // Load the input scene
    auto                                       basePath = fs::path(inputFilename).parent_path();
    std::unique_ptr<micromesh_tool::ToolScene> scene;
    if(!inputFilename.empty())
    {
      scene = micromesh_tool::ToolScene::create(inputFilename);
      if(!scene)
      {
        LOGE("Error: Failed to load '%s'\n", inputFilename.c_str());
        return EXIT_FAILURE;
      }
      LOGI("Loaded %s (%s)\n", fs::path(inputFilename).filename().string().c_str(),
           micromesh_tool::ToolSceneStats(*scene).str().c_str());
    }
    else
    {
      // HACK: when generating geometry there is no input file, but subsequent
      // ops need to know where textures get generated.
      // TODO: add ToolTexture arrays to ToolScene to avoid the need for
      // basePath on the ToolScene or writing textures to disk unitl the
      // pipeline is done. See MICROSDK-382.
      basePath = fs::path(outputFilename).parent_path();
    }

    // May contain a copy of a scene before remeshing to be used by the baker.
    std::unique_ptr<micromesh_tool::ToolScene> bakerReference;

    // Execute all subcommands. This is done one at a time. After each, the
    // input variable is replaced with the output mesh from the subcommand. This
    // frees the original input and provides the new input for the next command.
    for(size_t i = 0; i < parser.subcommands().size(); ++i)
    {
      nvh::Stopwatch timer;
      auto&          subcommand = parser.subcommands()[i];
      auto&          anyArgs    = subcommandArgs[i];

      if(subcommand.first == "generate")
      {
        auto& args = std::any_cast<tool_generate::ToolGenerateArgs&>(anyArgs);
        if(!tool_generate::toolGenerate(context, args, scene))
        {
          LOGE("micromesh_tool: generate failure. Aborting.\n");
          return EXIT_FAILURE;
        }
      }
      else if(subcommand.first == "pretessellate")
      {
        auto& args = std::any_cast<tool_tessellate::ToolPreTessellateArgs&>(anyArgs);
        if(!tool_tessellate::toolPreTessellate(context, args, scene))
        {
          LOGE("micromesh_tool: pretessellate failure. Aborting.\n");
          return EXIT_FAILURE;
        }
      }
      else if(subcommand.first == "bake")
      {
        auto& args             = std::any_cast<tool_bake::ToolBakeArgs&>(anyArgs);
        args.outputTextureStem = fs::path(outputFilename).stem().string();

        // Support legacy --bary /absolute/path/to/file.bary
        auto outputBasePath = fs::path(outputFilename).parent_path();
        if(!args.baryFilename.empty() && !outputBasePath.empty())
        {
          args.baryFilename = fs::path(args.baryFilename).lexically_proximate(outputBasePath).string();
        }

        // Bake using the bakerReference from before remeshing if it exists, or
        // rely on the --high argument. If neither exist, the baker will bake
        // against the base scene.
        bool result = bakerReference ? tool_bake::toolBake(context, args, *bakerReference, scene) :
                                       tool_bake::toolBake(context, args, scene);
        if(!result)
        {
          LOGE("micromesh_tool: bake failure. Aborting.\n");
          return EXIT_FAILURE;
        }

        bakerReference.reset();
      }
      else if(subcommand.first == "displacedtessellate")
      {
        auto& args    = std::any_cast<tool_tessellate::ToolDisplacedTessellateArgs&>(anyArgs);
        args.basePath = fs::path(inputFilename).parent_path().string();
        if(!tool_tessellate::toolDisplacedTessellate(context, args, scene))
        {
          LOGE("micromesh_tool: displacedtessellate failure. Aborting.\n");
          return EXIT_FAILURE;
        }
      }
      else if(subcommand.first == "merge")
      {
        auto& args = std::any_cast<tool_merge::ToolMergeArgs&>(anyArgs);
        if(!tool_merge::toolMerge(args, scene))
        {
          LOGE("micromesh_tool: merge failure. Aborting.\n");
          return EXIT_FAILURE;
        }
      }
      else if(subcommand.first == "remesh")
      {
        // The last thing passed to the remesher is likely the high-res
        // reference mesh. If this is the last remesh command before a future
        // baker command that is also missing a --high argument, make a copy of
        // the scene now to be used then.
        for(size_t j = i + 1; j < parser.subcommands().size(); ++j)
        {
          if(parser.subcommands()[j].first == "remesh")
          {
            break;
          }
          else if(parser.subcommands()[j].first == "bake")
          {
            auto& nextBakeArgs = std::any_cast<tool_bake::ToolBakeArgs&>(subcommandArgs[j]);
            if(nextBakeArgs.highFilename.empty())
            {
              LOGI("Copying the scene before running the remesher, to be used by the next baker stage\n");
              bakerReference = micromesh_tool::ToolScene::create(scene);
              if(!bakerReference)
              {
                LOGE("Failed to duplicate scene before remeshing\n");
                return EXIT_FAILURE;
              }
            }
            break;
          }
        }

        auto& args = std::any_cast<tool_remesh::ToolRemeshArgs&>(anyArgs);
        if(!tool_remesh::toolRemesh(context, args, scene))
        {
          LOGE("micromesh_tool: remesh failure. Aborting.\n");
          return EXIT_FAILURE;
        }
      }
      else if(subcommand.first == "print")
      {
        micromesh_tool::sceneWriteDebug(*scene, std::cout);
      }
      else if(subcommand.first == "optimize")
      {
        auto& args = std::any_cast<tool_optimize::ToolOptimizeArgs&>(anyArgs);
        if(!tool_optimize::toolOptimize(context, args, scene))
        {
          LOGE("micromesh_tool: optimize failure. Aborting\n");
          return EXIT_FAILURE;
        }
      }
      LOGI("Finished %s in %.1fms (%s)\n", subcommand.first.c_str(), timer.elapsed(),
           micromesh_tool::ToolSceneStats(*scene).str().c_str());
    }

    // Save the result. Special case a hidden /dev/null if the user just wants
    // to "{print}" stats (it would normally fail due to permissions on the
    // gltf's /dev/null.bin)
    if(outputFilename != "/dev/null")
    {
      if(!scene->save(outputFilename))
      {
        LOGE("Error: failed to write %s\n", outputFilename.c_str());
        return EXIT_FAILURE;
      }
    }

    assert(!bakerReference);
    return EXIT_SUCCESS;
  }
  catch(const std::exception& e)
  {
    // This prevents exceptions including std::system_error (from fs::path)
    // and std::bad_array_new_length (from std::map) from escaping main().
    LOGE("micromesh_tool processing threw an exception! Additional information: %s\n", e.what());
    return EXIT_FAILURE;
  }
}
