/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "include/tool_merge_args.hpp"
#include "nvh/nvprint.hpp"

namespace tool_merge {

void printHelp(std::ostream& os)
{
  os << "merge: merges multiple glTF files into 1, with support for micromesh extensions." << std::endl;
  os << "Example usage: micromesh_tool --input input1.gltf --output output.gltf merge input2.gltf ... lastInput.gltf" << std::endl;
}

bool toolMergeParse(int argc, char** argv, ToolMergeArgs& args, std::ostream& os)
{
  for(int i = 0; i < argc; i++)
  {
    const char* arg = argv[i];
    if(strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0)
    {
      printHelp(os);
      // TODO: add a way to gracefully print subcommand help text
      return false;
    }
    else
    {
      args.inputs.push_back(std::string(arg));
    }
  }
  return true;
}

}  // namespace tool_merge