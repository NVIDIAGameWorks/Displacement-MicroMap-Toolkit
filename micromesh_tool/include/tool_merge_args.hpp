/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// Parser structures for the merge tool.
#pragma once

#include <string>
#include <vector>
#include <ostream>

namespace tool_merge {

struct ToolMergeArgs
{
  std::vector<std::string> inputs;  // List of additional files to merge into the existing ToolScene
};

// Parses a command-line into merge tool arguments. Returns true on success.
bool toolMergeParse(int argc, char** argv, ToolMergeArgs& args, std::ostream& os);

}  // namespace tool_merge