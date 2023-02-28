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

// Implementation of the merge tool.

#include "tool_merge_args.hpp"
#include <tool_scene.hpp>

namespace tool_merge {

// Runs the merge tool. Returns whether it succeeded.
bool toolMerge(const ToolMergeArgs& args, std::unique_ptr<micromesh_tool::ToolScene>& scene);

}  // namespace tool_merge