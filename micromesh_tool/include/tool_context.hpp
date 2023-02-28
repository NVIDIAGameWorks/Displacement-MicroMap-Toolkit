/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>
#include <meshops/meshops_operations.h>
#include <meshops/meshops_vk.h>

namespace micromesh_tool {

// Class to hold common resources to all tools. This avoids redundant
// construction costs for things such as the vulkan context.
class ToolContext
{
public:
  ToolContext(const meshops::ContextConfig& config);
  ToolContext(const meshops::ContextConfig& config, const meshops::ContextVK& sharedContextVK);
  ~ToolContext() { meshops::meshopsContextDestroy(m_meshopsContext); }

  bool valid() const { return m_createResult == micromesh::Result::eSuccess; }

  meshops::Context&   meshopsContext() { return m_meshopsContext; }
  meshops::ContextVK* meshopsContextVK() { return meshops::meshopsContextGetVK(m_meshopsContext); }

  // Disable copying
  ToolContext(const ToolContext& other) = delete;
  ToolContext& operator=(const ToolContext& other) = delete;

private:
  meshops::Context  m_meshopsContext;
  micromesh::Result m_createResult = micromesh::Result::eFailure;
};

}  // namespace micromesh_tool
