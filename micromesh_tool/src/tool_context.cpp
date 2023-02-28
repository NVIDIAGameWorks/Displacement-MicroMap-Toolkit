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

#include <tool_context.hpp>

namespace micromesh_tool {

ToolContext::ToolContext(const meshops::ContextConfig& config)
{
  m_createResult = meshops::meshopsContextCreate(config, &m_meshopsContext);
  if(m_createResult != micromesh::Result::eSuccess)
  {
    LOGE("Error: failed to create meshops context\n");
    assert(!"meshopsContextCreate()");
  }
}
ToolContext::ToolContext(const meshops::ContextConfig& config, const meshops::ContextVK& sharedContextVK)
{
  m_createResult = meshops::meshopsContextCreateVK(config, sharedContextVK, &m_meshopsContext);
  if(m_createResult != micromesh::Result::eSuccess)
  {
    LOGE("Error: failed to create meshops context\n");
    assert(!"meshopsContextCreate()");
  }
}

}  // namespace micromesh_tool