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

#include "vulkan/vulkan_core.h"
#include <nvvk/commands_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>
#include <nvvk/buffers_vk.hpp>
#include <nvvk/debug_util_vk.hpp>
#include <nvvk/error_vk.hpp>
#include <glm/detail/type_half.hpp>
#include <nvmath/nvmath_types.h>

namespace microdisp {

using uint  = uint32_t;
using uvec2 = nvmath::vec2ui;
using uvec3 = nvmath::vec3ui;
using vec2  = nvmath::vec2f;
using ivec2 = nvmath::vec2i;
using mat4  = nvmath::mat4f;

struct RBuffer : nvvk::Buffer
{
  VkDescriptorBufferInfo info = {VK_NULL_HANDLE};
  VkDeviceAddress        addr = 0;
};

struct TempCommandsVK
{
};

struct ResourcesVK
{
  // Allocator must be passed when constructing to provide the reference m_allocator
  ResourcesVK(nvvk::ResourceAllocator& allocator, VkCommandBuffer cmd)
      : m_device(allocator.getDevice())
      , m_allocator(allocator)
      , m_cmd(cmd)
  {
  }

  // Care must be taken not to destroy this object while commands are pending
  ~ResourcesVK() {}

  VkCommandBuffer cmdBuffer() { return m_cmd; }

  RBuffer createBuffer(VkDeviceSize size, VkBufferUsageFlags flags, VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
  {
    RBuffer entry = {nullptr};

    if(size)
    {
      ((nvvk::Buffer&)entry) = m_allocator.createBuffer(
          size, flags | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, memFlags);
      entry.info.buffer = entry.buffer;
      entry.info.offset = 0;
      entry.info.range  = size;
      entry.addr        = nvvk::getBufferDeviceAddress(m_device, entry.buffer);
    }

    return entry;
  }

  void destroy(RBuffer& obj)
  {
    m_allocator.destroy(obj);
    obj.info = {nullptr};
    obj.addr = 0;
  }

  void simpleUploadBuffer(const RBuffer& dst, const void* src)
  {
    if(src && dst.info.range)
    {
      m_allocator.getStaging()->cmdToBuffer(m_cmd, dst.buffer, 0, dst.info.range, src);
    }
  }

  VkDevice                 m_device;
  nvvk::ResourceAllocator& m_allocator;
  VkCommandBuffer          m_cmd;
};

class float16_t
{
private:
  glm::detail::hdata h;

public:
  float16_t() {}
  float16_t(float f) { h = glm::detail::toFloat16(f); }

  operator float() const { return glm::detail::toFloat32(h); }
};

using f16vec2 = nvmath::vector2<float16_t>;
using f16vec4 = nvmath::vector4<float16_t>;
using u8vec2  = nvmath::vector2<uint8_t>;
using u8vec4  = nvmath::vector4<uint8_t>;
using u16vec2 = nvmath::vector2<uint16_t>;
using u16vec4 = nvmath::vector4<uint16_t>;

}  // namespace microdisp
