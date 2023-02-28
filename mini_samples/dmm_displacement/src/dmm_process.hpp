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

#include <vector>
#include <vulkan/vulkan_core.h>

#include "nvvk/context_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "vulkan_nv/vk_nv_micromesh.h"
#include "nvh/primitives.hpp"

// Setting for the terrain generator
struct Terrain
{
  float seed{0.0F};
  float freq{2.0F};
  float power{2.0F};
  int   octave{4};
};


class MicromapProcess
{

public:
  MicromapProcess(nvvk::Context* ctx, nvvk::ResourceAllocator* allocator);
  ~MicromapProcess();

  bool createMicromapData(VkCommandBuffer cmd, const nvh::PrimitiveMesh& mesh, uint16_t subdivLevel, const Terrain& terrain);
  void createMicromapBuffers(VkCommandBuffer cmd, const nvh::PrimitiveMesh& mesh, const nvmath::vec2f& biasScale);
  void cleanBuildData();

  const nvvk::Buffer&                    primitiveFlags() { return m_primitiveFlags; }
  const nvvk::Buffer&                    displacementDirections() { return m_displacementDirections; }
  const nvvk::Buffer&                    displacementBiasAndScale() { return m_displacementBiasAndScale; }
  const VkMicromapEXT&                   micromap() { return m_micromap; }
  const std::vector<VkMicromapUsageEXT>& usages() { return m_usages; }

private:
  struct MicromapData
  {
    std::vector<uint8_t>               values;
    std::vector<VkMicromapTriangleEXT> triangles;
    std::vector<VkMicromapUsageEXT>    usages;
  };

  // Raw values per triangles
  struct RawTriangle
  {
    uint32_t           subdivLevel{0};
    std::vector<float> values;
  };

  struct MicroDistances
  {
    std::vector<RawTriangle> rawTriangles;
  };


  bool        buildMicromap(VkCommandBuffer cmd);
  static void barrier(VkCommandBuffer cmd);
  static MicroDistances createDisplacements(const nvh::PrimitiveMesh& mesh, uint16_t subdivLevel, const Terrain& terrain);
  static MicromapData prepareData(const nvh::PrimitiveMesh& mesh, uint32_t subdivLevel, const MicroDistances& inputValues);

  VkDevice                 m_device;
  nvvk::ResourceAllocator* m_alloc;

  nvvk::Buffer m_inputData;
  nvvk::Buffer m_microData;
  nvvk::Buffer m_trianglesBuffer;
  nvvk::Buffer m_primitiveFlags;
  nvvk::Buffer m_displacementDirections;
  nvvk::Buffer m_displacementBiasAndScale;
  nvvk::Buffer m_scratchBuffer;

  VkMicromapEXT                   m_micromap{VK_NULL_HANDLE};
  std::vector<VkMicromapUsageEXT> m_usages;
};
