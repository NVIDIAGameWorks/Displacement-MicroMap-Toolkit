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
#define _USE_MATH_DEFINES
#include <map>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/noise.hpp>  // Perlin noise

#include "vulkan_nv/vk_nv_micromesh.h"  // prototype

#include "nvmath/nvmath.h"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/error_vk.hpp"
#include "nvh/parallel_work.hpp"

#include "meshops/meshops_types.h"
#include "micromesh/micromesh_displacement_compression.h"
#include "micromesh/micromesh_format_types.h"
#include "micromesh/micromesh_types.h"
#include "micromesh/micromesh_utils.h"

#include "dmm_process.hpp"
#include "nesting_scoped_timer.hpp"

MicromapProcess::MicromapProcess(nvvk::Context* ctx, nvvk::ResourceAllocator* allocator)
    : m_alloc(allocator)
    , m_device(ctx->m_device)
{
}

MicromapProcess::~MicromapProcess()
{
  m_alloc->destroy(m_inputData);
  m_alloc->destroy(m_microData);
  m_alloc->destroy(m_trianglesBuffer);
  m_alloc->destroy(m_primitiveFlags);
  m_alloc->destroy(m_displacementDirections);
  m_alloc->destroy(m_displacementBiasAndScale);
  m_alloc->destroy(m_scratchBuffer);
  vkDestroyMicromapEXT(m_device, m_micromap, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Create the data for displacement
// - Get a vector of displacement values per triangle
// - Pack the data to 11 bit (64_TRIANGLES_64_BYTES format)
// - Get the usage
// - Create the vector of VkMicromapTriangleEXT
bool MicromapProcess::createMicromapData(VkCommandBuffer cmd, const nvh::PrimitiveMesh& mesh, uint16_t subdivLevel, const Terrain& terrain)
{
  NestingScopedTimer stimer("Create Micromap Data");

  vkDestroyMicromapEXT(m_device, m_micromap, nullptr);
  m_alloc->destroy(m_scratchBuffer);
  m_alloc->destroy(m_inputData);
  m_alloc->destroy(m_microData);
  m_alloc->destroy(m_trianglesBuffer);

  // Get an array of displacement per triangle
  MicroDistances micro_dist = createDisplacements(mesh, subdivLevel, terrain);

  {
    // Compress all the data using SDK functions

    MicromapProcess::MicromapData outdata = prepareData(mesh, subdivLevel, micro_dist);
    m_usages                              = outdata.usages;

    m_inputData       = m_alloc->createBuffer(cmd, outdata.values,
                                              VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_trianglesBuffer = m_alloc->createBuffer(cmd, outdata.triangles,
                                              VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT
                                                  | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  }

  barrier(cmd);

  buildMicromap(cmd);

  return true;
}

//--------------------------------------------------------------------------------------------------
// Building the micromap using: triangle data, input data (values), usage
//
bool MicromapProcess::buildMicromap(VkCommandBuffer cmd)
{
  NestingScopedTimer stimer("Build Micromap");

  // Find the size required
  VkMicromapBuildSizesInfoEXT size_info{VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT};
  VkMicromapBuildInfoEXT      build_info{VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT};
  build_info.mode             = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
  build_info.usageCountsCount = static_cast<uint32_t>(m_usages.size());
  build_info.pUsageCounts     = m_usages.data();
  build_info.type             = VK_MICROMAP_TYPE_DISPLACEMENT_MICROMAP_NV;  // Displacement
  vkGetMicromapBuildSizesEXT(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info, &size_info);
  assert(size_info.micromapSize && "sizeInfo.micromeshSize was zero");

  // create micromeshData buffer
  m_microData = m_alloc->createBuffer(size_info.micromapSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                                  | VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT);

  uint64_t scratch_size = std::max(size_info.buildScratchSize, static_cast<VkDeviceSize>(4));
  m_scratchBuffer =
      m_alloc->createBuffer(scratch_size, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT);

  // Create micromesh
  VkMicromapCreateInfoEXT mm_create_info{VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT};
  mm_create_info.buffer = m_microData.buffer;
  mm_create_info.size   = size_info.micromapSize;
  mm_create_info.type   = VK_MICROMAP_TYPE_DISPLACEMENT_MICROMAP_NV;
  NVVK_CHECK(vkCreateMicromapEXT(m_device, &mm_create_info, nullptr, &m_micromap));

  {
    // Fill in the pointers we didn't have at size query
    build_info.dstMicromap                 = m_micromap;
    build_info.scratchData.deviceAddress   = nvvk::getBufferDeviceAddress(m_device, m_scratchBuffer.buffer);
    build_info.data.deviceAddress          = nvvk::getBufferDeviceAddress(m_device, m_inputData.buffer);
    build_info.triangleArray.deviceAddress = nvvk::getBufferDeviceAddress(m_device, m_trianglesBuffer.buffer);
    build_info.triangleArrayStride         = sizeof(VkMicromapTriangleEXT);
    vkCmdBuildMicromapsEXT(cmd, 1, &build_info);
  }
  barrier(cmd);

  return true;
}

//--------------------------------------------------------------------------------------------------
// This can be called when the Micromap has been build
//
void MicromapProcess::cleanBuildData()
{
  m_alloc->destroy(m_scratchBuffer);
  m_alloc->destroy(m_inputData);
  m_alloc->destroy(m_trianglesBuffer);
}

//--------------------------------------------------------------------------------------------------
//
//
void MicromapProcess::createMicromapBuffers(VkCommandBuffer cmd, const nvh::PrimitiveMesh& mesh, const nvmath::vec2f& biasScale)
{
  m_alloc->destroy(m_primitiveFlags);
  m_alloc->destroy(m_displacementDirections);
  m_alloc->destroy(m_displacementBiasAndScale);

  auto num_tri = static_cast<uint32_t>(mesh.indices.size() / 3U);

  // Direction vectors
  {
    // We are taking the normal of the triangle as direction vectors
    using f16vec4 = glm::vec<4, glm::detail::hdata, glm::defaultp>;
    std::vector<f16vec4> temp;
    temp.reserve(mesh.vertices.size());
    for(const auto& v : mesh.vertices)
    {
      temp.emplace_back(glm::detail::toFloat16(v.n.x), glm::detail::toFloat16(v.n.y), glm::detail::toFloat16(v.n.z),
                        glm::detail::toFloat16(0.0F));  // convert to a vector of half float
    }
    m_displacementDirections =
        m_alloc->createBuffer(cmd, temp, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  }


  // Direction Bounds
  {
    // Making the bias-scale uniform across all triangle vertices
    std::vector<nvmath::vec2f> bias_scale;
    bias_scale.reserve(num_tri * 3ULL);
    for(uint32_t i = 0; i < num_tri; i++)
    {
      bias_scale.emplace_back(biasScale);
      bias_scale.emplace_back(biasScale);
      bias_scale.emplace_back(biasScale);
    }
    m_displacementBiasAndScale =
        m_alloc->createBuffer(cmd, bias_scale, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  }


  // // Primitive flags [unused for now, since all triangles have the same subdivision level]
  {
    std::vector<uint8_t> primitive_flags;
    if(!primitive_flags.empty())
    {
      m_primitiveFlags = m_alloc->createBuffer(cmd, primitive_flags,
                                               VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT
                                                   | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    }
  }


  barrier(cmd);
}

//--------------------------------------------------------------------------------------------------
// Make sure all the data are ready before building the micromap
void MicromapProcess::barrier(VkCommandBuffer cmd)
{
  // barrier for upload finish
  VkMemoryBarrier2 mem_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,         nullptr,
                               VK_PIPELINE_STAGE_2_TRANSFER_BIT,           VK_ACCESS_2_TRANSFER_WRITE_BIT,
                               VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT, VK_ACCESS_2_MICROMAP_READ_BIT_EXT};
  VkDependencyInfo dep_info{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
  dep_info.memoryBarrierCount = 1;
  dep_info.pMemoryBarriers    = &mem_barrier;
  vkCmdPipelineBarrier2(cmd, &dep_info);
}

// Interpolate the 3 values with bary; using auto as a templated function
inline auto getInterpolated = [](const auto& v0, const auto& v1, const auto& v2, const auto& bary) {
  return v0 * bary[0] + v1 * bary[1] + v2 * bary[2];
};

// Set the float value on 11 bit for packing
inline auto floatToR11 = [](float val) { return static_cast<uint16_t>(val * ((1 << 11) - 1)); };  // Mult by 2047


//--------------------------------------------------------------------------------------------------
// Get the displacement values per triangle; UV distance from [0.5,0.5]
// -
MicromapProcess::MicroDistances MicromapProcess::createDisplacements(const nvh::PrimitiveMesh& mesh, uint16_t subdivLevel, const Terrain& terrain)
{
  NestingScopedTimer stimer("Create Displacements");

  MicroDistances displacements;  // Result: displacement values for all triangles


  auto num_tri = static_cast<uint32_t>(mesh.indices.size() / 3);
  displacements.rawTriangles.resize(num_tri);

  // Find the distances in parallel
  // Faster than : for(size_t tri_index = 0; tri_index < num_tri; tri_index++)
  nvh::parallel_batches<32>(
      num_tri,
      [&](uint64_t tri_index) {
        // Retrieve the UV of the triangle
        nvmath::vec2f t0 = mesh.vertices[mesh.indices[tri_index * 3 + 0]].t;
        nvmath::vec2f t1 = mesh.vertices[mesh.indices[tri_index * 3 + 1]].t;
        nvmath::vec2f t2 = mesh.vertices[mesh.indices[tri_index * 3 + 2]].t;

        // Working on this triangle
        uint32_t     num_sub_vertices = micromesh::subdivLevelGetVertexCount(subdivLevel);
        RawTriangle& triangle         = displacements.rawTriangles[tri_index];
        triangle.values.resize(num_sub_vertices);
        triangle.subdivLevel = subdivLevel;

        // We are getting the data in UMajor, not Bird Curve. The information
        // in Bird Curve for the GPU will be done in `compressData()`
        for(uint32_t index = 0; index < num_sub_vertices; index++)
        {
          micromesh::BaryUV_uint16 coord       = micromesh::umajorLinearToUV(index, subdivLevel);
          micromesh::BaryWUV_float coord_float = micromesh::baryUVtoWUV_float(coord, subdivLevel);

          nvmath::vec2f uv = getInterpolated(t0, t1, t2, coord_float);

          // Simple perlin noise
          float v     = 0.0F;
          float scale = terrain.power;
          float freq  = terrain.freq;
          for(int oct = 0; oct < terrain.octave; oct++)
          {
            v += glm::perlin(glm::vec3(uv.x, uv.y, terrain.seed) * freq) / scale;
            freq *= 2.0F;            // Double the frequency
            scale *= terrain.power;  // Next power of b
          }

          // Adjusting and setting the value
          float disp_value       = nvmath::clamp((1.0F + v) * 0.5F, 0.0F, 1.0F);
          triangle.values[index] = disp_value;
        }
      },
      std::thread::hardware_concurrency());

  return displacements;
}

//--------------------------------------------------------------------------------------------------
// Prepare the data to be uploaded on the GPU.
// Depending on the subdivision level, the SDK will encode the raw data to the appropriate
// encoding.
//
MicromapProcess::MicromapData MicromapProcess::prepareData(const nvh::PrimitiveMesh& mesh, uint32_t subdivLevel, const MicroDistances& inputValues)
{
  NestingScopedTimer stimer("Compress Data");

  MicromapData result_data{};

  // Set the layout for the input into the SDK
  micromesh::MicromapLayout layout{};
  micromesh::micromeshLayoutInitStandard(&layout, micromesh::StandardLayoutType::eUmajor);

  // The compression library needs information about the topology of the mesh.
  micromesh::ScopedOpContext         ctx{};
  [[maybe_unused]] micromesh::Result result{};
  meshops::MeshTopologyData          topodata{};
  result = topodata.buildFindingWatertightIndices(ctx, mesh.indices.size(), mesh.indices.data(), mesh.vertices.size(),
                                                  reinterpret_cast<const micromesh::Vector_float_3*>(&mesh.vertices[0].p),
                                                  sizeof(nvh::PrimitiveVertex));
  assert(result == micromesh::Result::eSuccess);


  // Preparing the data to be digested by the compressor.
  // All data must be unorm11 stored on uint16_t
  // We also store the subdivision level per triangle (this could be different)
  // and the `offset` to know where each triangle starts
  std::vector<uint16_t> data16;
  std::vector<uint16_t> triangle_subdiv_levels;
  std::vector<uint32_t> triangle_value_index_offsets;
  for(const auto& triangle : inputValues.rawTriangles)
  {
    auto offset = static_cast<uint32_t>(data16.size());
    for(const auto& disp_value : triangle.values)
    {
      data16.push_back(floatToR11(disp_value));  // eR11_unorm_pack16
    }
    triangle_subdiv_levels.push_back(static_cast<uint16_t>(subdivLevel));
    triangle_value_index_offsets.push_back(offset);
  }

  // Setting up the uncompressed data
  micromesh::Micromap uncompressed_map{};
  micromesh::arraySetFormatTypeDataVec(uncompressed_map.values, data16);
  uncompressed_map.values.format = micromesh::Format::eR11_unorm_pack16;
  micromesh::arraySetFormatTypeDataVec(uncompressed_map.triangleSubdivLevels, triangle_subdiv_levels);
  micromesh::arraySetFormatTypeDataVec(uncompressed_map.triangleValueIndexOffsets, triangle_value_index_offsets);
  uncompressed_map.frequency      = micromesh::Frequency::ePerMicroVertex;
  uncompressed_map.minSubdivLevel = subdivLevel;
  uncompressed_map.maxSubdivLevel = subdivLevel;
  uncompressed_map.layout         = layout;


  // run compression begin function
  micromesh::MicromapCompressed              compressed_map{};
  micromesh::OpCompressDisplacement_settings settings{};
  micromesh::OpCompressDisplacement_input    input_decompressed{};
  micromesh::OpCompressDisplacement_output   output_compressed{};

  // we actually only have one family format, so this currently always is micromesh::Format::eDispC1_r11_unorm_block
  input_decompressed.compressedFormatFamily = micromesh::Format::eDispC1_r11_unorm_block;
  // the uncompressed unorm11 input data
  input_decompressed.data = &uncompressed_map;
  // the micromesh::MeshTopology used to ensure watertightness
  input_decompressed.topology = &topodata.topology;
  // the output micromap: this struct has some other optional outputs as well.
  output_compressed.compressed = &compressed_map;


  // Overriding settings
  settings.enabledBlockFormatBits = 0;
  settings.enabledBlockFormatBits |= 1U << (uint32_t)micromesh::BlockFormatDispC1::eR11_unorm_lvl3_pack512;
  settings.enabledBlockFormatBits |= 1U << (uint32_t)micromesh::BlockFormatDispC1::eR11_unorm_lvl4_pack1024;
  settings.enabledBlockFormatBits |= 1U << (uint32_t)micromesh::BlockFormatDispC1::eR11_unorm_lvl5_pack1024;


  result = micromesh::micromeshOpCompressDisplacementBegin(ctx, &settings, &input_decompressed, &output_compressed);
  assert(result == micromesh::Result::eSuccess);

  // resize number of triangles and values in `baryCompressed`
  result_data.triangles.resize(compressed_map.triangleBlockFormats.count);
  result_data.values.resize(compressed_map.values.count);

  stimer.print("Size needed: %lld \n", result_data.values.size());


  // setup pointers / adjust stride etc. for the compressedMap, which is to be passed into the end function
  compressed_map.values.data                         = result_data.values.data();
  compressed_map.triangleBlockFormats.data           = &result_data.triangles[0].format;
  compressed_map.triangleBlockFormats.byteStride     = sizeof(VkMicromapTriangleEXT);
  compressed_map.triangleSubdivLevels.data           = &result_data.triangles[0].subdivisionLevel;
  compressed_map.triangleSubdivLevels.byteStride     = sizeof(VkMicromapTriangleEXT);
  compressed_map.triangleValueByteOffsets.data       = &result_data.triangles[0].dataOffset;
  compressed_map.triangleValueByteOffsets.byteStride = sizeof(VkMicromapTriangleEXT);

  result = micromesh::micromeshOpCompressDisplacementEnd(ctx, &output_compressed);
  assert(result == micromesh::Result::eSuccess);


  // Create a histogram listing how many times each(compression format, subdivision level) pair
  // appears in the compressed data. The extension uses this to size its in-memory structures.
  std::map<std::pair<uint32_t, uint32_t>, uint32_t> histogram;
  for(auto& t : result_data.triangles)
  {
    ++histogram[std::make_pair(t.format, t.subdivisionLevel)];
  }
  for(auto& h : histogram)
  {
    VkMicromapUsageEXT u{};
    u.count            = h.second;
    u.format           = h.first.first;
    u.subdivisionLevel = h.first.second;
    result_data.usages.emplace_back(u);
  }


  return result_data;
}
