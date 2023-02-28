/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include "micromesh_compressed_vk.hpp"
#include "nvh/parallel_work.hpp"

#include "shaders/common.h"
#include "shaders/common_barymap.h"
#include "shaders/common_micromesh_compressed.h"

namespace microdisp {


static_assert(sizeof(MicromeshSubTri) == 16, "MicromeshSubTri not 16 bytes");

/*
* umajorUV
* 
*   w - e2 - v
*   |       /
*   |      /
*   e0    e1
*   |    /
*   |   /
*   |  /
*    u
*
* Elements are stored in this order, with n=5:
*   00 01 02 03 04
*   05 06 07 08
*   09 10 11
*   12 13
*   14
*/

inline uint umajorUV_toLinear(uint n, ivec2 uv)
{
  uint x      = uv.y;
  uint y      = uv.x;
  uint trinum = (y * (y + 1)) / 2;
  return y * (n + 1) - trinum + x;
}

inline uint32_t getLevel(uint32_t numSegments)
{
  uint32_t i = 0;
  while((1u << i) <= numSegments)
  {
    i++;
  }
  i--;

  assert((1u << i) == numSegments);

  return i;
}

inline uint32_t getFormatLevel(uint16_t blockFormat)
{
  return bary::baryBlockFormatDispC1GetSubdivLevel(static_cast<bary::BlockFormatDispC1>(blockFormat));
}

inline uint32_t getFormatIndex(uint16_t blockFormat)
{
  return getFormatLevel(blockFormat) - 3;
}

inline uint32_t packBits(uint32_t value, int offset, int width)
{
  assert(value <= ((1u << width) - 1));
  return (uint32_t)((value & ((1 << width) - 1)) << offset);
}
inline uint32_t unpackBits(uint32_t value, int offset, int width)
{
  return (uint32_t)((value >> offset) & ((1 << width) - 1));
}

nvmath::vec3f getMicroBarycentric(const MicromeshSubTri& micromesh, nvmath::vec2i primUV);

#if 0
nvmath::vec4f computeSphere(const MeshSet& meshSet, size_t baseTriangleIdx, const nvmath::vec3f barys[3], float minDisp, float maxDisp, bool directionBoundsAreUniform);

inline nvmath::vec4f computeSphere(const MicromeshSubTri& micro, const MeshSet& meshSet, size_t baseTriangleIdx, float minDisp, float maxDisp, bool directionBoundsAreUniform)
{
  nvmath::vec3f vertBarys[3] = {getMicroBarycentric(micro, nvmath::vec2i(0, 0)), getMicroBarycentric(micro, nvmath::vec2i(1, 0)),
                                getMicroBarycentric(micro, nvmath::vec2i(0, 1))};
  return computeSphere(meshSet, baseTriangleIdx, vertBarys, minDisp, maxDisp, directionBoundsAreUniform);
}

inline nvmath::vec4f computeSphere(const MicromeshBaseTri& micro, const MeshSet& meshSet, size_t baseTriangleIdx, float minDisp, float maxDisp, bool directionBoundsAreUniform)
{
  nvmath::vec3f vertBarys[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  return computeSphere(meshSet, baseTriangleIdx, vertBarys, minDisp, maxDisp, directionBoundsAreUniform);
}
#endif

struct MicromeshFormatInfo
{
  // clang-format off
    const static uint32_t num_formats = 3;
    const static uint32_t num_levels  = 6;
    struct BitValues {
      uint32_t v[num_levels];
    };

    BitValues widths[num_formats];
    BitValues starts[num_formats];

    BitValues corr_widths[num_formats];
    BitValues corr_starts[num_formats];

    MicromeshFormatInfo()
    {
      widths[MICRO_FORMAT_64T_512B]    = {11, 11, 11,  11,    0,    0};
      widths[MICRO_FORMAT_256T_1024B]  = {11, 11, 11,  10,    5,    0};
      widths[MICRO_FORMAT_1024T_1024B] = {11, 11,  8,   4,    2,    1};

      starts[MICRO_FORMAT_64T_512B]    = {0,  33, 66, 165,    0,    0};
      starts[MICRO_FORMAT_256T_1024B]  = {0,  33, 66, 165,  465,    0};
      starts[MICRO_FORMAT_1024T_1024B] = {0,  33, 66, 138,  258,  474};

      corr_widths[MICRO_FORMAT_64T_512B]    = {0,  0,  0,  0,  0,  0};
      corr_widths[MICRO_FORMAT_256T_1024B]  = {0,  0,  0,  1,  3,  0};
      corr_widths[MICRO_FORMAT_1024T_1024B] = {0,  0,  2,  3,  4,  4};

      // stored in descending levels
      // last 2 bits reserved: 62 minus 4 * widths
      corr_starts[MICRO_FORMAT_64T_512B]    = {0,  0,  0,  0,  0,  0};
      corr_starts[MICRO_FORMAT_256T_1024B]  = {0,  0,  0, 58, 46,  0};
      corr_starts[MICRO_FORMAT_1024T_1024B] = {0, 0,  54, 42, 26, 10};
    }

    uint32_t getBlockIndex(uint32_t format, uint32_t level, uint32_t levelIndex) const
    {
      return starts[format].v[level] + widths[format].v[level] * levelIndex;
    }

    uint32_t getWidth(uint32_t format, uint32_t level) const
    {
      return widths[format].v[level];
    }

    uint32_t getCorrIndex(uint32_t format, uint32_t level, uint32_t vertexType) const
    {
      return corr_starts[format].v[level] + vertexType * corr_widths[format].v[level];
    }

    uint32_t getCorrWidth(uint32_t format, uint32_t level) const
    {
      return corr_widths[format].v[level];
    }

    uint32_t getVertexType(baryutils::BaryWUV_uint16 coord) const
    {
      uint32_t interior = 0;
      uint32_t edge0    = 1;
      uint32_t edge1    = 2;
      uint32_t edge2    = 3;
      if(coord.w == 0)
        return edge1;
      else if(coord.u == 0)
        return edge2;
      else if(coord.v == 0)
        return edge0;
      else
        return interior;
    }
  // clang-format on
};

struct MicromeshCombinedData
{
  MicromeshData        bindingData;
  uint8_t              _pad[((sizeof(MicromeshData) + 127) & (~127)) - sizeof(MicromeshData) + 128];
  MicromeshFormatDescr formats[MICRO_MAX_FORMATS];

  void initFormats()
  {
    const MicromeshFormatInfo info;

    for(uint32_t f = 0; f < info.num_formats; f++)
    {
      MicromeshFormatDescr& formatConstants = formats[f];
      for(uint32_t i = 0; i < info.num_levels; i++)
      {
        formatConstants.width_start[i] =
            static_cast<uint16_t>(info.widths[f].v[i] | (info.starts[f].v[i] << MICRO_FORMATINFO_START_SHIFT));
      }
    }
  }

  MicromeshCombinedData()
  {
    memset(this, 0, sizeof(MicromeshCombinedData));

    initFormats();
  }

  void fillAddresses(const MicromeshSetCompressedVK& micro, const MicromeshSetCompressedVK::MeshData& meshData)
  {
    // per mesh
    bindingData.formats      = meshData.binding.addr + offsetof(MicromeshCombinedData, formats);
    bindingData.distances    = meshData.distances.addr;
    bindingData.mipDistances = meshData.mipDistances.addr;

    bindingData.subtriangles = meshData.subTriangles.addr;
    bindingData.subspheres   = meshData.subSpheres.addr;

    bindingData.basetriangles       = meshData.baseTriangles.addr;
    bindingData.basespheres         = meshData.baseSpheres.addr;
    bindingData.basetriangleMinMaxs = meshData.baseTriangleMinMaxs.addr;

    bindingData.attrNormals         = meshData.attrNormals.addr;
    bindingData.attrTriangleOffsets = meshData.attrTriangles.addr;

    // common
    bindingData.vertices        = micro.vertices.addr;
    bindingData.triangleIndices = micro.triangleIndices.addr;
    bindingData.descendInfos    = micro.descends.addr;

    // the missing `bindingData.umajor2bmap` is handled in `initBmapIndices`
  }
};

// align formats to 128 bytes (for perf)
static_assert(offsetof(MicromeshCombinedData, formats) % 128 == 0, "MicromeshFormatDescr formats alignment unexpected");

void initAttributes(MicromeshSetCompressedVK& micro, ResourcesVK& res, const bary::ContentView& bary, uint32_t maxSubdivLevel, uint32_t numThreads);

// MicroSplitParts is a utility class that stores
// various useful information that are attributes of
// the hierarchical encoding of vertices as well
// as splitting a base-triangle into sub-triangles and
// parts (part being the biggest unit a mesh-shader can
// work on).
class MicroSplitParts
{
public:
  struct MergePair
  {
    uint32_t a;
    uint32_t b;
  };

  MergePair partVertexMergeIndices[MICRO_PART_VERTICES_STRIDE];

  baryutils::BaryLevelsMap               map;
  const baryutils::BaryLevelsMap::Level& partLevel;

  // maybe rewrite procedurally using MICRO_MAX_SUBDIV and MICRO_FORMAT_MAX_SUBDIV and MICRO_PART_MAX_SUBDIV
  bary::BlockTriangle triLevel4to3[4];
  bary::BlockTriangle triLevel5to4[4];
  bary::BlockTriangle triLevel5to3[16];

  // warning sparsely filled, pointint to above
  bary::BlockTriangle* triLevelNtoN[MICRO_MAX_LEVELS][MICRO_MAX_LEVELS];
  uint32_t             numLevelNtoN[MICRO_MAX_LEVELS][MICRO_MAX_LEVELS];

private:
  void initMergeIndices();

  void initSplits()
  {
    memset(triLevelNtoN, 0, sizeof(triLevelNtoN));
    memset(numLevelNtoN, 0, sizeof(numLevelNtoN));

    triLevelNtoN[4][3] = triLevel4to3;
    triLevelNtoN[5][3] = triLevel5to3;
    triLevelNtoN[5][4] = triLevel5to4;

    numLevelNtoN[4][3] = NV_ARRAY_SIZE(triLevel4to3);
    numLevelNtoN[5][3] = NV_ARRAY_SIZE(triLevel5to3);
    numLevelNtoN[5][4] = NV_ARRAY_SIZE(triLevel5to4);

    static_assert(MICRO_MAX_SUBDIV - 1 == 4, "unexpected MAX_MICRO_SUBDIV");
    static_assert(MICRO_MAX_SUBDIV == 5, "unexpected MAX_MICRO_SUBDIV");

    bary::baryBlockFormatDispC1GetBlockTriangles(bary::BlockFormatDispC1::eR11_unorm_lvl3_pack512, 4, 4, triLevel4to3);
    bary::baryBlockFormatDispC1GetBlockTriangles(bary::BlockFormatDispC1::eR11_unorm_lvl4_pack1024, 5, 4, triLevel5to4);
    bary::baryBlockFormatDispC1GetBlockTriangles(bary::BlockFormatDispC1::eR11_unorm_lvl3_pack512, 5, 16, triLevel5to3);
  }

public:
  MicroSplitParts()
      : map(bary::ValueLayout::eTriangleBirdCurve, MICRO_MAX_SUBDIV)
      , partLevel(map.getLevel(MICRO_PART_MAX_SUBDIV))
  {
    initSplits();
    initMergeIndices();
  }

  void uploadTriangleIndices(nvvk::StagingMemoryManager* staging, VkCommandBuffer cmd, const RBuffer& triangleIndices, bool doPartFlip = true);
};

}  // namespace microdisp
