/*
* SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: LicenseRef-NvidiaProprietary
*
* NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
* property and proprietary rights in and to this material, related
* documentation and any modifications thereto. Any use, reproduction,
* disclosure or distribution of this material and related documentation
* without an express license agreement from NVIDIA CORPORATION or
* its affiliates is strictly prohibited.
*/

#include "barymap_vk.hpp"
#include "bary/bary_types.h"
#include "host_device.h"
#include "vulkan_mutex.h"

namespace microdisp {

struct PrimitiveCache
{
  typedef uint8_t IndexType;
  //  Utility class to generate the meshlets from triangle indices.
  //  It finds the unique vertex set used by a series of primitives.
  //  The cache is exhausted if either of the maximums is hit.
  //  The effective limits used with the cache must be < MAX.

  IndexType primitives[256][3];
  uint32_t  vertices[256];
  uint32_t  numPrims;
  uint32_t  numVertices;

  uint32_t maxVertexSize;
  uint32_t maxPrimitiveSize;

  bool empty() const { return numVertices == 0; }

  void reset()
  {
    numPrims    = 0;
    numVertices = 0;
    // reset
    memset(vertices, 0xFFFFFFFF, sizeof(vertices));
  }

  bool cannotInsert(uint32_t idxA, uint32_t idxB, uint32_t idxC) const
  {
    const uint32_t indices[3] = {idxA, idxB, idxC};

    uint32_t found = 0;
    for(uint32_t v = 0; v < numVertices; v++)
    {
      for(int i = 0; i < 3; i++)
      {
        uint32_t idx = indices[i];
        if(vertices[v] == idx)
        {
          found++;
        }
      }
    }
    // out of bounds
    return (numVertices + 3 - found) > maxVertexSize || (numPrims + 1) > maxPrimitiveSize;
  }

  void insert(uint32_t idxA, uint32_t idxB, uint32_t idxC)
  {
    const uint32_t indices[3] = {idxA, idxB, idxC};
    uint32_t       tri[3];

    for(int i = 0; i < 3; i++)
    {
      uint32_t idx   = indices[i];
      bool     found = false;
      for(uint32_t v = 0; v < numVertices; v++)
      {
        if(idx == vertices[v])
        {
          tri[i] = v;
          found  = true;
          break;
        }
      }
      if(!found)
      {
        vertices[numVertices] = idx;
        tri[i]                = numVertices;

        numVertices++;
      }
    }

    primitives[numPrims][0] = tri[0];
    primitives[numPrims][1] = tri[1];
    primitives[numPrims][2] = tri[2];
    numPrims++;
  }
};

void BaryLevelsMapVK::init(nvvk::ResourceAllocator& alloc, VkCommandBuffer cmd, const baryutils::BaryLevelsMap& baryMap)
{
  assert(baryMap.getLayout() != bary::ValueLayout::eUndefined);

  binding = RBuffer::create(alloc, sizeof(BaryMapData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  uint32_t numLevels = std::min(baryMap.getNumLevels(), uint32_t(MAX_BARYMAP_LEVELS));

  std::vector<BaryMapMeshlet> meshletHeaders;
  std::vector<uint32_t>       meshletData;

  levels.clear();
  levels.resize(MAX_BARYMAP_LEVELS * MAX_BARYMAP_TOPOS);

  auto addMeshlet = [&](const PrimitiveCache& cache, size_t begin, uint32_t subdLevel, uint32_t edgeBits) {
    BaryMapMeshlet header;
    header.numPrimitives = uint16_t(cache.numPrims);
    header.numVertices   = uint16_t(cache.numVertices);
    header.offsetPrims   = uint16_t(meshletData.size() - begin);
    assert(meshletData.size() - begin <= 0xFFFF);

    for(uint32_t p = 0; p < cache.numPrims; p++)
    {
      meshletData.push_back(cache.primitives[p][0] | (cache.primitives[p][1] << 8) | (cache.primitives[p][2] << 16));
    }

    header.offsetVertices = uint16_t((meshletData.size() - begin));
    assert(meshletData.size() - begin <= 0xFFFF);

    {
      // add vertex indices of current configuration
      for(uint32_t v = 0; v < cache.numVertices; v++)
      {
        const baryutils::BaryLevelsMap::Level& blevel = baryMap.getLevel(subdLevel);
        baryutils::BaryWUV_uint16              bcoord = blevel.coordinates[cache.vertices[v]];
        meshletData.push_back(bcoord.u | (bcoord.v << 8) | (cache.vertices[v] << 16));
      }
    }

    if(edgeBits == 0)
    {
      // generate vertexIndices for "uppper" target lod levels
      // we remove the edge decimation on lods for now, so only do this for edgeBits 0 configs

      uint32_t delta = 1;
      for(uint32_t delta = 1; subdLevel + delta < numLevels; delta++)
      {
        for(uint32_t v = 0; v < cache.numVertices; v++)
        {
          const baryutils::BaryLevelsMap::Level& blevel       = baryMap.getLevel(subdLevel);
          const baryutils::BaryLevelsMap::Level& blevelTarget = baryMap.getLevel(subdLevel + delta);

          // get current subdLevel coord
          baryutils::BaryWUV_uint16 bcoord = blevel.coordinates[cache.vertices[v]];
          // convert for upper target coord
          bcoord.u *= (1 << delta);
          bcoord.v *= (1 << delta);
          bcoord.w *= (1 << delta);
          uint32_t targetIndex = blevelTarget.getCoordIndex(bcoord);
          assert(targetIndex != ~0 && targetIndex <= 0xFFFF);

          meshletData.push_back(bcoord.u | (bcoord.v << 8) | (targetIndex << 16));
        }
      }
    }

    meshletHeaders.push_back(header);
  };

  size_t sz = 0;
  for(uint32_t lvl = 0; lvl < MAX_BARYMAP_LEVELS * MAX_BARYMAP_TOPOS; lvl++)
  {
    uint32_t subdLevel = lvl % MAX_BARYMAP_LEVELS;
    uint32_t edgeBits  = lvl / MAX_BARYMAP_LEVELS;

    if(subdLevel >= numLevels)
      continue;

    Level&                                 level  = levels[lvl];
    const baryutils::BaryLevelsMap::Level& blevel = baryMap.getLevel(subdLevel);

    level.firstHeader = meshletHeaders.size();
    level.firstData   = meshletData.size();

    PrimitiveCache cache;
    cache.maxPrimitiveSize = MAX_BARYMAP_PRIMITIVES;
    cache.maxVertexSize    = MAX_BARYMAP_VERTICES;
    cache.reset();

    size_t numTriangles = size_t(1) << (subdLevel * 2);

    // We keep degenerate triangles so that all lower subdlevels independent
    // of edgeBits output the same number of triangles
    // This is makes lod packing more predictable, as we only need to
    // account for subdiv level.

    std::vector<baryutils::BaryLevelsMap::Triangle> triangles = blevel.buildTrianglesWithCollapsedEdges(edgeBits, subdLevel < 3);

    for(uint32_t i = 0; i < uint32_t(triangles.size()); i++)
    {
      baryutils::BaryLevelsMap::Triangle tri = triangles[i];
      if(cache.cannotInsert(tri.a, tri.b, tri.c))
      {
        // finish old and reset
        addMeshlet(cache, level.firstData, subdLevel, edgeBits);
        cache.reset();
      }
      cache.insert(tri.a, tri.b, tri.c);
    }

    if(!cache.empty())
    {
      addMeshlet(cache, level.firstData, subdLevel, edgeBits);
    }

    level.headersCount = meshletHeaders.size() - level.firstHeader;
    level.dataCount    = meshletData.size() - level.firstData;

    if(level.headersCount * 32 > 0xFFFF)
    {
      printf("ERROR: exceeding max headers for task shader prefix sum\n");
      exit(-1);
    }

    level.coordsOffset = sz;
    sz += blevel.coordinates.size() * sizeof(uint32_t);
    sz                  = (sz + 3) & ~3;
    level.headersOffset = sz;
    sz += sizeof(BaryMapMeshlet) * level.headersCount;
    level.dataOffset = sz;
    sz += sizeof(uint32_t) * level.dataCount;
  }

  data = RBuffer::create(alloc, sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  nvvk::StagingMemoryManager* staging = alloc.getStaging();

  BaryMapData* map = staging->cmdToBufferT<BaryMapData>(cmd, binding.buffer, 0, binding.info.range);
  uint8_t*     raw = staging->cmdToBufferT<uint8_t>(cmd, data.buffer, 0, data.info.range);

  for(uint32_t lvl = 0; lvl < MAX_BARYMAP_LEVELS * MAX_BARYMAP_TOPOS; lvl++)
  {
    uint32_t subdLevel = lvl % MAX_BARYMAP_LEVELS;
    uint32_t edgeBits  = lvl / MAX_BARYMAP_LEVELS;

    if(subdLevel >= numLevels)
      continue;

    const baryutils::BaryLevelsMap::Level& blevel = baryMap.getLevel(subdLevel);
    map->levelsUni[lvl].triangleCount             = uint32_t(blevel.triangles.size());
    map->levelsUni[lvl].meshletCount              = uint32_t(levels[lvl].headersCount);
    map->levelsUni[lvl].coordsAddress             = data.addr + levels[lvl].coordsOffset;
    map->levelsUni[lvl].meshletDataAddress        = data.addr + levels[lvl].dataOffset;
    map->levelsUni[lvl].meshletHeadersAddress     = data.addr + levels[lvl].headersOffset;
    {
      uint32_t* out = reinterpret_cast<uint32_t*>(raw + levels[lvl].coordsOffset);
      for(size_t i = 0; i < blevel.coordinates.size(); i++)
      {
        out[i] = blevel.coordinates[i].w | (blevel.coordinates[i].u << 8) | (blevel.coordinates[i].v << 16);
      }
    }
    if(levels[lvl].dataCount)
    {
      uint32_t* out = reinterpret_cast<uint32_t*>(raw + levels[lvl].dataOffset);
      memcpy(out, &meshletData[levels[lvl].firstData], levels[lvl].dataCount * sizeof(uint32_t));
    }
    if(levels[lvl].headersCount)
    {
      BaryMapMeshlet* out = reinterpret_cast<BaryMapMeshlet*>(raw + levels[lvl].headersOffset);
      memcpy(out, &meshletHeaders[levels[lvl].firstHeader], levels[lvl].headersCount * sizeof(BaryMapMeshlet));
    }
  }
  map->levelsAddress = binding.addr + offsetof(BaryMapData, levelsUni);
}

void BaryLevelsMapVK::deinit(nvvk::ResourceAllocator& alloc)
{

  RBuffer::destroy(alloc, binding);
  RBuffer::destroy(alloc, data);
}

}  // namespace microdisp
