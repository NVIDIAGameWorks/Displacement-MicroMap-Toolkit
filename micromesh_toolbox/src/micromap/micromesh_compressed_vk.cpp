/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "micromesh_compressed_vk.hpp"
#include "bary/bary_types.h"
#include "micromesh_decoder_utils_vk.hpp"
#include "nvh/parallel_work.hpp"

namespace microdisp {

void MicromeshSetCompressedVK::initBasics(ResourcesVK& res, const bary::ContentView& bary, bool useBaseTriangles, bool useMips)
{
  memset(usedFormats, 0, sizeof(usedFormats));
  hasBaseTriangles = useBaseTriangles;

  assert(bary.basic.groupsCount == 1);
  const bary::BasicView& basic     = bary.basic;
  const bary::Group&     baryGroup = basic.groups[0];

  meshDatas.clear();
  meshDatas.resize(1);

  // allocation phase & smaller uploads
  nvvk::StagingMemoryManager* staging = res.m_allocator.getStaging();
  VkCommandBuffer             cmd     = res.cmdBuffer();

  {
    MeshData&                 meshData       = meshDatas[0];
    const bary::BasicView&    basic          = bary.basic;
    bary::Group               baryGroup      = basic.groups[0];
    bary::GroupHistogramRange baryHistoGroup = basic.groupHistogramRanges[0];

    meshData.combinedData = new MicromeshCombinedData;

    // init buffers
    meshData.binding = res.createBuffer(sizeof(MicromeshCombinedData),
                                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    for(uint32_t i = 0; i < baryHistoGroup.entryCount; i++)
    {
      usedFormats[basic.histogramEntries[i + baryHistoGroup.entryFirst].blockFormat] = true;
    }

    if(useBaseTriangles)
    {
      uint32_t baseTriangleCount = baryGroup.triangleCount;

      meshData.baseTriangles = res.createBuffer(sizeof(MicromeshBaseTri) * baseTriangleCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      meshData.baseSpheres = res.createBuffer(sizeof(nvmath::vec4f) * baseTriangleCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      meshData.microTriangleCount = baseTriangleCount;
    }
    else
    {
      uint32_t subTriangleCount =
          bary::baryHistogramGetBlockCount(baryHistoGroup.entryCount, basic.histogramEntries + baryHistoGroup.entryFirst,
                                           basic.valuesInfo->valueFormat);

      meshData.subTriangles = res.createBuffer(sizeof(MicromeshSubTri) * subTriangleCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      meshData.subSpheres = res.createBuffer(sizeof(nvmath::vec4f) * subTriangleCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
      meshData.microTriangleCount = subTriangleCount;
    }

    // add safety margin for out-of-bounds access
    meshData.distances =
        res.createBuffer(basic.valuesInfo->valueByteSize * baryGroup.valueCount + 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // only for visualization purposes, not required for actual rendering
    meshData.baseTriangleMinMaxs = res.createBuffer(basic.triangleMinMaxsInfo->elementByteSize * 2 * baryGroup.triangleCount,
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    staging->cmdToBuffer(cmd, meshData.baseTriangleMinMaxs.buffer, 0, meshData.baseTriangleMinMaxs.info.range,
                         basic.triangleMinMaxs + (basic.triangleMinMaxsInfo->elementByteSize * 2 * baryGroup.triangleFirst));

    if(useMips)
    {
      const bary::MiscView&      misc         = bary.misc;
      bary::GroupUncompressedMip baryMipGroup = misc.groupUncompressedMips[0];

      meshData.mipDistances = res.createBuffer(misc.uncompressedMipsInfo->elementByteSize * baryMipGroup.mipCount,
                                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

      staging->cmdToBuffer(cmd, meshData.mipDistances.buffer, 0, meshData.mipDistances.info.range,
                           misc.uncompressedMips + (misc.uncompressedMipsInfo->elementByteSize * baryMipGroup.mipFirst));
    }
  }

  // slightly bigger uploads
  {
    const MeshData& meshData = meshDatas[0];
    res.simpleUploadBuffer(meshData.distances, basic.values + (basic.valuesInfo->valueByteSize * baryGroup.valueFirst));
  }
}
void MicromeshSetCompressedVK::uploadMeshDatasBinding(nvvk::StagingMemoryManager* staging, VkCommandBuffer cmd)
{
  for(const auto& meshData : meshDatas)
  {
    meshData.combinedData->fillAddresses(*this, meshData);

    staging->cmdToBuffer(cmd, meshData.binding.info.buffer, meshData.binding.info.offset, meshData.binding.info.range,
                         meshData.combinedData);
  }
}

void MicromeshSetCompressedVK::initAttributeNormals(ResourcesVK& res, const bary::ContentView& bary, uint32_t numThreads)
{
  VkCommandBuffer cmd = res.cmdBuffer();

  assert(bary.basic.groupsCount == 1);
  {
    MeshData& meshData = meshDatas[0];

    const bary::BasicView basic = bary.basic;
    const bary::Group&    group = basic.groups[0];

    meshData.attrTriangles = res.createBuffer(sizeof(uint32_t) * group.triangleCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    meshData.attrNormals = res.createBuffer(basic.valuesInfo->valueByteSize * group.valueCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // create per-triangle offsets for attributes, for every mesh triangle
    {
      uint32_t* flatData = res.m_allocator.getStaging()->cmdToBufferT<uint32_t>(
          cmd, meshData.attrTriangles.buffer, meshData.attrTriangles.info.offset, meshData.attrTriangles.info.range);

      nvh::parallel_batches(
          group.triangleCount,
          [&](uint64_t baryLocalTriIdx) {
            const size_t          baryGlobalTriIdx = group.triangleFirst + baryLocalTriIdx;
            const bary::Triangle& baryPrim         = basic.triangles[baryGlobalTriIdx];
            assert(baryGlobalTriIdx < basic.trianglesCount);

            // Compute the flat triangle
            flatData[baryLocalTriIdx] = baryPrim.valuesOffset;
          },
          numThreads);
    }
  }

  // bigger uploads
  {
    const MeshData& meshData = meshDatas[0];

    {
      const bary::BasicView& basic = bary.basic;
      const bary::Group&     group = basic.groups[0];

      res.simpleUploadBuffer(meshData.attrNormals, basic.values + (basic.valuesInfo->valueByteSize * group.valueFirst));
    }
  }
}

void MicromeshSetCompressedVK::deinit(ResourcesVK& res)
{
  for(MeshData& mdata : meshDatas)
  {
    res.destroy(mdata.subTriangles);
    res.destroy(mdata.subSpheres);
    res.destroy(mdata.baseTriangles);
    res.destroy(mdata.baseSpheres);
    res.destroy(mdata.distances);
    res.destroy(mdata.mipDistances);
    res.destroy(mdata.baseTriangleMinMaxs);
    res.destroy(mdata.binding);
    res.destroy(mdata.attrNormals);
    res.destroy(mdata.attrTriangles);

    if(mdata.combinedData)
    {
      delete mdata.combinedData;
    }
  }

  res.destroy(umajor2bmap);
  res.destroy(triangleIndices);
  res.destroy(vertices);
  res.destroy(descends);

  meshDatas.clear();
}

}  // namespace microdisp
