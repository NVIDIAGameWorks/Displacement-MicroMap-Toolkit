/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "micromesh_decoder_subtri_vk.hpp"
#include "micromesh_decoder_utils_vk.hpp"

namespace microdisp {

void MicromeshSubTriangleDecoderVK::init(ResourcesVK&             res,
                                         const bary::ContentView& bary,
                                         uint8_t*                 decimateEdgeFlags,
                                         uint32_t                 maxSubdivLevel,
                                         bool                     useBaseTriangles,
                                         bool                     withAttributes,
                                         uint32_t                 numThreads)
{
  // check support for this renderer
  {
    uint32_t minSubdivLevel;
    uint32_t maxSubdivLevel;
    bary::baryBasicViewGetMinMaxSubdivLevels(&bary.basic, &minSubdivLevel, &maxSubdivLevel);

    if(maxSubdivLevel > 5)
    {
      // sub triangle renderer is able to achieve higher subdiv levels
      useBaseTriangles = false;
    }
  }

  m_micro.initBasics(res, bary, useBaseTriangles, false);
  if(withAttributes)
  {
    initAttributes(m_micro, res, bary, maxSubdivLevel, numThreads);
  }

  // common look-up tables independent of data

  m_micro.descends = res.createBuffer(sizeof(MicromeshSTriDescend) * MICRO_STRI_DESCENDS_COUNT, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  m_micro.triangleIndices =
      res.createBuffer(sizeof(uint32_t) * (MICRO_MESHLET_PRIMS * MICRO_MESHLET_TOPOS), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  // one set of vertices for each partID config, plus zeroed dummy to allow safe out of bounds access
  m_micro.vertices = res.createBuffer(sizeof(MicromeshSTriVertex) * MICRO_STRI_VTX_COUNT, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  nvvk::StagingMemoryManager* staging = res.m_allocator.getStaging();
  VkCommandBuffer             cmd     = res.cmdBuffer();
  m_micro.uploadMeshDatasBinding(staging, cmd);

  {
    if(useBaseTriangles)
    {
      uploadMicroBaseTriangles(staging, cmd, bary, decimateEdgeFlags, maxSubdivLevel, numThreads);
    }
    else
    {
      uploadMicroSubTriangles(staging, cmd, bary, decimateEdgeFlags, maxSubdivLevel, numThreads);
    }
  }

  {
    MicroSplitParts splits;

    // setup indices
    splits.uploadTriangleIndices(staging, cmd, m_micro.triangleIndices, false);

    // setup vertices
    uploadVertices(staging, cmd, splits);

    // setup descend info
    uploadDescends(staging, cmd, splits);
  }
}

void MicromeshSubTriangleDecoderVK::uploadMicroSubTriangles(nvvk::StagingMemoryManager* staging,
                                                            VkCommandBuffer             cmd,
                                                            const bary::ContentView&    bary,
                                                            uint8_t*                    decimateEdgeFlags,
                                                            uint32_t                    maxSubdivLevel,
                                                            uint32_t                    numThreads)
{
  assert(m_micro.meshDatas.size() == 1);
  const MicromeshSetCompressedVK::MeshData& meshData = m_micro.meshDatas[0];

  assert(bary.basic.groupsCount == 1);
  const bary::BasicView& basic          = bary.basic;
  const bary::Group&     baryGroup      = basic.groups[0];
  const size_t           minMaxByteSize = basic.triangleMinMaxsInfo->elementByteSize * 2;

  MicromeshSubTri* subTriData =
      staging->cmdToBufferT<MicromeshSubTri>(cmd, meshData.subTriangles.buffer, meshData.subTriangles.info.offset,
                                             meshData.subTriangles.info.range);

  nvmath::vec4f* sphereData =
      staging->cmdToBufferT<nvmath::vec4f>(cmd, meshData.subSpheres.buffer, meshData.subSpheres.info.offset,
                                           meshData.subSpheres.info.range);

  baryutils::BarySplitTable splitTable;
  splitTable.init(bary::Format::eDispC1_r11_unorm_block, maxSubdivLevel);

  struct SubRange
  {
    uint32_t first;
    uint32_t count;
  };

  std::vector<SubRange> subRanges(baryGroup.triangleCount);

  // compute running offsets for sub triangles
  {
    uint32_t subOffset = 0;
    for(uint32_t i = 0; i < baryGroup.triangleCount; i++)
    {
      const bary::Triangle& tri      = basic.triangles[baryGroup.triangleFirst + i];
      uint32_t              subCount = bary::baryBlockFormatDispC1GetBlockCount(tri.blockFormatDispC1, tri.subdivLevel);
      subRanges[i].first             = subOffset;
      subRanges[i].count             = subCount;
      subOffset += subCount;
    }
  }

  nvh::parallel_batches(
      baryGroup.triangleCount,
      [&](uint64_t baryLocalTriIdx) {
        const size_t          meshGlobalTriIdx = baryLocalTriIdx;
        const size_t          baryGlobalTriIdx = baryGroup.triangleFirst + baryLocalTriIdx;
        const bary::Triangle& baseTri          = basic.triangles[baryGlobalTriIdx];
        const SubRange&       baseSubRange     = subRanges[baryLocalTriIdx];
        const uint16_t* baseMinMaxs = reinterpret_cast<const uint16_t*>(basic.triangleMinMaxs + baryGlobalTriIdx * minMaxByteSize);

        float valueBias  = baryGroup.floatBias.r;
        float valueScale = baryGroup.floatScale.r;

        const baryutils::BarySplitTable::Entry& splitConfig = splitTable.get(baseTri.blockFormatDispC1, baseTri.subdivLevel);

        for(uint32_t s = 0; s < baseSubRange.count; s++)
        {
          uint32_t subMeshIdx = s + baseSubRange.first;

          const bary::BlockTriangle& subSplit = splitConfig.tris[s];

          uint32_t formatIndex = getFormatIndex(baseTri.blockFormat);

          MicromeshSubTri& micro  = subTriData[subMeshIdx];
          nvmath::vec4f&   sphere = sphereData[subMeshIdx];

          const int32_t upperValueBound = 0x7FF;
          const int32_t upperCullBound  = MICRO_SUB_CULLDIST_MASK;

          // FIXME no longer per-sub-triangle minmaxs in default files
          float minDisp = float(baseMinMaxs[0]) / float(upperValueBound) * valueScale + valueBias;
          float maxDisp = float(baseMinMaxs[1]) / float(upperValueBound) * valueScale + valueBias;

          //float   delta = !mesh.directionBoundsAreUniform && meshSet.attributes.directionBounds.empty() ? 0.0f : -0.5f;
          //float   absDist  = std::max(std::abs(minDisp + delta), std::abs(maxDisp + delta));
          //int32_t cullDist = int32_t(((absDist - valueBias) / valueScale) * float(upperCullBound));
          //cullDist         = std::min(std::max(cullDist, 1), upperCullBound);

          const uint32_t baseTopo =
              (decimateEdgeFlags ? bary::baryBlockTriangleBaseToLocalFlags(&subSplit, decimateEdgeFlags[meshGlobalTriIdx]) : 0);

          micro.dataOffset = static_cast<uint32_t>((size_t(baseTri.valuesOffset) + subSplit.blockByteOffset) / sizeof(uint32_t));

          micro.baseOffset      = {uint16_t(subSplit.vertices[0].u), uint16_t(subSplit.vertices[0].v)};
          micro.baseTriangleIdx = static_cast<uint32_t>(baryLocalTriIdx);
          micro.packedBits      = 0;
          int32_t mapOffset{0};  // An unsigned integer that will be added to each of the mapIndices values.

          micro.packedBits |= packBits(baseTri.subdivLevel, MICRO_SUB_LVL_SHIFT, MICRO_SUB_LVL_WIDTH);
          micro.packedBits |= packBits(baseTopo, MICRO_SUB_TOPO_SHIFT, MICRO_SUB_TOPO_WIDTH);
          micro.packedBits |= packBits(formatIndex, MICRO_SUB_FMT_SHIFT, MICRO_SUB_FMT_WIDTH);
          micro.packedBits |= packBits(subSplit.signBits, MICRO_SUB_SIGN_SHIFT, MICRO_SUB_SIGN_WIDTH);
          micro.packedBits |= subSplit.flipped ? MICRO_SUB_FLIP : 0;
          //micro.packedBits |= packBits(cullDist, MICRO_SUB_CULLDIST_SHIFT, MICRO_SUB_CULLDIST_WIDTH);

          //sphere = computeSphere(micro, meshSet, meshGlobalTriIdx, minDisp, maxDisp, mesh.directionBoundsAreUniform);
        }
      },
      numThreads);
}

void MicromeshSubTriangleDecoderVK::uploadMicroBaseTriangles(nvvk::StagingMemoryManager* staging,
                                                             VkCommandBuffer             cmd,
                                                             const bary::ContentView&    bary,
                                                             uint8_t*                    decimateEdgeFlags,
                                                             uint32_t                    maxSubdivLevel,
                                                             uint32_t                    numThreads)
{
  assert(m_micro.meshDatas.size() == 1);
  const MicromeshSetCompressedVK::MeshData& meshData = m_micro.meshDatas[0];

  assert(bary.basic.groupsCount == 1);
  const bary::BasicView& basic          = bary.basic;
  const bary::Group&     baryGroup      = basic.groups[0];
  const size_t           minMaxByteSize = basic.triangleMinMaxsInfo->elementByteSize * 2;

  MicromeshBaseTri* baseTriData =
      staging->cmdToBufferT<MicromeshBaseTri>(cmd, meshData.baseTriangles.buffer, meshData.baseTriangles.info.offset,
                                              meshData.baseTriangles.info.range);

  nvmath::vec4f* sphereData =
      staging->cmdToBufferT<nvmath::vec4f>(cmd, meshData.baseSpheres.buffer, meshData.baseSpheres.info.offset,
                                           meshData.baseSpheres.info.range);

  nvh::parallel_batches(
      baryGroup.triangleCount,
      [&](uint64_t baryLocalTriIdx) {
        const size_t          meshGlobalTriIdx = baryLocalTriIdx;
        const size_t          baryGlobalTriIdx = baryGroup.triangleFirst + baryLocalTriIdx;
        const bary::Triangle& baseTri          = basic.triangles[baryGlobalTriIdx];
        const uint16_t* baseMinMaxs = reinterpret_cast<const uint16_t*>(basic.triangleMinMaxs + baryGlobalTriIdx * minMaxByteSize);

        uint32_t formatIndex = getFormatIndex(baseTri.blockFormat);

        float valueBias  = baryGroup.floatBias.r;
        float valueScale = baryGroup.floatScale.r;

        MicromeshBaseTri& micro  = baseTriData[baryLocalTriIdx];
        nvmath::vec4f&    sphere = sphereData[baryLocalTriIdx];

        const int32_t upperValueBound = 0x7FF;
        const int32_t upperCullBound  = MICRO_BASE_CULLDIST_MASK;

        float minDisp = float(baseMinMaxs[0]) / float(upperValueBound) * valueScale + valueBias;
        float maxDisp = float(baseMinMaxs[1]) / float(upperValueBound) * valueScale + valueBias;

        //float   delta    = !mesh.directionBoundsAreUniform && meshSet.attributes.directionBounds.empty() ? 0.0f : -0.5f;
        //float   absDist  = std::max(std::abs(minDisp + delta), std::abs(maxDisp + delta));
        //int32_t cullDist = int32_t(((absDist - valueBias) / valueScale) * float(upperCullBound));
        //cullDist         = std::min(std::max(cullDist, 1), upperCullBound);

        const uint32_t baseTopo = (decimateEdgeFlags ? decimateEdgeFlags[meshGlobalTriIdx] : 0);

        micro.dataOffset = (baseTri.valuesOffset) / sizeof(uint32_t);

        micro.packedBits = 0;
        micro.packedBits |= packBits(baseTri.subdivLevel, MICRO_BASE_LVL_SHIFT, MICRO_BASE_LVL_WIDTH);
        micro.packedBits |= packBits(baseTopo, MICRO_BASE_TOPO_SHIFT, MICRO_BASE_TOPO_WIDTH);
        micro.packedBits |= packBits(formatIndex, MICRO_BASE_FMT_SHIFT, MICRO_BASE_FMT_WIDTH);
        //micro.packedBits |= packBits(cullDist, MICRO_BASE_CULLDIST_SHIFT, MICRO_BASE_CULLDIST_WIDTH);

        //sphere = computeSphere(micro, meshSet, meshGlobalTriIdx, minDisp, maxDisp, mesh.directionBoundsAreUniform);
      },
      numThreads);
}

void MicromeshSubTriangleDecoderVK::uploadVertices(nvvk::StagingMemoryManager* staging, VkCommandBuffer cmd, const MicroSplitParts& splits)
{
  MicromeshSTriVertex* verticesAll =
      staging->cmdToBufferT<MicromeshSTriVertex>(cmd, m_micro.vertices.buffer, m_micro.vertices.info.offset,
                                                 m_micro.vertices.info.range);

  memset(verticesAll, 0, m_micro.vertices.info.range);

  uint32_t totalMeshlets = 0;
  for(uint32_t subdivLevel = 3; subdivLevel <= 5; subdivLevel++)
  {

    uint32_t numMeshlets = 1 << (subdivLevel - MICRO_PART_MAX_SUBDIV);
    numMeshlets          = numMeshlets * numMeshlets;

    for(uint32_t meshlet = 0; meshlet < numMeshlets; meshlet++)
    {
      MicromeshSTriVertex* verticesMeshlet = verticesAll + (totalMeshlets + meshlet) * MICRO_PART_VERTICES_STRIDE;

      for(uint32_t v = 0; v < MICRO_PART_MAX_VERTICES; v++)
      {
        baryutils::BaryWUV_uint16 coord   = splits.partLevel.coordinates[v];
        bary::BaryUV_uint16       coordUV = {coord.u, coord.v};

        if(numMeshlets > 1)
        {
          const bary::BlockTriangle* splitLevel = &splits.triLevelNtoN[subdivLevel][3][meshlet];

          // apply split transform
          coordUV = bary::baryBlockTriangleLocalToBaseUV(splitLevel, coordUV);
          coord   = {uint16_t((1 << subdivLevel) - coordUV.u - coordUV.v), coordUV.u, coordUV.v};
        }

        uint32_t decodeLevel;
        uint32_t decodeLevelCoordIndex;
        bary::baryBirdLayoutGetVertexLevelInfo(coord.u, coord.v, subdivLevel, &decodeLevel, &decodeLevelCoordIndex);

        MicromeshSTriVertex& mvtx = verticesMeshlet[v];
        mvtx.packed               = 0;
        mvtx.packed |= packBits(decodeLevel, MICRO_STRI_VTX_LVL_SHIFT, MICRO_STRI_VTX_LVL_WIDTH);
        mvtx.packed |= packBits(decodeLevelCoordIndex, MICRO_STRI_VTX_IDX_SHIFT, MICRO_STRI_VTX_IDX_WIDTH);
        mvtx.packed |= packBits(coord.u, MICRO_STRI_VTX_U_SHIFT, MICRO_STRI_VTX_UV_WIDTH);
        mvtx.packed |= packBits(coord.v, MICRO_STRI_VTX_V_SHIFT, MICRO_STRI_VTX_UV_WIDTH);
        mvtx.packed |= packBits(splits.partVertexMergeIndices[v].a, MICRO_STRI_VTX_A_SHIFT, MICRO_STRI_VTX_AB_WIDTH);
        mvtx.packed |= packBits(splits.partVertexMergeIndices[v].b, MICRO_STRI_VTX_B_SHIFT, MICRO_STRI_VTX_AB_WIDTH);
      }

      verticesMeshlet = verticesMeshlet;
    }

    totalMeshlets += numMeshlets;
  }
}

void MicromeshSubTriangleDecoderVK::uploadDescends(nvvk::StagingMemoryManager* staging, VkCommandBuffer cmd, const MicroSplitParts& splits)
{
  MicromeshSTriDescend* descendInfosAll =
      staging->cmdToBufferT<MicromeshSTriDescend>(cmd, m_micro.descends.buffer, m_micro.descends.info.offset,
                                                  m_micro.descends.info.range);

  for(uint32_t subdivLevel = 4; subdivLevel <= 5; subdivLevel++)
  {
    MicromeshSTriDescend* descendCur = descendInfosAll + (subdivLevel == 5 ? 4 : 0);

    uint32_t subdivDescend = subdivLevel - 3;

    const bary::BlockTriangle* splitLevels[2] = {subdivLevel == 4 ? splits.triLevel4to3 : splits.triLevel5to4, splits.triLevel5to3};

    for(uint32_t partID = 0; partID < (subdivLevel == 4u ? 4u : 16u); partID++)
    {
      MicromeshSTriDescend partMicroInfo;
      memset(&partMicroInfo, 0, sizeof(MicromeshSTriDescend));

      for(uint32_t vtx = 0; vtx < 3; vtx++)
      {
        for(uint32_t lvl = 0; lvl < subdivDescend; lvl++)
        {
          nvmath::vec2ui mergeIndices[3];

          switch((partID / ((subdivLevel == 5 && lvl == 0) ? 4 : 1)) & 3)
          {
            case 0:
              mergeIndices[0] = {0, 0};
              mergeIndices[1] = {0, 1};
              mergeIndices[2] = {0, 2};
              break;
            case 1:
              mergeIndices[0] = {0, 2};
              mergeIndices[1] = {1, 2};
              mergeIndices[2] = {0, 1};
              break;
            case 2:
              mergeIndices[0] = {0, 1};
              mergeIndices[1] = {1, 1};
              mergeIndices[2] = {1, 2};
              break;
            case 3:
              mergeIndices[0] = {1, 2};
              mergeIndices[1] = {0, 2};
              mergeIndices[2] = {2, 2};
              break;
          }

          // find appropriate splitlevel
          uint32_t                   splitIdx = subdivLevel == 5 && lvl == 0 ? (partID / 4) : partID;
          const bary::BlockTriangle& split    = splitLevels[lvl][splitIdx];

          uint32_t decodeLevel;
          uint32_t decodeLevelCoordIndex;

          baryutils::BaryWUV_uint16 coord = {0, split.vertices[vtx].u, split.vertices[vtx].v};
          coord.w                         = uint16_t((1 << subdivLevel) - coord.u - coord.v);

          bary::baryBirdLayoutGetVertexLevelInfo(coord.u, coord.v, subdivLevel, &decodeLevel, &decodeLevelCoordIndex);
          MicromeshSTriVertex& mvtx = partMicroInfo.vertices[vtx + 3 * lvl];

          mvtx.packed = 0;
          mvtx.packed |= packBits(decodeLevel, MICRO_STRI_VTX_LVL_SHIFT, MICRO_STRI_VTX_LVL_WIDTH);
          mvtx.packed |= packBits(decodeLevelCoordIndex, MICRO_STRI_VTX_IDX_SHIFT, MICRO_STRI_VTX_IDX_WIDTH);
          mvtx.packed |= packBits(split.vertices[vtx].u, MICRO_STRI_VTX_U_SHIFT, MICRO_STRI_VTX_UV_WIDTH);
          mvtx.packed |= packBits(split.vertices[vtx].v, MICRO_STRI_VTX_V_SHIFT, MICRO_STRI_VTX_UV_WIDTH);
          mvtx.packed |= packBits(mergeIndices[vtx].x, MICRO_STRI_VTX_A_SHIFT, MICRO_STRI_VTX_AB_WIDTH);
          mvtx.packed |= packBits(mergeIndices[vtx].y, MICRO_STRI_VTX_B_SHIFT, MICRO_STRI_VTX_AB_WIDTH);
        }
      }

      descendCur[partID] = partMicroInfo;
    }
  }
}

}  // namespace microdisp