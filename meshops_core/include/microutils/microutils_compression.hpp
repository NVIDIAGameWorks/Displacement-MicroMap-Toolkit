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

#include <microutils/microutils.hpp>
#include <baryutils/baryutils.h>
#include <micromesh/micromesh_displacement_compression.h>

namespace microutils {

class ThreadedTriangleDecoder
{
public:
  void init(bary::Format format, bary::ValueLayout layout, uint32_t maxSubdivLevel, uint32_t numThreads);

  uint16_t* tempThreadDecode(uint32_t threadIndex, const bary::BasicView& basic, uint32_t groupIndex, uint32_t triangleIndex, uint32_t& out_valueCount);

private:
  uint32_t                  numDecoderValuesMax{};
  size_t                    numUncompressedValuesMax{};
  uint64_t                  scratchDataSize{};
  std::vector<uint16_t>     threadDecoderValues;
  std::vector<uint16_t>     threadUncompressedValues;
  std::vector<uint8_t>      threadScratchData;
  baryutils::BarySplitTable splitTable;
  baryutils::BaryLevelsMap  levelsMap;
};

// use baryBasicDataCompressedInit to start compressing and and appending new groups
// into BaryBasicData
void baryBasicDataCompressedInit(baryutils::BaryBasicData& baryCompressed, bary::Format format);

void baryMiscDataUncompressedMipInit(baryutils::BaryMiscData& baryMisc);

struct UncompressedMipSettings
{
  // the following settings are tuned to work well with the reference rasterization
  // implementation provided in one of the samples. It speeds up the decoding
  // compress of displacement blocks within the shaders.

  // the subdiv level of the per triangle mip, due to hierarchical scheme of bird curve
  // the first n vertices represent these levels.
  uint32_t mipSubdiv = 2;
  // which base triangle subdiv levels need mip data
  uint32_t minSubdiv = 4;
  // which base triangle blockformats should not require mip data
  uint32_t skipBlockFormatBits = 1u << uint32_t(micromesh::BlockFormatDispC1::eR11_unorm_lvl3_pack512);
};

// compress one group at a time and append it to BaryBasicData
// must call baryBasicDataCompressedInit once prior append
micromesh::Result baryBasicDataCompressedAppend(baryutils::BaryBasicData&                         baryCompressed,
                                                micromesh::OpContext                              ctx,
                                                const micromesh::OpCompressDisplacement_settings& settings,
                                                const micromesh::MeshTopology&                    meshTopo,
                                                const micromesh::MicromapGeneric&                 uncompressedMap,
                                                const micromesh::ArrayInfo& uncompressedTriangleMinMaxs,
                                                // meshMinMaxs optional for floats
                                                const float* meshMinMaxs = nullptr,
                                                // optional perVertexImportance (typically magnitude of final direction vector)
                                                const micromesh::ArrayInfo_float* perVertexImportances = nullptr,
                                                // optional mip output
                                                baryutils::BaryMiscData* mipBaryMisc = nullptr,
                                                // mipSettings must be provided if mipBaryMisc is non-null
                                                UncompressedMipSettings* mipSettings = nullptr);

// uncompress and fill BaryBasicData from compressed data
micromesh::Result baryBasicDataUncompressedFill(baryutils::BaryBasicData& baryUncompressed,
                                                micromesh::OpContext      ctx,
                                                const bary::BasicView&    basicCompressed);

// skipFlat    = do not add uncompressed / easy to decode format
// minSubdiv   = minimum subdiv level base-primitive must have to get mip data (must >= mipSubdiv)
// mipSubdiv   = the subdiv level for the mipmap data
micromesh::Result baryMiscDataSetupMips(baryutils::BaryMiscData&       mip,
                                        micromesh::OpContext           ctx,
                                        const bary::BasicView&         basicCompressed,
                                        const UncompressedMipSettings& settings);

micromesh::Result baryBasicDataCompressedUpdateTriangleMinMaxs(baryutils::BaryBasicData& bary, micromesh::OpContext ctx);
}  // namespace microutils
