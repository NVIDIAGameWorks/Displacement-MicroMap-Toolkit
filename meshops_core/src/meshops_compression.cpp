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

#include <meshops_internal/meshops_context.h>
#include <meshops/meshops_operations.h>
#include <microutils/microutils_compression.hpp>

namespace meshops {
//////////////////////////////////////////////////////////////////////////

MESHOPS_API micromesh::Result MESHOPS_CALL meshopsOpCompressDisplacementMicromaps(Context context,
                                                                                  size_t  count,
                                                                                  const OpCompressDisplacementMicromap_input* inputs,
                                                                                  OpCompressDisplacementMicromap_output* outputs)

{
  assert(context && inputs && outputs);

  micromesh::Result result;

  for(size_t i = 0; i < count; ++i)
  {
    const OpCompressDisplacementMicromap_input& input  = inputs[i];
    OpCompressDisplacementMicromap_output&      output = outputs[i];

    assert(input.uncompressedDisplacement && output.compressedDisplacement);

    const bary::BasicView&    basicUncompressed = *input.uncompressedDisplacement;
    baryutils::BaryBasicData& dataCompressed    = *output.compressedDisplacement;


    const uint32_t groupIdx = input.uncompressedDisplacementGroupIndex;

    if(groupIdx >= basicUncompressed.groupsCount || basicUncompressed.groups[groupIdx].maxSubdivLevel > 5
       || input.meshView.vertexDirections.empty())
    {
      assert(0);
      return micromesh::Result::eInvalidValue;
    }

    // initialize compressed
    dataCompressed = baryutils::BaryBasicData();

    microutils::baryBasicDataCompressedInit(dataCompressed, bary::Format::eDispC1_r11_unorm_block);
    if(output.compressedDisplacementRasterMips)
    {
      microutils::baryMiscDataUncompressedMipInit(*output.compressedDisplacementRasterMips);
    }

    bary::Group               baryGroupCompressed = {0};
    bary::GroupHistogramRange baryGroupHistogram  = {0};

    const bary::Group& baryGroupUncompressed = basicUncompressed.groups[groupIdx];

    micromesh::Micromap  inputMap;
    micromesh::ArrayInfo inputMinMaxs;


    micromesh::MicromapGeneric uncompressedMap;
    bary::Result baryResult = microutils::baryBasicViewToMicromap(basicUncompressed, groupIdx, uncompressedMap);
    if(baryResult != bary::Result::eSuccess)
    {
      assert(0);
      return micromesh::Result::eFailure;
    }

    if(uncompressedMap.type != micromesh::MicromapType::eUncompressed)
    {
      // don't bother with packed for now
      // complicates a few things
      return micromesh::Result::eFailure;
    }

    baryResult = microutils::baryBasicViewToMinMaxs(basicUncompressed, groupIdx, inputMinMaxs);
    if(baryResult != bary::Result::eSuccess)
    {
      assert(0);
      return micromesh::Result::eFailure;
    }

    inputMap = uncompressedMap.uncompressed;

    // use magnitude of direction vector to help drive compression heuristic
    std::vector<float>         vtxImportance(input.meshView.vertexCount());
    micromesh::ArrayInfo_float perVertexImportance;
    {
      const auto& directionView = input.meshView.vertexDirections;
      const auto& boundsView    = input.meshView.vertexDirectionBounds;
      bool        useBounds     = !boundsView.empty();

      size_t vertexCount = vtxImportance.size();
      for(size_t v = 0; v < vertexCount; v++)
      {
        float norm       = directionView[v].norm();
        vtxImportance[v] = norm * (useBounds ? boundsView[v].y : 1.0f);
      }

      double importanceSum = 0.0;
      for(const float& v : vtxImportance)
      {
        importanceSum += static_cast<double>(v);
      }

      const float importanceMul = float(double(vtxImportance.size()) / importanceSum);
      for(size_t v = 0; v < vertexCount; v++)
      {
        vtxImportance[v] *= importanceMul;
      }

      micromesh::arraySetData(perVertexImportance, vtxImportance.data(), vtxImportance.size());
    }

    // The default rasterization implementation expects these settings
    microutils::UncompressedMipSettings mipSettings;
    mipSettings.minSubdiv           = 4;
    mipSettings.mipSubdiv           = 2;
    mipSettings.skipBlockFormatBits = (1u << uint32_t(micromesh::BlockFormatDispC1::eR11_unorm_lvl3_pack512));

    result = microutils::baryBasicDataCompressedAppend(dataCompressed, context->m_micromeshContext, input.settings,
                                                       *input.meshTopology, inputMap, inputMinMaxs, nullptr, &perVertexImportance,
                                                       output.compressedDisplacementRasterMips, &mipSettings);
    if(result != micromesh::Result::eSuccess)
    {
      assert(0);
      return result;
    }
  }

  return micromesh::Result::eSuccess;
}

}  // namespace meshops