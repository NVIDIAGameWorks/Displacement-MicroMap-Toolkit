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

#include <microutils/microutils_compression.hpp>
#include <algorithm>
#include <limits>

namespace microutils {
//////////////////////////////////////////////////////////////////////////

void baryBasicDataCompressedInit(baryutils::BaryBasicData& baryCompressed, bary::Format format)
{
  assert(bary::Format::eDispC1_r11_unorm_block == format && "incompatible compressed format");

  baryCompressed = baryutils::BaryBasicData();

  memset(&baryCompressed.valuesInfo, 0, sizeof(baryCompressed.valuesInfo));
  memset(&baryCompressed.triangleMinMaxsInfo, 0, sizeof(baryCompressed.triangleMinMaxsInfo));
  baryCompressed.valuesInfo.valueByteAlignment            = 128;
  baryCompressed.valuesInfo.valueByteSize                 = 1;
  baryCompressed.valuesInfo.valueFrequency                = bary::ValueFrequency::ePerVertex;
  baryCompressed.valuesInfo.valueLayout                   = bary::ValueLayout::eTriangleBirdCurve;
  baryCompressed.valuesInfo.valueFormat                   = format;
  baryCompressed.triangleMinMaxsInfo.elementByteSize      = sizeof(uint16_t);
  baryCompressed.triangleMinMaxsInfo.elementByteAlignment = 4;
  baryCompressed.triangleMinMaxsInfo.elementFormat        = bary::Format::eR11_unorm_pack16;
}

void baryMiscDataUncompressedMipInit(baryutils::BaryMiscData& baryMisc)
{
  baryMisc.groupUncompressedMips    = std::vector<bary::GroupUncompressedMip>();
  baryMisc.triangleUncompressedMips = std::vector<bary::TriangleUncompressedMip>();
  baryMisc.uncompressedMips         = std::vector<uint8_t>();

  baryMisc.uncompressedMipsInfo.elementCount         = 0;
  baryMisc.uncompressedMipsInfo.elementByteAlignment = 4;
  baryMisc.uncompressedMipsInfo.elementByteSize      = 1;
  baryMisc.uncompressedMipsInfo.elementFormat        = bary::Format::eR11_unorm_packed_align32;
}

micromesh::Result baryBasicDataCompressedAppend(baryutils::BaryBasicData&                         baryCompressed,
                                                micromesh::OpContext                              ctx,
                                                const micromesh::OpCompressDisplacement_settings& settings,
                                                const micromesh::MeshTopology&                    topo,
                                                const micromesh::Micromap&                        inputMap,
                                                const micromesh::ArrayInfo&                       inputTriangleMinMaxs,
                                                const float*                                      meshMinMaxs,
                                                const micromesh::ArrayInfo_float*                 perVertexImportance,
                                                baryutils::BaryMiscData*                          mipBaryMisc,
                                                UncompressedMipSettings*                          mipSettings)
{
  assert(baryCompressed.valuesInfo.valueFormat == bary::Format::eDispC1_r11_unorm_block
         && "incompatible compressed format");

  bary::Group               baryGroupCompressed = {0};
  bary::GroupHistogramRange baryGroupHistogram  = {0};
  micromesh::Result         result;

  micromesh::Micromap uncompressedMap = inputMap;

  uint32_t triangleCount = uint32_t(inputMap.triangleSubdivLevels.count);

  // we may need to convert to r11_16
  std::vector<uint16_t> values_unorm11;
  if(inputMap.values.format != micromesh::Format::eR11_unorm_pack16)
  {
    micromesh::Micromap convertMapIn  = inputMap;
    micromesh::Micromap convertMapOut = inputMap;

    values_unorm11.resize(inputMap.values.count);

    convertMapOut.values.byteStride = sizeof(uint16_t);
    convertMapOut.values.format     = micromesh::Format::eR11_unorm_pack16;
    convertMapOut.values.count      = inputMap.values.count;
    convertMapOut.values.data       = values_unorm11.data();

    // if we are coming from float, convert to quantized
    // use provided meshMinMaxs pointer if available otherwise
    // get min/max from triangleMinMaxs
    float floatValueMin = meshMinMaxs ? meshMinMaxs[0] : std::numeric_limits<float>::max();
    float floatValueMax = meshMinMaxs ? meshMinMaxs[1] : -std::numeric_limits<float>::max();
    if(convertMapIn.values.format == micromesh::Format::eR32_sfloat)
    {
      if(!meshMinMaxs)
      {
        for(uint64_t i = 0; i < inputTriangleMinMaxs.count / 2; i++)
        {
          floatValueMin = std::min(floatValueMin, micromesh::arrayGetV<float>(inputTriangleMinMaxs, i * 2 + 0));
          floatValueMax = std::max(floatValueMax, micromesh::arrayGetV<float>(inputTriangleMinMaxs, i * 2 + 1));
        }
      }

      micromesh::OpFloatToQuantized_input inputConvert;
      inputConvert.floatMicromap            = &convertMapIn;
      inputConvert.outputUnsignedSfloat     = true;
      inputConvert.globalMin.value_float[0] = floatValueMin;
      inputConvert.globalMax.value_float[0] = floatValueMax;
      result = micromesh::micromeshOpFloatToQuantized(ctx, &inputConvert, &convertMapOut);
    }
    else
    {
      result = micromesh::micromeshOpQuantizedToQuantized(ctx, &convertMapIn, &convertMapOut);
    }
    if(result != micromesh::Result::eSuccess)
    {
      assert(0);
      return result;
    }
    uncompressedMap = convertMapOut;
  }

  micromesh::MicromapPacked mipPacked;

  if(mipBaryMisc)
  {
    assert(mipSettings);

    mipPacked.minSubdivLevel = mipSettings->mipSubdiv;
    mipPacked.maxSubdivLevel = mipSettings->mipSubdiv;
    micromesh::micromeshLayoutInitStandard(&mipPacked.layout, micromesh::StandardLayoutType::eBirdCurve);
    mipPacked.frequency = micromesh::Frequency::ePerMicroVertex;

    bary::GroupUncompressedMip mipGroup = {};

    assert(baryCompressed.triangles.size() == mipBaryMisc->triangleUncompressedMips.size());

    size_t mipTrianglesBegin = mipBaryMisc->triangleUncompressedMips.size();
    mipBaryMisc->triangleUncompressedMips.resize(mipBaryMisc->triangleUncompressedMips.size() + triangleCount);

    bary::TriangleUncompressedMip* mipTriangles = mipBaryMisc->triangleUncompressedMips.data() + mipTrianglesBegin;

    // warning &mipTriangles[0].subdivLevel is u32 rather than u16, need to account for that later

    micromesh::arraySetData(mipPacked.triangleSubdivLevels, &mipTriangles[0].subdivLevel, triangleCount,
                            sizeof(bary::TriangleUncompressedMip));
    micromesh::arraySetData(mipPacked.triangleValueByteOffsets, &mipTriangles[0].mipOffset, triangleCount,
                            sizeof(bary::TriangleUncompressedMip));

    const uint32_t numMipValues = bary::baryValueFrequencyGetCount(bary::ValueFrequency::ePerVertex, mipSettings->mipSubdiv);
    const uint32_t mipEntrySize = ((numMipValues * 11 + 31) / 32) * uint32_t(sizeof(uint32_t));  // Number of bytes in a triangle's mip entry

    uint32_t mipOffset = 0;

    // compute mip subdiv levels and bytes for storage
    for(uint32_t i = 0; i < triangleCount; i++)
    {
      // u32/u16 works only on little endian systems
      uint16_t subdivLevel        = micromesh::arrayGetV<uint16_t>(inputMap.triangleSubdivLevels, i);
      bool     needMip            = subdivLevel >= mipSettings->minSubdiv;
      mipTriangles[i].subdivLevel = needMip ? mipSettings->mipSubdiv : settings.mipIgnoredSubdivLevel;
      mipTriangles[i].mipOffset   = mipOffset;

      mipOffset += needMip ? mipEntrySize : 0;
    }

    // add dummy 4 bytes if there is nothing for now
    // avoids various validation errors around zero length etc.
    mipOffset = std::max(mipOffset, 4u);

    // append group
    mipGroup.mipFirst = mipBaryMisc->uncompressedMipsInfo.elementCount;
    mipGroup.mipCount = mipOffset;
    mipBaryMisc->groupUncompressedMips.push_back(mipGroup);

    // append values
    mipBaryMisc->uncompressedMipsInfo.elementCount += mipOffset;
    mipBaryMisc->uncompressedMips.resize(mipBaryMisc->uncompressedMipsInfo.elementCount);

    // setup mipPacked values
    mipPacked.values.format = micromesh::Format::eR11_unorm_packed_align32;
    micromesh::arraySetData(mipPacked.values, &mipBaryMisc->uncompressedMips[mipGroup.mipFirst], mipOffset, 1);
  }

  // compression operation

  // setup input
  // this is the input struct passed to compressor
  micromesh::OpCompressDisplacement_input input;
  // get family format for the compression
  // we actually only have one family format, so this currently always is
  // micromesh::Format::eDispC1_r11_unorm_block
  input.compressedFormatFamily = microutils::getMicromeshFormat(baryCompressed.valuesInfo.valueFormat);
  // the uncompressed unorm11 input data
  input.data = &uncompressedMap;
  // the micromesh::MeshTopology used to ensure watertightness
  input.topology = &topo;
  // optional input
  if(perVertexImportance)
  {
    // per vertex importance is the magnitude of the linear direction vectors
    // it allows to make a better global judgment for compression quality
    input.perVertexImportance = *perVertexImportance;
  }

  // output map struct, we pass it with default constructor state,
  // without real data to begin, however we get information back how much to allocate
  micromesh::MicromapCompressed            compressedMap;
  micromesh::OpCompressDisplacement_output output{};
  output.compressed = &compressedMap;
  if(mipBaryMisc)
  {
    output.mipData = &mipPacked;
  }

  // run compression begin function
  result = micromesh::micromeshOpCompressDisplacementBegin(ctx, &settings, &input, &output);
  if(result != micromesh::Result::eSuccess)
  {
    assert(0);
    return result;
  }

  // after begin function completes, we have the sizing information
  // for the output compressedMap

  // setup new baryGroup inside `baryutils::BaryBasicData baryCompressed`
  // we append to the data already stored in the provided bary file container.
  // get offsets and counts for the new group
  baryGroupCompressed.triangleFirst = uint32_t(baryCompressed.triangles.size());
  baryGroupCompressed.triangleCount = uint32_t(compressedMap.triangleBlockFormats.count);
  baryGroupCompressed.valueFirst    = uint32_t(baryCompressed.values.size());
  baryGroupCompressed.valueCount    = uint32_t(compressedMap.values.count);
  // resize number of triangles and values in `baryCompressed`
  baryCompressed.triangles.resize(baryCompressed.triangles.size() + compressedMap.triangleBlockFormats.count);
  baryCompressed.values.resize(baryCompressed.values.size() + compressedMap.values.count);
  baryCompressed.valuesInfo.valueCount += uint32_t(compressedMap.values.count);
  // setup pointers / adjust stride etc. for the compressedMap,
  // which is to be passed into the end function
  compressedMap.values.data               = baryCompressed.values.data() + baryGroupCompressed.valueFirst;
  compressedMap.triangleBlockFormats.data = &baryCompressed.triangles[baryGroupCompressed.triangleFirst].blockFormat;
  compressedMap.triangleBlockFormats.byteStride = sizeof(bary::Triangle);
  compressedMap.triangleSubdivLevels.data = &baryCompressed.triangles[baryGroupCompressed.triangleFirst].subdivLevel;
  compressedMap.triangleSubdivLevels.byteStride = sizeof(bary::Triangle);
  compressedMap.triangleValueByteOffsets.data = &baryCompressed.triangles[baryGroupCompressed.triangleFirst].valuesOffset;
  compressedMap.triangleValueByteOffsets.byteStride = sizeof(bary::Triangle);

  // Also set up the optional triangle min/max values. These contain the
  // min/max of the compressed data, which may be different than the input min/max.
  output.triangleMinMaxs.count      = 2 * triangleCount;
  output.triangleMinMaxs.byteStride = 2;
  output.triangleMinMaxs.format     = micromesh::Format::eR11_unorm_pack16;
  baryCompressed.triangleMinMaxsInfo.elementCount += uint32_t(output.triangleMinMaxs.count);
  baryCompressed.triangleMinMaxs.resize(baryCompressed.triangleMinMaxs.size()
                                        + output.triangleMinMaxs.count * output.triangleMinMaxs.byteStride);
  output.triangleMinMaxs.data =
      &baryCompressed.triangleMinMaxs[baryGroupCompressed.triangleFirst * 2 * output.triangleMinMaxs.byteStride];

  // run compression end function
  // this fills in all the pointers we just setup for the compressedMap
  result = micromesh::micromeshOpCompressDisplacementEnd(ctx, &output);
  if(result != micromesh::Result::eSuccess)
  {
    assert(0);
    return result;
  }

  // TODO: if we have just normalized values in micromeshOpFloatToQuantized, the bias and scale should now be 0 and 1.
  // Maybe slightly out due to floating point error, but in that case shouldn't this be replaced with exactly 0 and 1?
  baryGroupCompressed.floatBias.r    = compressedMap.valueFloatExpansion.bias[0];
  baryGroupCompressed.floatScale.r   = compressedMap.valueFloatExpansion.scale[0];
  baryGroupCompressed.minSubdivLevel = compressedMap.minSubdivLevel;
  baryGroupCompressed.maxSubdivLevel = compressedMap.maxSubdivLevel;
  // append new group
  baryCompressed.groups.push_back(baryGroupCompressed);

  {
    // for compressed values we always want histogram information
    // over how much certain blockformats are used. This aids
    // decompressing but also the 3D APIs sizing estimates.

    // setup input
    micromesh::OpComputeBlockFormatUsages_input histoInput;
    // the data of compressedMap is our input
    histoInput.compressed = &compressedMap;
    // we leave this one empty, as is because
    // we are not interested in the instanced histogram, but pure data histogram
    // (also our application use 1:1 mapping)
    histoInput.meshTriangleMappings;

    // reserve worst-case histogram bins
    std::vector<micromesh::BlockFormatUsage> blockFormatUsages(micromesh::micromeshGetBlockFormatUsageReserveCount(&compressedMap));

    // setup output
    micromesh::OpComputeBlockFormatUsages_output histoOutput;
    histoOutput.pUsages            = blockFormatUsages.data();
    histoOutput.reservedUsageCount = uint32_t(blockFormatUsages.size());
    histoOutput.usageCount         = 0;
    result                         = micromesh::micromeshOpComputeBlockFormatUsages(ctx, &histoInput, &histoOutput);
    if(result != micromesh::Result::eSuccess)
    {
      assert(0);
      return result;
    }

    // append the data from the micromesh operation to `baryCompressed`

    baryGroupHistogram.entryFirst = uint32_t(baryCompressed.histogramEntries.size());
    baryGroupHistogram.entryCount = histoOutput.usageCount;

    for(uint32_t i = 0; i < histoOutput.usageCount; i++)
    {
      baryCompressed.histogramEntries.push_back(microutils::getBaryHistogramEntry(blockFormatUsages[i]));
    }

    baryCompressed.groupHistogramRanges.push_back(baryGroupHistogram);
  }

  if(mipBaryMisc)
  {
    // let's filter out mip blocks of skipped formats

    for(uint32_t i = 0; i < triangleCount; i++)
    {
      const bary::Triangle&          tri    = baryCompressed.triangles[i + baryGroupCompressed.triangleFirst];
      bary::TriangleUncompressedMip& triMip = mipBaryMisc->triangleUncompressedMips[i + baryGroupCompressed.triangleFirst];
      if((mipSettings->skipBlockFormatBits & (1u << uint32_t(tri.blockFormatDispC1))) != 0)
      {
        // using ~0 means we won't use the mip data
        // FIXME compaction of mipvalues / mipOffsets
        // with the current scheme we have over-allocated mipValues,
        // we could compact the values etc. to reduce overall container size
        triMip.mipOffset   = ~0;
        triMip.subdivLevel = 0;
      }
    }
  }

  return micromesh::Result::eSuccess;
}

micromesh::Result baryBasicDataUncompressedFill(baryutils::BaryBasicData& baryUncompressed,
                                                micromesh::OpContext      ctx,
                                                const bary::BasicView&    basicCompressed)
{
  micromesh::Result result;

  baryUncompressed = baryutils::BaryBasicData();

  baryUncompressed.triangles.resize(basicCompressed.trianglesCount);
  baryUncompressed.valuesInfo.valueByteAlignment = 4;
  baryUncompressed.valuesInfo.valueByteSize      = sizeof(uint16_t);
  baryUncompressed.valuesInfo.valueFormat        = bary::Format::eR11_unorm_pack16;
  baryUncompressed.valuesInfo.valueLayout        = bary::ValueLayout::eTriangleBirdCurve;
  baryUncompressed.valuesInfo.valueFrequency     = bary::ValueFrequency::ePerVertex;
  baryUncompressed.valuesInfo.valueCount         = 0;

  if(basicCompressed.triangleMinMaxsInfo && baryUncompressed.triangleMinMaxsInfo.elementCount)
  {
    baryUncompressed.triangleMinMaxsInfo = *basicCompressed.triangleMinMaxsInfo;
    baryUncompressed.triangleMinMaxs.resize(baryUncompressed.triangleMinMaxsInfo.elementByteSize
                                            * baryUncompressed.triangleMinMaxsInfo.elementCount);
    assert(baryUncompressed.triangleMinMaxsInfo.elementFormat == bary::Format::eR11_unorm_pack16);
    // copy in bulk
    memcpy(baryUncompressed.triangleMinMaxs.data(), basicCompressed.triangleMinMaxs, baryUncompressed.triangleMinMaxs.size());
  }

  bary::baryBasicViewGetMinMaxSubdivLevels(&basicCompressed, &baryUncompressed.minSubdivLevel, &baryUncompressed.maxSubdivLevel);

  baryUncompressed.groups.resize(basicCompressed.groupsCount);

  // must setup basic after triangles/groups arrays have been allocated
  // values will be manually set
  bary::BasicView basicUncompressed = baryUncompressed.getView();

  for(uint32_t g = 0; g < basicCompressed.groupsCount; g++)
  {
    // we preserve everything, except value related data
    bary::Group& groupUncompressed = baryUncompressed.groups[g];
    groupUncompressed              = basicCompressed.groups[g];

    // get micromap accessor for uncompressed group
    micromesh::Micromap mapUncompressed = micromapFromBasicGroup(basicUncompressed, g);

    // get micromap accessor for compressed group
    micromesh::MicromapGeneric micromapGen;
    bary::Result               baryResult = baryBasicViewToMicromap(basicCompressed, g, micromapGen);
    if(baryResult != bary::Result::eSuccess || micromapGen.type != micromesh::MicromapType::eCompressed
       || micromapGen.compressed.values.format != micromesh::Format::eDispC1_r11_unorm_block)
    {
      baryUncompressed = baryutils::BaryBasicData();
      assert(0);
      return micromesh::Result::eInvalidFormat;
    }

    // begin
    result = micromesh::micromeshOpDecompressDisplacementBegin(ctx, &micromapGen.compressed, &mapUncompressed);
    if(result != micromesh::Result::eSuccess)
    {
      baryUncompressed = baryutils::BaryBasicData();
      assert(0);
      return result;
    }
    // after begin we know the number of uncompressed values
    groupUncompressed.valueFirst = baryUncompressed.valuesInfo.valueCount;
    groupUncompressed.valueCount = uint32_t(mapUncompressed.values.count);
    // resize values data
    baryUncompressed.valuesInfo.valueCount += groupUncompressed.valueCount;
    baryUncompressed.values.resize(baryUncompressed.valuesInfo.valueCount * baryUncompressed.valuesInfo.valueByteSize);
    // setup mapUncompressed pointer for actual decoding
    mapUncompressed.values.data =
        baryUncompressed.values.data() + (groupUncompressed.valueFirst * baryUncompressed.valuesInfo.valueByteSize);
    // do decoding
    result = micromesh::micromeshOpDecompressDisplacementEnd(ctx, &mapUncompressed);
    if(result != micromesh::Result::eSuccess)
    {
      baryUncompressed = baryutils::BaryBasicData();
      assert(0);
      return result;
    }
  }
  return micromesh::Result::eSuccess;
}

static micromesh::Result decodeBlockInto(const uint8_t*                         blockValues,
                                         bary::BlockFormatDispC1                blockFormat,
                                         const bary::BlockTriangle*             blockSplit,
                                         const baryutils::BaryLevelsMap::Level& blockLevel,
                                         const baryutils::BaryLevelsMap::Level& baseLevel,
                                         uint16_t*                              blockDecoded,
                                         uint16_t*                              baseUncompressed,
                                         uint64_t                               scratchDataSize,
                                         void*                                  scratchData)
{

  micromesh::DisplacementBlock_settings settings;
  settings.compressedBlockFormatDispC1 = microutils::getMicromeshBlockFormatDispC1(blockFormat);
  settings.compressedFormat            = micromesh::Format::eDispC1_r11_unorm_block;
  settings.decompressedFormat          = micromesh::Format::eR11_unorm_pack16;
  settings.subdivLevel                 = blockLevel.subdivLevel;
  micromesh::Result result             = micromesh::micromeshLayoutInitStandard(&settings.decompressedLayout,
                                                                    microutils::getMicromeshLayoutType(blockLevel.layout));
  assert(result == micromesh::Result::eSuccess);

  result = micromesh::micromeshDecompressDisplacementBlock(&settings, scratchDataSize, scratchData, blockValues, blockDecoded, nullptr);
  assert(result == micromesh::Result::eSuccess);

  // write values into uncompressed data
  for(uint32_t v = 0; v < uint32_t(blockLevel.coordinates.size()); v++)
  {
    baryutils::BaryWUV_uint16 blockWUV = blockLevel.coordinates[v];
    bary::BaryUV_uint16       blockUV  = {blockWUV.u, blockWUV.v};
    bary::BaryUV_uint16       baseUV   = bary::baryBlockTriangleLocalToBaseUV(blockSplit, blockUV);
    baryutils::BaryWUV_uint16 baseWUV  = baryutils::makeWUV(baseUV, baseLevel.subdivLevel);

    uint32_t baseIndex = baseLevel.getCoordIndex(baseWUV);
    assert(baseIndex != ~0U);
    baseUncompressed[baseIndex] = blockDecoded[v];
  }

  return result;
}

void ThreadedTriangleDecoder::init(bary::Format format, bary::ValueLayout layout, uint32_t maxSubdivLevel, uint32_t numThreads)
{
  assert(format == bary::Format::eDispC1_r11_unorm_block);
  splitTable.init(format, maxSubdivLevel);
  levelsMap.initialize(layout, maxSubdivLevel);
  uint32_t maxBlockSubdivLevel = bary::baryBlockFormatDispC1GetMaxSubdivLevel();
  numDecoderValuesMax      = bary::baryValueFrequencyGetCount(bary::ValueFrequency::ePerVertex, maxBlockSubdivLevel);
  numUncompressedValuesMax = bary::baryValueFrequencyGetCount(bary::ValueFrequency::ePerVertex, maxSubdivLevel);

  micromesh::DisplacementBlock_settings settings;
  settings.compressedFormat   = microutils::getMicromeshFormat(format);
  settings.decompressedFormat = micromesh::Format::eR11_unorm_pack16;
  settings.subdivLevel        = maxSubdivLevel;
  scratchDataSize             = micromesh::micromeshGetDisplacementBlockScratchSize(&settings);

  threadDecoderValues.resize(numThreads * numDecoderValuesMax);
  threadUncompressedValues.resize(numThreads * numUncompressedValuesMax);
  threadScratchData.resize(numThreads * scratchDataSize);
}

uint16_t* ThreadedTriangleDecoder::tempThreadDecode(uint32_t               threadIdx,
                                                    const bary::BasicView& basicCompressed,
                                                    uint32_t               groupIndex,
                                                    uint32_t               triangleIndex,
                                                    uint32_t&              out_valueCount)
{
  uint16_t* blockDecoded    = threadDecoderValues.data() + (threadIdx * numDecoderValuesMax);
  uint16_t* triUncompressed = threadUncompressedValues.data() + (threadIdx * numUncompressedValuesMax);
  uint8_t*  scratchData     = threadScratchData.data() + (threadIdx * scratchDataSize);

  assert(groupIndex < basicCompressed.groupsCount);
  assert(triangleIndex < basicCompressed.trianglesCount);

  const bary::Group&    baryGroup  = basicCompressed.groups[groupIndex];
  const bary::Triangle& baryTri    = basicCompressed.triangles[triangleIndex];
  uint32_t              baseSubdiv = baryTri.subdivLevel;

  // iterate over each block and decompress it
  const baryutils::BarySplitTable::Entry& splitConfig = splitTable.get(baryTri.blockFormatDispC1, baseSubdiv);
  uint32_t blockSubdiv = bary::baryBlockFormatDispC1GetSubdivLevel(baryTri.blockFormatDispC1);

  blockSubdiv = std::min(blockSubdiv, baseSubdiv);

  const baryutils::BaryLevelsMap::Level& triLevel   = levelsMap.getLevel(baseSubdiv);
  const baryutils::BaryLevelsMap::Level& blockLevel = levelsMap.getLevel(blockSubdiv);

  for(uint32_t i = 0; i < splitConfig.getCount(); i++)
  {
    const uint8_t* blockValues = basicCompressed.values + size_t(baryGroup.valueFirst)
                                 + (baryTri.valuesOffset + splitConfig.tris[i].blockByteOffset);

    micromesh::Result result = decodeBlockInto(blockValues, baryTri.blockFormatDispC1, &splitConfig.tris[i], blockLevel,
                                               triLevel, blockDecoded, triUncompressed, scratchDataSize, scratchData);
    assert(result == micromesh::Result::eSuccess);
  }

  out_valueCount = bary::baryValueFrequencyGetCount(basicCompressed.valuesInfo->valueFrequency, baseSubdiv);
  return triUncompressed;
}

micromesh::Result baryMiscDataSetupMips(baryutils::BaryMiscData&       mip,
                                        micromesh::OpContext           ctx,
                                        const bary::BasicView&         basicCompressed,
                                        const UncompressedMipSettings& settings)
{
  mip.uncompressedMipsInfo.elementByteAlignment = 4;
  mip.uncompressedMipsInfo.elementByteSize      = 1;
  mip.uncompressedMipsInfo.elementFormat        = bary::Format::eR11_unorm_packed_align32;

  mip.groupUncompressedMips.resize(basicCompressed.groupsCount);

  // iterate all triangles and compute mip Offsets
  bary::TriangleUncompressedMip zeroMip;
  zeroMip.mipOffset   = ~0;
  zeroMip.subdivLevel = 0;
  mip.triangleUncompressedMips.resize(basicCompressed.trianglesCount, zeroMip);


  const uint32_t numMipValues = bary::baryValueFrequencyGetCount(bary::ValueFrequency::ePerVertex, settings.mipSubdiv);
  const uint32_t mipEntrySize = ((numMipValues * 11 + 31) / 32) * uint32_t(sizeof(uint32_t));  // Number of bytes in a triangle's mip entry

  uint32_t mipTotalBytes = 0;  // Total number of bytes of mip data (#elements == #bytes, since elementByteSize is 1)
  for(uint32_t g = 0; g < basicCompressed.groupsCount; g++)
  {
    const bary::Group&          groupCompressed = basicCompressed.groups[g];
    bary::GroupUncompressedMip& groupMip        = mip.groupUncompressedMips[g];

    groupMip.mipFirst = mipTotalBytes;

    for(uint32_t i = 0; i < groupCompressed.triangleCount; i++)
    {
      uint32_t triGlobalIdx = i + groupCompressed.triangleFirst;
      assert(triGlobalIdx < basicCompressed.trianglesCount);
      const bary::Triangle& tri = basicCompressed.triangles[triGlobalIdx];
      if((settings.skipBlockFormatBits & (1u << uint32_t(tri.blockFormatDispC1))) == 0 && tri.subdivLevel >= settings.minSubdiv)
      {
        mip.triangleUncompressedMips[triGlobalIdx].mipOffset   = mipTotalBytes - groupMip.mipFirst;
        mip.triangleUncompressedMips[triGlobalIdx].subdivLevel = std::min(settings.mipSubdiv, uint32_t(tri.subdivLevel));
        mipTotalBytes += mipEntrySize;
      }
    }

    groupMip.mipCount = mipTotalBytes - groupMip.mipFirst;
  }

  // add dummy 32bit so mip.uncompressedMips is not empty
  // (work around for the shortcut utility functions that would omit a property if this was nullptr)
  mip.uncompressedMipsInfo.elementCount = std::max(mipTotalBytes, 4u);
  mip.uncompressedMips.resize(mip.uncompressedMipsInfo.elementCount, 0);

  uint32_t numThreads = micromesh::micromeshOpContextGetConfig(ctx).threadCount;

  // second pass is parallel and computes the actual mip data
  for(uint32_t g = 0; g < basicCompressed.groupsCount; g++)
  {
    const bary::Group& baryGroup      = basicCompressed.groups[g];
    uint8_t*           mipGroupValues = (mip.uncompressedMips.data() + (mip.groupUncompressedMips[g].mipFirst));
    const uint8_t* blockGroupValues = basicCompressed.values + (baryGroup.valueFirst * basicCompressed.valuesInfo->valueByteSize);

    struct Payload
    {
      uint32_t                 numMipValues;
      const bary::BasicView&   basicCompressed;
      const bary::Group&       baryGroup;
      uint32_t                 baryGroupIndex;
      baryutils::BaryMiscData& mip;
      uint8_t*                 mipGroupValues;
      ThreadedTriangleDecoder  threadedDecoder;
    } payload = {numMipValues, basicCompressed, baryGroup, g, mip, mipGroupValues};

    payload.threadedDecoder.init(bary::Format::eDispC1_r11_unorm_block, bary::ValueLayout::eTriangleBirdCurve,
                                 baryGroup.maxSubdivLevel, numThreads);

    auto fnProcess = [](uint64_t triIdx, uint32_t threadIdx, void* userData) {
      Payload& payload = *reinterpret_cast<Payload*>(userData);

      uint64_t triGlobalIdx = triIdx + payload.baryGroup.triangleFirst;
      uint32_t mipOffset    = payload.mip.triangleUncompressedMips[triGlobalIdx].mipOffset;

      if(mipOffset == ~0U)
        return;

      uint32_t        triUncompressedCount;
      const uint16_t* triUncompressed =
          payload.threadedDecoder.tempThreadDecode(threadIdx, payload.basicCompressed, payload.baryGroupIndex,
                                                   uint32_t(triGlobalIdx), triUncompressedCount);

      uint8_t* mipValues = payload.mipGroupValues + mipOffset;
      for(uint32_t v = 0; v < payload.numMipValues; v++)
      {
        micromesh::packedWriteR11UnormPackedAlign32(mipValues, v, triUncompressed[v]);
      }
    };

    micromesh::OpDistributeWork_input input;
    input.pfnGenericSingleWorkload = fnProcess;
    input.userData                 = &payload;

    micromesh::micromeshOpDistributeWork(ctx, &input, baryGroup.triangleCount);
  }
  return micromesh::Result::eSuccess;
}


micromesh::Result baryBasicDataCompressedUpdateTriangleMinMaxs(baryutils::BaryBasicData& bary, micromesh::OpContext ctx)
{
  if(bary.valuesInfo.valueFormat != bary::Format::eDispC1_r11_unorm_block)
  {
    return micromesh::Result::eInvalidFormat;
  }

  bary::BasicView basicCompressed = bary.getView();

  bary.triangleMinMaxsInfo.elementByteAlignment = 4;
  bary.triangleMinMaxsInfo.elementByteSize      = sizeof(uint16_t);
  bary.triangleMinMaxsInfo.elementCount         = uint32_t(bary.triangles.size() * 2);
  bary.triangleMinMaxsInfo.elementFormat        = bary::Format::eR11_unorm_pack16;

  bary.triangleMinMaxs.resize(bary.triangleMinMaxsInfo.elementCount * bary.triangleMinMaxsInfo.elementByteSize);

  uint32_t numThreads = micromesh::micromeshOpContextGetConfig(ctx).threadCount;

  // second pass is parallel and computes the actual mip data
  for(uint32_t g = 0; g < basicCompressed.groupsCount; g++)
  {
    const bary::Group& baryGroup = basicCompressed.groups[g];
    const uint8_t* blockGroupValues = basicCompressed.values + (baryGroup.valueFirst * basicCompressed.valuesInfo->valueByteSize);

    struct Payload
    {
      const bary::BasicView&  basicCompressed;
      const bary::Group&      baryGroup;
      uint32_t                baryGroupIndex;
      uint16_t*               triangleMinMaxs;
      ThreadedTriangleDecoder threadedDecoder;
    } payload = {basicCompressed, baryGroup, g, reinterpret_cast<uint16_t*>(bary.triangleMinMaxs.data())};

    payload.threadedDecoder.init(bary::Format::eDispC1_r11_unorm_block, bary::ValueLayout::eTriangleBirdCurve,
                                 baryGroup.maxSubdivLevel, numThreads);

    auto fnProcess = [](uint64_t triIdx, uint32_t threadIdx, void* userData) {
      Payload& payload = *reinterpret_cast<Payload*>(userData);

      uint64_t triGlobalIdx = triIdx + payload.baryGroup.triangleFirst;

      uint32_t        triUncompressedCount;
      const uint16_t* triUncompressed =
          payload.threadedDecoder.tempThreadDecode(threadIdx, payload.basicCompressed, payload.baryGroupIndex,
                                                   uint32_t(triGlobalIdx), triUncompressedCount);

      uint16_t triMin = 0x7FF;
      uint16_t triMax = 0;
      for(uint32_t i = 0; i < triUncompressedCount; i++)
      {
        triMax = std::max(triMax, triUncompressed[i]);
        triMin = std::min(triMin, triUncompressed[i]);
      }

      payload.triangleMinMaxs[triGlobalIdx * 2 + 0] = triMin;
      payload.triangleMinMaxs[triGlobalIdx * 2 + 1] = triMax;
    };

    micromesh::OpDistributeWork_input input;
    input.pfnGenericSingleWorkload = fnProcess;
    input.userData                 = &payload;

    micromesh::micromeshOpDistributeWork(ctx, &input, baryGroup.triangleCount);
  }

  return micromesh::Result::eSuccess;
}

}  // namespace microutils