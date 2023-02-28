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
#include <cassert>
#include <algorithm>
#include <micromesh/micromesh_utils.h>
#include <micromesh/micromesh_operations.h>
#include <microutils/microutils.hpp>

#include <nvh/nvprint.hpp>
#include <thread>

namespace microutils {

//////////////////////////////////////////////////////////////////////////

static void defaultCallback(micromesh::MessageSeverity severity, const char* message, uint32_t threadIndex, const void* userData)
{
  if(severity == micromesh::MessageSeverity::eInfo)
  {
    LOGI("umesh INFO: (t%u) %s\n", threadIndex, message);
  }
  else if(severity == micromesh::MessageSeverity::eWarning)
  {
    LOGW("umesh WARNING: (t%u) %s\n", threadIndex, message);
  }
  else if(severity == micromesh::MessageSeverity::eError)
  {
    LOGE("umesh ERROR: (t%u) %s\n", threadIndex, message);
  }
}

micromesh::MessageCallbackInfo makeDefaultMessageCallback()
{
  micromesh::MessageCallbackInfo messenger{defaultCallback};
  return messenger;
}

//////////////////////////////////////////////////////////////////////////

inline void baryGroupPropsToMicromap(const bary::Group& group, micromesh::Micromap& micromap)
{
  micromap.valueFloatExpansion.bias[0]  = group.floatBias.r;
  micromap.valueFloatExpansion.bias[1]  = group.floatBias.g;
  micromap.valueFloatExpansion.bias[2]  = group.floatBias.b;
  micromap.valueFloatExpansion.bias[3]  = group.floatBias.a;
  micromap.valueFloatExpansion.scale[0] = group.floatScale.r;
  micromap.valueFloatExpansion.scale[1] = group.floatScale.g;
  micromap.valueFloatExpansion.scale[2] = group.floatScale.b;
  micromap.valueFloatExpansion.scale[3] = group.floatScale.a;
  micromap.minSubdivLevel               = group.minSubdivLevel;
  micromap.maxSubdivLevel               = group.maxSubdivLevel;
}

inline void baryGroupPropsToMicromap(const bary::Group& group, micromesh::MicromapPacked& micromap)
{
  micromap.valueFloatExpansion.bias[0]  = group.floatBias.r;
  micromap.valueFloatExpansion.bias[1]  = group.floatBias.g;
  micromap.valueFloatExpansion.bias[2]  = group.floatBias.b;
  micromap.valueFloatExpansion.bias[3]  = group.floatBias.a;
  micromap.valueFloatExpansion.scale[0] = group.floatScale.r;
  micromap.valueFloatExpansion.scale[1] = group.floatScale.g;
  micromap.valueFloatExpansion.scale[2] = group.floatScale.b;
  micromap.valueFloatExpansion.scale[3] = group.floatScale.a;
  micromap.minSubdivLevel               = group.minSubdivLevel;
  micromap.maxSubdivLevel               = group.maxSubdivLevel;
}

inline void baryGroupPropsToMicromap(const bary::Group& group, micromesh::MicromapCompressed& micromap)
{
  micromap.valueFloatExpansion.bias[0]  = group.floatBias.r;
  micromap.valueFloatExpansion.bias[1]  = group.floatBias.g;
  micromap.valueFloatExpansion.bias[2]  = group.floatBias.b;
  micromap.valueFloatExpansion.bias[3]  = group.floatBias.a;
  micromap.valueFloatExpansion.scale[0] = group.floatScale.r;
  micromap.valueFloatExpansion.scale[1] = group.floatScale.g;
  micromap.valueFloatExpansion.scale[2] = group.floatScale.b;
  micromap.valueFloatExpansion.scale[3] = group.floatScale.a;
  micromap.minSubdivLevel               = group.minSubdivLevel;
  micromap.maxSubdivLevel               = group.maxSubdivLevel;
}

inline void baryTrianglesToMicromap(uint32_t triangleCount, bary::Triangle* triangles, micromesh::Micromap& micromap)
{
  micromap.triangleSubdivLevels.byteStride = uint32_t(sizeof(bary::Triangle));
  micromap.triangleSubdivLevels.count      = triangleCount;
  micromap.triangleSubdivLevels.data       = &triangles[0].subdivLevel;
  micromap.triangleSubdivLevels.format     = micromap.triangleSubdivLevels.s_format;

  micromap.triangleValueIndexOffsets.byteStride = uint32_t(sizeof(bary::Triangle));
  micromap.triangleValueIndexOffsets.count      = triangleCount;
  micromap.triangleValueIndexOffsets.data       = &triangles[0].valuesOffset;
  micromap.triangleValueIndexOffsets.format     = micromap.triangleValueIndexOffsets.s_format;
}

inline void baryTrianglesToMicromap(uint32_t triangleCount, bary::Triangle* triangles, micromesh::MicromapPacked& micromap)
{
  micromap.triangleSubdivLevels.byteStride = uint32_t(sizeof(bary::Triangle));
  micromap.triangleSubdivLevels.count      = triangleCount;
  micromap.triangleSubdivLevels.data       = &triangles[0].subdivLevel;
  micromap.triangleSubdivLevels.format     = micromap.triangleSubdivLevels.s_format;

  micromap.triangleValueByteOffsets.byteStride = uint32_t(sizeof(bary::Triangle));
  micromap.triangleValueByteOffsets.count      = triangleCount;
  micromap.triangleValueByteOffsets.data       = &triangles[0].valuesOffset;
  micromap.triangleValueByteOffsets.format     = micromap.triangleValueByteOffsets.s_format;
}

inline void baryTrianglesToMicromap(uint32_t triangleCount, bary::Triangle* triangles, micromesh::MicromapCompressed& micromap)
{
  micromap.triangleSubdivLevels.byteStride = uint32_t(sizeof(bary::Triangle));
  micromap.triangleSubdivLevels.count      = triangleCount;
  micromap.triangleSubdivLevels.data       = &triangles[0].subdivLevel;
  micromap.triangleSubdivLevels.format     = micromap.triangleSubdivLevels.s_format;

  micromap.triangleValueByteOffsets.byteStride = uint32_t(sizeof(bary::Triangle));
  micromap.triangleValueByteOffsets.count      = triangleCount;
  micromap.triangleValueByteOffsets.data       = &triangles[0].valuesOffset;
  micromap.triangleValueByteOffsets.format     = micromap.triangleValueByteOffsets.s_format;

  micromap.triangleBlockFormats.byteStride = uint32_t(sizeof(bary::Triangle));
  micromap.triangleBlockFormats.count      = triangleCount;
  micromap.triangleBlockFormats.data       = &triangles[0].blockFormat;
  micromap.triangleBlockFormats.format     = micromap.triangleBlockFormats.s_format;
}

inline void baryValuesToMicromap(const bary::ValuesInfo& valueInfo, uint32_t valueCount, uint8_t* values, micromesh::Micromap& micromap)
{
  micromap.values.byteStride = valueInfo.valueByteSize;
  micromap.values.format     = getMicromeshFormat(valueInfo.valueFormat);
  micromap.values.count      = valueCount;
  micromap.values.data       = values;

  micromap.frequency                       = getMicromeshFrequncy(valueInfo.valueFrequency);
  micromesh::StandardLayoutType layoutType = getMicromeshLayoutType(valueInfo.valueLayout);
  assert(layoutType != micromesh::StandardLayoutType::eUnknown);
  micromesh::micromeshLayoutInitStandard(&micromap.layout, layoutType);
}

inline void baryValuesToMicromap(const bary::ValuesInfo& valueInfo, uint32_t valueCount, uint8_t* values, micromesh::MicromapPacked& micromap)
{
  micromap.values.byteStride = valueInfo.valueByteSize;
  micromap.values.format     = getMicromeshFormat(valueInfo.valueFormat);
  micromap.values.count      = valueCount;
  micromap.values.data       = values;

  micromap.frequency = getMicromeshFrequncy(valueInfo.valueFrequency);
  micromesh::micromeshLayoutInitStandard(&micromap.layout, getMicromeshLayoutType(valueInfo.valueLayout));
}

inline void baryValuesToMicromap(const bary::ValuesInfo& valueInfo, uint32_t valueCount, uint8_t* values, micromesh::MicromapCompressed& micromap)
{
  micromap.values.byteStride = valueInfo.valueByteSize;
  micromap.values.format     = getMicromeshFormat(valueInfo.valueFormat);
  micromap.values.count      = valueCount;
  micromap.values.data       = values;
}

micromesh::MicromapType micromapTypeFromBasic(const bary::BasicView& basic)
{
  return micromesh::micromeshFormatGetMicromapType(getMicromeshFormat(basic.valuesInfo->valueFormat));
}

micromesh::Micromap micromapFromBasicGroup(const bary::BasicView& basic, uint32_t groupIndex)
{
  assert(groupIndex < basic.groupsCount);
  assert(getMicromeshLayoutType(basic.valuesInfo->valueLayout) != micromesh::StandardLayoutType::eUnknown
         && micromapTypeFromBasic(basic) == micromesh::MicromapType::eUncompressed);

  const bary::Group& group = basic.groups[groupIndex];

  micromesh::Micromap uncompressed;
  baryTrianglesToMicromap(group.triangleCount, const_cast<bary::Triangle*>(basic.triangles + group.triangleFirst), uncompressed);
  baryValuesToMicromap(*basic.valuesInfo, group.valueCount,
                       const_cast<uint8_t*>(basic.values + (group.valueFirst * basic.valuesInfo->valueByteSize)), uncompressed);
  baryGroupPropsToMicromap(group, uncompressed);
  return uncompressed;
}
micromesh::MicromapPacked micromapPackedFromBasicGroup(const bary::BasicView& basic, uint32_t groupIndex)
{
  assert(groupIndex < basic.groupsCount);
  assert(getMicromeshLayoutType(basic.valuesInfo->valueLayout) != micromesh::StandardLayoutType::eUnknown
         && micromapTypeFromBasic(basic) == micromesh::MicromapType::ePacked);

  const bary::Group& group = basic.groups[groupIndex];

  micromesh::MicromapPacked packed;
  baryTrianglesToMicromap(group.triangleCount, const_cast<bary::Triangle*>(basic.triangles + group.triangleFirst), packed);
  baryValuesToMicromap(*basic.valuesInfo, group.valueCount,
                       const_cast<uint8_t*>(basic.values + (group.valueFirst * basic.valuesInfo->valueByteSize)), packed);
  baryGroupPropsToMicromap(group, packed);
  return packed;
}
micromesh::MicromapCompressed micromapCompressedFromBasicGroup(const bary::BasicView& basic, uint32_t groupIndex)
{
  assert(groupIndex < basic.groupsCount);
  assert(getMicromeshLayoutType(basic.valuesInfo->valueLayout) != micromesh::StandardLayoutType::eUnknown
         && micromapTypeFromBasic(basic) == micromesh::MicromapType::eCompressed);

  const bary::Group& group = basic.groups[groupIndex];

  micromesh::MicromapCompressed compressed;
  baryTrianglesToMicromap(group.triangleCount, const_cast<bary::Triangle*>(basic.triangles + group.triangleFirst), compressed);
  baryValuesToMicromap(*basic.valuesInfo, group.valueCount,
                       const_cast<uint8_t*>(basic.values + (group.valueFirst * basic.valuesInfo->valueByteSize)), compressed);
  baryGroupPropsToMicromap(group, compressed);
  return compressed;
}

bary::Result baryBasicViewToMicromap(const bary::BasicView& basic, uint32_t groupIndex, micromesh::MicromapGeneric& micromap)
{
  if(!basic.groups || !basic.triangles || !basic.valuesInfo || !basic.values)
  {
    return bary::Result::eErrorMissingProperty;
  }
  if(groupIndex >= basic.groupsCount)
  {
    return bary::Result::eErrorIndex;
  }
  if(getMicromeshLayoutType(basic.valuesInfo->valueLayout) == micromesh::StandardLayoutType::eUnknown)
  {
    return bary::Result::eErrorValue;
  }

  const bary::Group& group = basic.groups[groupIndex];
  if(group.triangleFirst >= basic.trianglesCount || (basic.trianglesCount - group.triangleFirst) < group.triangleCount)
  {
    return bary::Result::eErrorRange;
  }
  if(group.valueFirst >= basic.valuesInfo->valueCount || (basic.valuesInfo->valueCount - group.valueFirst) < group.valueCount)
  {
    return bary::Result::eErrorRange;
  }

  micromap.type = micromapTypeFromBasic(basic);

  switch(micromap.type)
  {
    case micromesh::MicromapType::eUncompressed:
      micromap.uncompressed = micromapFromBasicGroup(basic, groupIndex);
      break;
    case micromesh::MicromapType::ePacked:
      micromap.packed = micromapPackedFromBasicGroup(basic, groupIndex);
      break;
    case micromesh::MicromapType::eCompressed:
      micromap.compressed = micromapCompressedFromBasicGroup(basic, groupIndex);
      break;
    default:
      return bary::Result::eErrorFormat;
  }

  return bary::Result::eSuccess;
}

bary::Result baryBasicViewToMinMaxs(const bary::BasicView& basic, uint32_t groupIndex, micromesh::ArrayInfo& arrayInfo)
{
  if(!basic.groups || !basic.triangleMinMaxsInfo || !basic.triangleMinMaxs)
  {
    return bary::Result::eErrorMissingProperty;
  }
  if(groupIndex >= basic.groupsCount)
  {
    return bary::Result::eErrorIndex;
  }

  const bary::Group& group = basic.groups[groupIndex];

  arrayInfo.byteStride = basic.triangleMinMaxsInfo->elementByteSize;
  arrayInfo.format     = getMicromeshFormat(basic.triangleMinMaxsInfo->elementFormat);
  arrayInfo.count      = group.triangleCount * 2;
  arrayInfo.data =
      const_cast<uint8_t*>(basic.triangleMinMaxs + (group.triangleFirst * 2 * basic.triangleMinMaxsInfo->elementByteSize));

  return bary::Result::eSuccess;
}


static_assert(sizeof(micromesh::BlockFormatUsage) == sizeof(bary::HistogramEntry),
              "bary::HistogramEntry mismatches micromesh::MicromapBlockFormatUsage");
static_assert(sizeof(micromesh::BlockFormatUsage::subdivLevel) == sizeof(bary::HistogramEntry::subdivLevel),
              "bary::HistogramEntry mismatches micromesh::MicromapBlockFormatUsage");
static_assert(sizeof(micromesh::BlockFormatUsage::count) == sizeof(bary::HistogramEntry::count),
              "bary::HistogramEntry mismatches micromesh::MicromapBlockFormatUsage");
static_assert(sizeof(micromesh::BlockFormatUsage::blockFormat) == sizeof(bary::HistogramEntry::blockFormat),
              "bary::HistogramEntry mismatches micromesh::MicromapBlockFormatUsage");

static_assert(offsetof(micromesh::BlockFormatUsage, subdivLevel) == offsetof(bary::HistogramEntry, subdivLevel),
              "bary::HistogramEntry mismatches micromesh::MicromapBlockFormatUsage");
static_assert(offsetof(micromesh::BlockFormatUsage, count) == offsetof(bary::HistogramEntry, count),
              "bary::HistogramEntry mismatches micromesh::MicromapBlockFormatUsage");
static_assert(offsetof(micromesh::BlockFormatUsage, blockFormat) == offsetof(bary::HistogramEntry, blockFormat),
              "bary::HistogramEntry mismatches micromesh::MicromapBlockFormatUsage");

bary::Result baryBasicViewToBlockFormatUsage(bary::BasicView& basic, uint32_t groupIndex, micromesh::MicromapBlockFormatUsage& mapUsage)
{
  if(!basic.groupHistogramRanges || !basic.histogramEntries)
  {
    return bary::Result::eErrorMissingProperty;
  }
  if(groupIndex >= basic.groupHistogramRangesCount)
  {
    return bary::Result::eErrorIndex;
  }

  const bary::GroupHistogramRange& group = basic.groupHistogramRanges[groupIndex];

  mapUsage.entriesCount = group.entryCount;
  mapUsage.entries      = const_cast<micromesh::BlockFormatUsage*>(
      reinterpret_cast<const micromesh::BlockFormatUsage*>(&basic.histogramEntries[group.entryFirst]));


  return bary::Result::eSuccess;
}

bary::HistogramEntry getBaryHistogramEntry(micromesh::BlockFormatUsage microBlockFormatUsage)
{
  bary::HistogramEntry out;
  memcpy(&out, &microBlockFormatUsage, sizeof(out));
  return out;
}
bary::MeshHistogramEntry getBaryMeshHistogramEntry(micromesh::BlockFormatUsage microBlockFormatUsage)
{
  bary::MeshHistogramEntry out;
  memcpy(&out, &microBlockFormatUsage, sizeof(out));
  return out;
}

micromesh::BlockFormatUsage getMicromeshBlockFormatUsage(bary::HistogramEntry baryHistoEntry)
{
  micromesh::BlockFormatUsage out;
  memcpy(&out, &baryHistoEntry, sizeof(out));
  return out;
}

//////////////////////////////////////////////////////////////////////////

// returns eUndefined if not possible
bary::Format getBaryFormat(micromesh::Format microFormat)
{

#define HANDLE_CASE(_e)                                                                                                \
  case micromesh::Format::_e:                                                                                          \
    return bary::Format::_e;

  switch(microFormat)
  {
    HANDLE_CASE(eUndefined);
    HANDLE_CASE(eR8_unorm);
    HANDLE_CASE(eR8_snorm);
    HANDLE_CASE(eR8_uint);
    HANDLE_CASE(eR8_sint);
    HANDLE_CASE(eRG8_unorm);
    HANDLE_CASE(eRG8_snorm);
    HANDLE_CASE(eRG8_uint);
    HANDLE_CASE(eRG8_sint);
    HANDLE_CASE(eRGB8_unorm);
    HANDLE_CASE(eRGB8_snorm);
    HANDLE_CASE(eRGB8_uint);
    HANDLE_CASE(eRGB8_sint);
    HANDLE_CASE(eRGBA8_unorm);
    HANDLE_CASE(eRGBA8_snorm);
    HANDLE_CASE(eRGBA8_uint);
    HANDLE_CASE(eRGBA8_sint);
    HANDLE_CASE(eR16_unorm);
    HANDLE_CASE(eR16_snorm);
    HANDLE_CASE(eR16_uint);
    HANDLE_CASE(eR16_sint);
    HANDLE_CASE(eR16_sfloat);
    HANDLE_CASE(eRG16_unorm);
    HANDLE_CASE(eRG16_snorm);
    HANDLE_CASE(eRG16_uint);
    HANDLE_CASE(eRG16_sint);
    HANDLE_CASE(eRG16_sfloat);
    HANDLE_CASE(eRGB16_unorm);
    HANDLE_CASE(eRGB16_snorm);
    HANDLE_CASE(eRGB16_uint);
    HANDLE_CASE(eRGB16_sint);
    HANDLE_CASE(eRGB16_sfloat);
    HANDLE_CASE(eRGBA16_unorm);
    HANDLE_CASE(eRGBA16_snorm);
    HANDLE_CASE(eRGBA16_uint);
    HANDLE_CASE(eRGBA16_sint);
    HANDLE_CASE(eRGBA16_sfloat);
    HANDLE_CASE(eR32_uint);
    HANDLE_CASE(eR32_sint);
    HANDLE_CASE(eR32_sfloat);
    HANDLE_CASE(eRG32_uint);
    HANDLE_CASE(eRG32_sint);
    HANDLE_CASE(eRG32_sfloat);
    HANDLE_CASE(eRGB32_uint);
    HANDLE_CASE(eRGB32_sint);
    HANDLE_CASE(eRGB32_sfloat);
    HANDLE_CASE(eRGBA32_uint);
    HANDLE_CASE(eRGBA32_sint);
    HANDLE_CASE(eRGBA32_sfloat);
    HANDLE_CASE(eR64_uint);
    HANDLE_CASE(eR64_sint);
    HANDLE_CASE(eR64_sfloat);
    HANDLE_CASE(eRG64_uint);
    HANDLE_CASE(eRG64_sint);
    HANDLE_CASE(eRG64_sfloat);
    HANDLE_CASE(eRGB64_uint);
    HANDLE_CASE(eRGB64_sint);
    HANDLE_CASE(eRGB64_sfloat);
    HANDLE_CASE(eRGBA64_uint);
    HANDLE_CASE(eRGBA64_sint);
    HANDLE_CASE(eRGBA64_sfloat);
    HANDLE_CASE(eOpaC1_rx_uint_block);
    HANDLE_CASE(eDispC1_r11_unorm_block);
    HANDLE_CASE(eR11_unorm_pack16);
    HANDLE_CASE(eR11_unorm_packed_align32);
    default:
      return bary::Format::eUndefined;
  }

#undef HANDLE_CASE
}

bary::ValueFrequency getBaryFrequency(micromesh::Frequency microFrequency)
{
  switch(microFrequency)
  {
    case micromesh::Frequency::ePerMicroVertex:
      return bary::ValueFrequency::ePerVertex;
    case micromesh::Frequency::ePerMicroTriangle:
      return bary::ValueFrequency::ePerTriangle;
    default:
      return bary::ValueFrequency::eUndefined;
  }
}
bary::ValueLayout getBaryValueLayout(micromesh::StandardLayoutType microStandardLayout)
{
  switch(microStandardLayout)
  {
    case micromesh::StandardLayoutType::eUmajor:
      return bary::ValueLayout::eTriangleUmajor;
    case micromesh::StandardLayoutType::eBirdCurve:
      return bary::ValueLayout::eTriangleBirdCurve;
    default:
      return bary::ValueLayout::eUndefined;
  }
}

bary::BlockFormatDispC1 getBaryBlockFormatDispC1(micromesh::BlockFormatDispC1 microBlockFormat)
{
  switch(microBlockFormat)
  {
    case micromesh::BlockFormatDispC1::eR11_unorm_lvl3_pack512:
      return bary::BlockFormatDispC1::eR11_unorm_lvl3_pack512;
    case micromesh::BlockFormatDispC1::eR11_unorm_lvl4_pack1024:
      return bary::BlockFormatDispC1::eR11_unorm_lvl4_pack1024;
    case micromesh::BlockFormatDispC1::eR11_unorm_lvl5_pack1024:
      return bary::BlockFormatDispC1::eR11_unorm_lvl5_pack1024;
    default:
      return bary::BlockFormatDispC1::eInvalid;
  }
}
bary::BlockFormatOpaC1 getBaryBlockFormatOpaC1(micromesh::BlockFormatOpaC1 microBlockFormat)
{
  switch(microBlockFormat)
  {
    case micromesh::BlockFormatOpaC1::eR1_uint_x8:
      return bary::BlockFormatOpaC1::eR1_uint_x8;
    case micromesh::BlockFormatOpaC1::eR2_uint_x4:
      return bary::BlockFormatOpaC1::eR2_uint_x4;
    default:
      return bary::BlockFormatOpaC1::eInvalid;
  }
}

// returns eUndefined if not possible
micromesh::Format getMicromeshFormat(bary::Format baryFormat)
{

#define HANDLE_CASE(_e)                                                                                                \
  case bary::Format::_e:                                                                                               \
    return micromesh::Format::_e;

  switch(baryFormat)
  {
    HANDLE_CASE(eUndefined);
    HANDLE_CASE(eR8_unorm);
    HANDLE_CASE(eR8_snorm);
    HANDLE_CASE(eR8_uint);
    HANDLE_CASE(eR8_sint);
    HANDLE_CASE(eRG8_unorm);
    HANDLE_CASE(eRG8_snorm);
    HANDLE_CASE(eRG8_uint);
    HANDLE_CASE(eRG8_sint);
    HANDLE_CASE(eRGB8_unorm);
    HANDLE_CASE(eRGB8_snorm);
    HANDLE_CASE(eRGB8_uint);
    HANDLE_CASE(eRGB8_sint);
    HANDLE_CASE(eRGBA8_unorm);
    HANDLE_CASE(eRGBA8_snorm);
    HANDLE_CASE(eRGBA8_uint);
    HANDLE_CASE(eRGBA8_sint);
    HANDLE_CASE(eR16_unorm);
    HANDLE_CASE(eR16_snorm);
    HANDLE_CASE(eR16_uint);
    HANDLE_CASE(eR16_sint);
    HANDLE_CASE(eR16_sfloat);
    HANDLE_CASE(eRG16_unorm);
    HANDLE_CASE(eRG16_snorm);
    HANDLE_CASE(eRG16_uint);
    HANDLE_CASE(eRG16_sint);
    HANDLE_CASE(eRG16_sfloat);
    HANDLE_CASE(eRGB16_unorm);
    HANDLE_CASE(eRGB16_snorm);
    HANDLE_CASE(eRGB16_uint);
    HANDLE_CASE(eRGB16_sint);
    HANDLE_CASE(eRGB16_sfloat);
    HANDLE_CASE(eRGBA16_unorm);
    HANDLE_CASE(eRGBA16_snorm);
    HANDLE_CASE(eRGBA16_uint);
    HANDLE_CASE(eRGBA16_sint);
    HANDLE_CASE(eRGBA16_sfloat);
    HANDLE_CASE(eR32_uint);
    HANDLE_CASE(eR32_sint);
    HANDLE_CASE(eR32_sfloat);
    HANDLE_CASE(eRG32_uint);
    HANDLE_CASE(eRG32_sint);
    HANDLE_CASE(eRG32_sfloat);
    HANDLE_CASE(eRGB32_uint);
    HANDLE_CASE(eRGB32_sint);
    HANDLE_CASE(eRGB32_sfloat);
    HANDLE_CASE(eRGBA32_uint);
    HANDLE_CASE(eRGBA32_sint);
    HANDLE_CASE(eRGBA32_sfloat);
    HANDLE_CASE(eR64_uint);
    HANDLE_CASE(eR64_sint);
    HANDLE_CASE(eR64_sfloat);
    HANDLE_CASE(eRG64_uint);
    HANDLE_CASE(eRG64_sint);
    HANDLE_CASE(eRG64_sfloat);
    HANDLE_CASE(eRGB64_uint);
    HANDLE_CASE(eRGB64_sint);
    HANDLE_CASE(eRGB64_sfloat);
    HANDLE_CASE(eRGBA64_uint);
    HANDLE_CASE(eRGBA64_sint);
    HANDLE_CASE(eRGBA64_sfloat);
    HANDLE_CASE(eOpaC1_rx_uint_block);
    HANDLE_CASE(eDispC1_r11_unorm_block);
    HANDLE_CASE(eR11_unorm_pack16);
    HANDLE_CASE(eR11_unorm_packed_align32);
    default:
      return micromesh::Format::eUndefined;
  }

#undef HANDLE_CASE
}
micromesh::Frequency getMicromeshFrequncy(bary::ValueFrequency baryFrequency)
{
  switch(baryFrequency)
  {
    case bary::ValueFrequency::ePerVertex:
      return micromesh::Frequency::ePerMicroVertex;
    case bary::ValueFrequency::ePerTriangle:
      return micromesh::Frequency::ePerMicroTriangle;
    default:
      assert(0 && "invalid bary::ValueFrequency");
      return micromesh::Frequency::ePerMicroVertex;
  }
}
micromesh::StandardLayoutType getMicromeshLayoutType(bary::ValueLayout baryLayout)
{
  switch(baryLayout)
  {
    case bary::ValueLayout::eTriangleUmajor:
      return micromesh::StandardLayoutType::eUmajor;
    case bary::ValueLayout::eTriangleBirdCurve:
      return micromesh::StandardLayoutType::eBirdCurve;
    default:
      return micromesh::StandardLayoutType::eUnknown;
  }
}

micromesh::BlockFormatDispC1 getMicromeshBlockFormatDispC1(bary::BlockFormatDispC1 microBlockFormat)
{
  switch(microBlockFormat)
  {
    case bary::BlockFormatDispC1::eR11_unorm_lvl3_pack512:
      return micromesh::BlockFormatDispC1::eR11_unorm_lvl3_pack512;
    case bary::BlockFormatDispC1::eR11_unorm_lvl4_pack1024:
      return micromesh::BlockFormatDispC1::eR11_unorm_lvl4_pack1024;
    case bary::BlockFormatDispC1::eR11_unorm_lvl5_pack1024:
      return micromesh::BlockFormatDispC1::eR11_unorm_lvl5_pack1024;
    default:
      return micromesh::BlockFormatDispC1::eInvalid;
  }
}
micromesh::BlockFormatOpaC1 getMicromeshBlockFormatOpaC1(bary::BlockFormatOpaC1 microBlockFormat)
{
  switch(microBlockFormat)
  {
    case bary::BlockFormatOpaC1::eR1_uint_x8:
      return micromesh::BlockFormatOpaC1::eR1_uint_x8;
    case bary::BlockFormatOpaC1::eR2_uint_x4:
      return micromesh::BlockFormatOpaC1::eR2_uint_x4;
    default:
      return micromesh::BlockFormatOpaC1::eInvalid;
  }
}

}  // namespace microutils
