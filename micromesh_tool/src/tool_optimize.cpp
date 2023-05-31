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

#include <tool_optimize.hpp>

#include <algorithm>
#include <filesystem>
#include <inputparser.hpp>
// To access meshopsContext->m_micromeshContext; TODO: can we add a function to avoid this?
#include <meshops_internal/meshops_context.h>
#include <nvh/parallel_work.hpp>
#include <tool_meshops_objects.hpp>
#include <micromesh/micromesh_utils.h>

namespace tool_optimize {

using namespace micromesh;
using namespace micromesh_tool;

using float4 = std::array<float, 4>;

constexpr uint16_t UNORM11_MASK = (1u << 11) - 1;

bary::ValueFloatVector float4ToBaryVector(const float4& f4)
{
  bary::ValueFloatVector v;
  v.r = f4[0];
  v.g = f4[1];
  v.b = f4[2];
  v.a = f4[3];
  return v;
}

bary::ValueFloatVector float4ToBaryVector(const float f4[4])
{
  bary::ValueFloatVector v;
  v.r = f4[0];
  v.g = f4[1];
  v.b = f4[2];
  v.a = f4[3];
  return v;
}

float4 baryVectorToFloat4(const bary::ValueFloatVector& v)
{
  float4 f4;
  f4[0] = v.r;
  f4[1] = v.g;
  f4[2] = v.b;
  f4[3] = v.a;
  return f4;
}

bool toolOptimizeParse(int argc, char** argv, ToolOptimizeArgs& args, std::ostream& os)
{
  bool              printHelp = false;
  CommandLineParser parser("optimize: Trims and compresses displacement data to save space and improve performance.");

  parser.addArgument({"--help"}, &printHelp, "Print help");

  parser.addArgument({"--trim", "-t"}, &args.trimSubdiv,
                     "Reduces the subdivision level of each triangle to at most this number. Removes "
                     "unused subdivision levels - like reducing the resolution of an image. (Default: 5)");

  parser.addArgument({"--psnr"}, &args.psnr,
                     "Minimum Peak Signal-to-Noise Ratio in decibels for lossy compression. 20 is very low quality; 30 "
                     "is low quality; 40 is normal quality; 50 is high quality. (Default: 40)");

  parser.addArgument({"--validate-edges"}, &args.validateEdges,
                     "Validates that the input and output displacements are watertight. (Default: true only in debug "
                     "builds; set using `--validate-edges true` or `--validate-edges false`)");

  if(!parser.parse(argc, argv, os) || printHelp)
  {
    parser.printHelp(printHelp ? std::cout : os);
    return false;
  }

  return true;
}

// Returns true if the bary satisfies the requirements for the optimizer;
// otherwise, prints an error and returns false.
bool checkBaryRequirements(const bary::ContentView& contentView, micromesh::Format& outFormat, micromesh::FormatInfo& outFormatInfo)
{
  const bary::BasicView& basic = contentView.basic;

  if(basic.valuesInfo == nullptr)
  {
    LOGE("Error: The bary had no value information!\n");
    return false;
  }

  if(basic.valuesInfo->valueFrequency != bary::ValueFrequency::ePerVertex)
  {
    LOGE("Error: The input's valueFrequency must be BARY_VALUE_FREQUENCY_VERTEX.\n");
    return false;
  }

  if(basic.valuesInfo->valueLayout == bary::ValueLayout::eUndefined)
  {
    LOGE("Error: The input's valueOrder was BARY_VALUE_ORDER_UNDEFINED.\n");
    return false;
  }

  outFormat                             = static_cast<Format>(basic.valuesInfo->valueFormat);
  const micromesh::Result fmtInfoResult = micromeshFormatGetInfo(outFormat, &outFormatInfo);
  if(fmtInfoResult != micromesh::Result::eSuccess)
  {
    LOGE("Failed to get format info for format %s.\n", micromeshGetFormatString(outFormat));
    return false;
  }

  if(outFormatInfo.byteSize != basic.valuesInfo->valueByteSize)
  {
    LOGE("Mismatch between micromesh format byte size (%u) and bary file byte size (%u).\n", outFormatInfo.byteSize,
         basic.valuesInfo->valueByteSize);
    return false;
  }

  // We support all uncompressed, unpacked formats, as well as these two formats:
  if(outFormatInfo.isCompressedOrPacked && outFormat != Format::eDispC1_r11_unorm_block && outFormat != Format::eR11_unorm_packed_align32)
  {
    LOGE("Format %s is not a supported input format.\n", micromeshGetFormatString(outFormat));
    return false;
  }

  // We could trim uncompressed files and output uncompressed files here.
  // We could also compress each channel independently, but currently
  // nothing will read that. We still have the structure below to handle
  // this, though.
  if(outFormatInfo.channelCount > 1)
  {
    LOGE("Trimming of multichannel bary files is currently not supported.\n");
    return false;
  }

  return true;
}

// Detects outliers using an approximation of the Grubbs test; returns false if
// there was a likely outlier. Uses a 99.9% confidence level.
bool grubbsTest(const float sum, const float squaredSum, const float minValue, const float maxValue, const size_t numValues, float& testStatistic, float& testStatisticLimit)
{
  // First off, the Grubbs test shouldn't be used for 6 or fewer points;
  // we treat this as if it always passes.
  if(numValues <= 6)
  {
    return true;
  }
  // If the squared sum overflowed, we definitely had an outlier.
  // Set variables so the printed message looks reasonable:
  if(isinf(squaredSum) || isnan(squaredSum))
  {
    testStatistic      = squaredSum;
    testStatisticLimit = testStatistic;
    return false;
  }
  // Compute the mean and sample standard deviation:
  const float n    = float(numValues);
  const float mean = sum / n;
  float       sd   = std::sqrt((squaredSum - sum * sum / n) / (n - 1));
  // If the sample standard deviation was 0, we pass!
  if(sd == 0)
  {
    return true;
  }
  // Now, Grubbs' test works on normal distributions, but our data is probably
  // uniformly distributed. Let's adjust the standard deviation to match
  // (this comes from the standard deviation of a uniform distribution, but
  // this is otherwise relatively ad-hoc!)
  sd *= std::sqrt(12.0f);
  // The Grubbs test statistic is max(abs(value_i - mean_value))/sd:
  testStatistic = std::max(maxValue - mean, mean - minValue) / sd;
  // For the full Grubbs test, we report an outlier if
  //
  //   G > ((n-1)/sqrt(n)) sqrt( t(a/(2n),n-2)^2 / (n - 2 + t(a/(2n),n-2)^2) )
  //
  // where a is our confidence level (0.001) and t(a/(2n),n-2) is the upper
  // two-sided critical value of the Students t distribution with n-2 degrees
  // of freedom and a significance level of a/(2n).
  // It turns out T := t(a/(2n),n-2) is a bit hard to compute! It's the value
  // such that
  //
  //   integral(pdf(x), {x, -T, T}) == 1-a/(2n),
  //
  // where pdf is the probability density function of the t distribution with
  // n-2 degrees of freedom. Luckily, the t distribution converges quickly to
  // a Gaussian, so we can approximate it using that. For a Gaussian,
  //
  //   integral(pdf(x), x) == 1/2 + erf(x/sqrt(2))/2
  //
  // so
  //
  //   erf(T/sqrt(2))/2 - erf(-T/sqrt(2))/2 == 1-a/(2n)
  //   erf(T/sqrt(2)) == 1-a/(2n)   (since erf is an odd function)
  //   T/sqrt(2) == erfinv(1-a/(2n))
  //   T = sqrt(2) * erfinv(1-a/(2n))
  //
  // Let's do one more approximation here! Giles' "Approximating the `erfinv`
  // function" from GPU Gems gives that when y = a/(2n) is small,
  //
  //   erfinv(1-a/(2n)) ~=    sqrt(-log(1-(1-y)^2))
  //                       == sqrt(-log(y(2-y)))
  //
  // So that gives us our approximation:
  const float a      = 0.001f;
  const float y      = a / (2.0f * n);
  const float t      = std::sqrt(-2.0f * std::log(y * (2.0f - y)));
  const float t2     = t * t;
  testStatisticLimit = (n - 1) * std::sqrt(t2 / ((n - 2 + t2) * n));
  return (testStatistic <= testStatisticLimit);
}

// Converts a series of bytes in a given format to floating-point values.
bool toFloats(const MessageCallbackInfo& messageCallback,
              const Format&              format,
              const FormatInfo&          formatInfo,
              const uint8_t*             triangleQuantizedValues,
              const uint64_t             numValues,
              const uint16_t             subdivLevel,
              const uint16_t             blockFormatIfCompressed,
              float*                     output)
{
  ArrayInfo triangleInput{};
  triangleInput.format     = format;
  triangleInput.data       = (void*)triangleQuantizedValues;
  triangleInput.count      = numValues;
  triangleInput.byteStride = formatInfo.byteSize;
  MicromapValueFloatExpansion triangleInputExpansion_identity;

  ArrayInfo triangleFloat{};
  switch(formatInfo.channelCount)
  {
    case 1:
      triangleFloat.format = Format::eR32_sfloat;
      break;
    case 2:
      triangleFloat.format = Format::eRG32_sfloat;
      break;
    case 3:
      triangleFloat.format = Format::eRGB32_sfloat;
      break;
    case 4:
      triangleFloat.format = Format::eRGBA32_sfloat;
      break;
  }
  triangleFloat.data       = output;
  triangleFloat.count      = triangleInput.count;
  triangleFloat.byteStride = sizeof(float);
  MicromapValueFloatExpansion triangleFloatExpansion_identity;

  micromesh::Result result;
  if(format == Format::eDispC1_r11_unorm_block)
  {
    // Decompress the triangle's blocks and assemble them.
    // This is a bit of a hack, to use micromeshOpDecompressDisplacement
    // within a parallel context: we create a 1-thread OpContext, and
    // a 1-triangle Micromap.
    micromesh::OpConfig singleThreadedConfig;
    singleThreadedConfig.threadCount = 1;
    ScopedOpContext singleThreadedCtx(singleThreadedConfig, messageCallback);

    MicromapCompressed mmInput{};
    mmInput.values                         = triangleInput;
    mmInput.valueFloatExpansion            = triangleFloatExpansion_identity;
    mmInput.minSubdivLevel                 = subdivLevel;
    mmInput.maxSubdivLevel                 = subdivLevel;
    mmInput.triangleSubdivLevels.data      = (void*)&subdivLevel;
    mmInput.triangleSubdivLevels.count     = 1;
    const uint32_t zero                    = 0;
    mmInput.triangleValueByteOffsets.data  = (void*)&zero;
    mmInput.triangleValueByteOffsets.count = 1;
    mmInput.triangleBlockFormats.data      = (void*)&blockFormatIfCompressed;
    mmInput.triangleBlockFormats.count     = 1;

    // We use an intermediate UNORM11_PACK16 buffer. This does mean that compressed
    // input can get re-scaled, which isn't ideal, but it works for now.
    Micromap mmIntermediate{};
    mmIntermediate.values.format = Format::eR11_unorm_pack16;
    micromeshLayoutInitStandard(&mmIntermediate.layout, StandardLayoutType::eUmajor);
    result = micromeshOpDecompressDisplacementBegin(singleThreadedCtx, &mmInput, &mmIntermediate);
    if(Result::eSuccess != result)
    {
      return false;
    }
    assert(numValues == mmIntermediate.values.count);

    std::vector<uint16_t> intermediateValues(numValues);
    mmIntermediate.values.data                    = intermediateValues.data();
    mmIntermediate.values.byteStride              = sizeof(uint16_t);
    mmIntermediate.triangleSubdivLevels.data      = (void*)&subdivLevel;
    mmIntermediate.triangleValueIndexOffsets.data = (void*)&zero;
    result = micromeshOpDecompressDisplacementEnd(singleThreadedCtx, &mmIntermediate);

    for(uint64_t i = 0; i < numValues; i++)
    {
      output[i] = float(intermediateValues[i]) / float(UNORM11_MASK);
    }
  }
  if(format == Format::eR11_unorm_packed_align32)
  {
    result = micromeshQuantizedPackedToFloatValues(true, &triangleInput, &triangleInputExpansion_identity,
                                                   &triangleFloat, &triangleFloatExpansion_identity, &messageCallback);
  }
  else
  {
    result = micromeshQuantizedToFloatValues(true, &triangleInput, &triangleInputExpansion_identity, &triangleFloat,
                                             &triangleFloatExpansion_identity, &messageCallback);
  }
  return (Result::eSuccess == result);
}

// Analyzes the input data, generating inputs for later operations:
// - Determines a group scale and bias so all values fit in [0,1]
// - Detects outliers that would make the scale/bias really large
// (producing quantization artifacts)
// - Determines the subdiv level of each output triangle
// - In the future, will determine a level and compression format for each
// triangle based on heuristics.
// Fills trimmedInfo.{minSubdivLevel, maxSubdivLevel, valueFloatExpansion}
// Returns true on success, prints and returns false on failure.
bool passAnalyze(
    // Inputs
    meshops::Context&                 ctx,
    const ToolOptimizeArgs&           args,
    const baryutils::BaryContentData& baryContent,
    const Format&                     format,
    const FormatInfo&                 formatInfo,
    // Output
    baryutils::BaryBasicData& analyzed)
{
  const baryutils::BaryBasicData& baryBasicData = baryContent.basic;
  assert(baryBasicData.groups.size() == 1 && "ToolScene should have split .bary files into groups.");
  const bary::Group&             baryGroup    = baryBasicData.groups[0];

  // Determine the subdiv level of each output triangle.
  const uint32_t trimSubdivU8   = uint32_t(std::min(255, args.trimSubdiv));
  uint32_t       minSubdivLevel = std::numeric_limits<uint32_t>::max();
  uint32_t       maxSubdivLevel = 0;
  {
    for(uint32_t outPrim = 0; outPrim < baryGroup.triangleCount; outPrim++)
    {
      const uint32_t inPrim           = baryGroup.triangleFirst + outPrim;
      const uint32_t inputSubdivLevel = uint32_t(std::min<int>(255, baryBasicData.triangles[inPrim].subdivLevel));
      const uint32_t outSubdivLevel   = std::min(inputSubdivLevel, trimSubdivU8);
      analyzed.triangles[outPrim].subdivLevel = static_cast<uint16_t>(outSubdivLevel);
      minSubdivLevel                          = std::min(minSubdivLevel, outSubdivLevel);
      maxSubdivLevel                          = std::max(maxSubdivLevel, outSubdivLevel);
    }
  }
  analyzed.minSubdivLevel = analyzed.groups[0].minSubdivLevel = minSubdivLevel;
  analyzed.maxSubdivLevel = analyzed.groups[0].maxSubdivLevel = maxSubdivLevel;

  // Determine a group scale and bias so all values fit in [0,1].
  // Note that the input group might already have a scale and bias but have
  // e.g. an fp32 input, so we need to do value * inputScale + inputBias
  // when reading values.
  // This stage also attempts to detect whether the input has outliers - these
  // are usually distances produced when dividing almost by 0; because they're
  // so large, they set the min and max to values far apart, which makes
  // 11-bit quantization highly visible. We detect outliers using the Grubbs
  // test; more details below!

  const float4 inputBias  = baryVectorToFloat4(baryGroup.floatBias);
  const float4 inputScale = baryVectorToFloat4(baryGroup.floatScale);

  // Note: Currently, this code decomposes the config into components and then
  // performs some threading manually. This could be cleaner.
  meshops::ContextConfig ctxConfig{};
  std::ignore = meshopsContextGetConfig(ctx, &ctxConfig);

  {
    std::vector<float4> primValueMinima(baryGroup.triangleCount);
    std::vector<float4> primValueMaxima(baryGroup.triangleCount);

    // The Grubbs test requires the sample mean and standard deviation. To keep
    // things simple, we compute these by accumulating sum(values) and
    // sum(values^2); there are numerically better ways, but the parallel
    // approach here should help with precision errors a bit.
    std::vector<float4> primValueSums(baryGroup.triangleCount);
    std::vector<float4> primValueSquaredSums(baryGroup.triangleCount);

    nvh::parallel_batches(
        baryGroup.triangleCount,
        [&](uint64_t primInGroup) {
          const bary::Triangle& inPrim             = baryBasicData.triangles[baryGroup.triangleFirst + primInGroup];
          const uint32_t        outPrimSubdivLevel = analyzed.triangles[primInGroup].subdivLevel;
          float4&               primMin            = primValueMinima[primInGroup];
          primMin                                  = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
          float4& primMax                          = primValueMaxima[primInGroup];
          primMax                                  = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
          float4& primSum                          = primValueSums[primInGroup];
          primSum                                  = {0.0f, 0.0f, 0.0f, 0.0f};
          float4& primSquaredSum                   = primValueSquaredSums[primInGroup];
          primSquaredSum                           = {0.0f, 0.0f, 0.0f, 0.0f};

          // Note that we only look at trimmed vertices. This makes the
          // indexing a bit more complex, but gives a tigher fit.
          const uint32_t inNumSegments  = subdivLevelGetSegmentCount(inPrim.subdivLevel);
          const uint32_t inNumVertices  = subdivLevelGetVertexCount(inPrim.subdivLevel);
          const uint32_t outNumSegments = subdivLevelGetSegmentCount(outPrimSubdivLevel);
          assert(inNumSegments >= outNumSegments);
          const uint32_t vtxStep       = inNumSegments / outNumSegments;
          const uint32_t valueByteSize = formatInfo.byteSize;

          float* floatData = (float*)alloca(sizeof(float) * formatInfo.channelCount * inNumVertices);
          bool   ok        = toFloats(ctxConfig.messageCallback, format, formatInfo,
                                      &baryBasicData.values[valueByteSize * (inPrim.valuesOffset + baryGroup.valueFirst)],
                                      inNumVertices, inPrim.subdivLevel, inPrim.blockFormat, floatData);
          if(!ok)
          {
            assert(!"Float conversion failed!");
            return;
          }

          for(uint32_t u = 0; u <= inNumSegments; u += vtxStep)
          {
            for(uint32_t v = 0; v <= inNumSegments - u; v += vtxStep)
            {
              const uint32_t valueIdx = bary::baryValueLayoutGetIndex(baryBasicData.valuesInfo.valueLayout,
                                                                      baryBasicData.valuesInfo.valueFrequency, u, v,
                                                                      0 /* isUpperTriangle */, inPrim.subdivLevel);
              for(uint32_t c = 0; c < formatInfo.channelCount; c++)
              {
                const float vApplied = floatData[valueIdx * formatInfo.channelCount + c] * inputScale[c] + inputBias[c];
                if(vApplied < primMin[c])
                {
                  primMin[c] = vApplied;
                }
                if(vApplied > primMax[c])
                {
                  primMax[c] = vApplied;
                }
                primSum[c] += vApplied;
                primSquaredSum[c] += vApplied * vApplied;
              }
            }
          }
        },
        ctxConfig.threadCount);

    // Accumulate results
    float4 groupMin{FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
    float4 groupMax{-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};
    float4 groupSum{0.0f, 0.0f, 0.0f, 0.0f};
    float4 groupSquaredSum{0.0f, 0.0f, 0.0f, 0.0f};
    size_t groupTrimmedNumValues = 0;
    for(uint32_t primInGroup = 0; primInGroup < baryGroup.triangleCount; primInGroup++)
    {
      const float4& primMin = primValueMinima[primInGroup];
      const float4& primMax = primValueMaxima[primInGroup];
      for(uint32_t c = 0; c < formatInfo.channelCount; c++)
      {
        if(primMin[c] < groupMin[c])
        {
          groupMin[c] = primMin[c];
        }
        if(primMax[c] > groupMax[c])
        {
          groupMax[c] = primMax[c];
        }
        groupSum[c] += primValueSums[primInGroup][c];
        groupSquaredSum[c] += primValueSquaredSums[primInGroup][c];
      }
      for(uint32_t c = formatInfo.channelCount; c < 4; c++)
      {
        groupMin[c] = 0.0f;
        groupMax[c] = 0.0f;
      }

      groupTrimmedNumValues += subdivLevelGetVertexCount(analyzed.triangles[primInGroup].subdivLevel);
    }

    {
      float4 tempBias, tempScale;
      for(uint32_t c = 0; c < 4; c++)
      {
        tempBias[c]  = groupMin[c];
        tempScale[c] = groupMax[c] - groupMin[c];
      }
      analyzed.groups[0].floatBias  = float4ToBaryVector(tempBias);
      analyzed.groups[0].floatScale = float4ToBaryVector(tempScale);
    }

    if(ctxConfig.verbosityLevel > 0)
    {
      LOGI("Minimum values: {%f, %f, %f, %f}\n", groupMin[0], groupMin[1], groupMin[2], groupMin[3]);
      LOGI("Maximum values: {%f, %f, %f, %f}\n", groupMax[0], groupMax[1], groupMax[2], groupMax[3]);
      LOGI("Computed global bias:  {%f, %f, %f, %f}\n", analyzed.groups[0].floatBias.r, analyzed.groups[0].floatBias.g,
           analyzed.groups[0].floatBias.b, analyzed.groups[0].floatBias.a);
      LOGI("Computed global scale: {%f, %f, %f, %f}\n", analyzed.groups[0].floatScale.r,
           analyzed.groups[0].floatScale.g, analyzed.groups[0].floatScale.b, analyzed.groups[0].floatScale.a);
    }

    // Apply the Grubbs test to find outliers.
    for(uint32_t c = 0; c < formatInfo.channelCount; c++)
    {
      float testStatistic, testStatisticLimit;
      if(!grubbsTest(groupSum[c], groupSquaredSum[c], groupMin[c], groupMax[c], groupTrimmedNumValues, testStatistic, testStatisticLimit))
      {
        LOGW(
            "It looks like these micromesh values aren't ready to be compressed: they appear to have outliers (most "
            "likely issues during the baking process, like dividing by the length of a direction close to 0; try "
            "enabling direction bounds fitting)! For channel %i, the minimum value was %f, and the maximum value was "
            "%f - if a baker did direction bounds fitting, we'd expect these to be close to 0 and 1 respectively. The "
            "Grubbs test statistic was %f; for %zu values in this group, we'd expect a test statistic less than %f "
            "with 99.9%% confidence if there were no outliers.",
            c, groupMin[c], groupMax[c], testStatistic, groupTrimmedNumValues, testStatisticLimit);
        break;
      }
    }
  }

  return true;
}

// Trims and quantizes the input data to uncompressed unorm11 x (# of channels).
// After this operation, trimmedMicromap will be a valid micromap.
// Returns true for success.
bool passTrimQuantize(
    // Inputs
    const meshops::Context&           ctx,
    const ToolOptimizeArgs&           args,
    const baryutils::BaryContentData& baryContent,
    const Format&                     format,
    const FormatInfo&                 formatInfo,
    // Output
    baryutils::BaryBasicData& output)
{
  const baryutils::BaryBasicData& baryBasicData = baryContent.basic;
  assert(baryBasicData.groups.size() == 1 && "ToolScene should have split .bary files into groups.");
  const bary::Group& baryGroup = baryBasicData.groups[0];

  // Compute the input triangle offsets based on the subdivision level of
  // each triangle.
  {
    uint32_t valueCount = 0;
    for(uint32_t primInGroup = 0; primInGroup < baryGroup.triangleCount; primInGroup++)
    {
      output.triangles[primInGroup].valuesOffset = valueCount;
      const uint32_t numValues                   = subdivLevelGetVertexCount(output.triangles[primInGroup].subdivLevel);
      valueCount += numValues;
    }
    output.valuesInfo.valueCount = output.groups[0].valueCount = valueCount;
    output.values.resize(sizeof(uint16_t) * valueCount);
  }

  uint16_t* outputValuesU16 = reinterpret_cast<uint16_t*>(output.values.data());

  const float4 inputBias  = baryVectorToFloat4(baryGroup.floatBias);
  const float4 inputScale = baryVectorToFloat4(baryGroup.floatScale);

  const float4 outputBias     = baryVectorToFloat4(output.groups[0].floatBias);
  float4       rcpOutputScale = baryVectorToFloat4(output.groups[0].floatScale);
  for(int c = 0; c < 4; c++)
  {
    rcpOutputScale[c] = ((rcpOutputScale[c] == 0.0f) ? 1.0f : (1.0f / rcpOutputScale[c]));
  }

  // Note: Currently, this code decomposes the config into components and then
  // performs some threading manually. This could be cleaner.
  meshops::ContextConfig ctxConfig{};
  std::ignore = meshopsContextGetConfig(ctx, &ctxConfig);

  // TODO: The parallel_batches structure here is the same as in passAnalyze;
  // this could be refactored into something common to the two.
  nvh::parallel_batches(
      baryGroup.triangleCount,
      [&](uint64_t primInGroup) {
        const bary::Triangle& inPrim               = baryBasicData.triangles[baryGroup.triangleFirst + primInGroup];
        const uint32_t        outPrimSubdivLevel   = output.triangles[primInGroup].subdivLevel;
        uint32_t              encoderInputValueIdx = output.triangles[primInGroup].valuesOffset;

        const uint32_t inNumSegments  = subdivLevelGetSegmentCount(inPrim.subdivLevel);
        const uint32_t inNumVertices  = subdivLevelGetVertexCount(inPrim.subdivLevel);
        const uint32_t outNumSegments = subdivLevelGetSegmentCount(outPrimSubdivLevel);
        const uint32_t vtxStep        = inNumSegments / outNumSegments;
        const uint32_t valueByteSize  = formatInfo.byteSize;

        float* floatData = (float*)alloca(sizeof(float) * formatInfo.channelCount * inNumVertices);
        bool   ok        = toFloats(ctxConfig.messageCallback, format, formatInfo,
                                    &baryBasicData.values[valueByteSize * (inPrim.valuesOffset + baryGroup.valueFirst)],
                                    inNumVertices, inPrim.subdivLevel, inPrim.blockFormat, floatData);
        if(!ok)
        {
          assert(!"Float conversion failed!");
          return;
        }

        for(uint32_t u = 0; u <= inNumSegments; u += vtxStep)
        {
          for(uint32_t v = 0; v <= inNumSegments - u; v += vtxStep)
          {
            const uint32_t valueIdx = bary::baryValueLayoutGetIndex(baryBasicData.valuesInfo.valueLayout,
                                                                    baryBasicData.valuesInfo.valueFrequency, u, v,
                                                                    0 /* isUpperTriangle */, inPrim.subdivLevel);

            for(uint32_t c = 0; c < formatInfo.channelCount; c++)
            {
              const float vApplied = floatData[valueIdx * formatInfo.channelCount + c] * inputScale[c] + inputBias[c];
              const float vUnorm   = (vApplied - outputBias[c]) * rcpOutputScale[c];
              // The +0.5f here is for correct quantization: see
              // http://cbloomrants.blogspot.com/2020/09/topics-in-quantization-for-games.html
              outputValuesU16[encoderInputValueIdx] =
                  static_cast<uint16_t>(std::min(float(UNORM11_MASK), vUnorm * float(UNORM11_MASK) + 0.5f));
              encoderInputValueIdx++;
            }
          }
        }
      },
      ctxConfig.threadCount);

  output.valuesInfo.valueFormat        = bary::Format::eR11_unorm_pack16;
  output.valuesInfo.valueByteSize      = sizeof(uint16_t);
  output.valuesInfo.valueByteAlignment = alignof(uint16_t*);
  output.valuesInfo.valueFrequency     = bary::ValueFrequency::ePerVertex;
  output.valuesInfo.valueLayout        = bary::ValueLayout::eTriangleUmajor;

  return true;
}

bool toolOptimize(ToolContext& context, const ToolOptimizeArgs& args, std::unique_ptr<micromesh_tool::ToolScene>& scene)
{
  meshops::Context& meshopsContext = context.meshopsContext();

  // We'll replace the scene's barys with a single bary.
  std::vector<baryutils::BaryContentData> outBaryContents;
  // This is here to handle the case where some meshes don't have DMMs.
  std::vector<size_t> outBaryGroupToMeshIndex;

  // For each mesh...
  for(size_t meshIdx = 0; meshIdx < scene->meshes().size(); meshIdx++)
  {
    LOGI("Mesh %zu (%zu/%zu)\n", meshIdx, meshIdx + 1, scene->meshes().size());

    // We will modify attributes of the mesh in-place.
    ToolMesh& mesh     = *scene->meshes()[meshIdx];
    auto&     meshView = mesh.view();

    if(meshView.triangleCount() == 0)
    {
      // No triangles; nothing to do
      LOGI("Mesh %zu has no triangles.\n", meshIdx);
      continue;
    }

    if(mesh.relations().bary == -1)
    {
      // No micromap data; nothing to do
      LOGI("Mesh %zu has no micromap data.\n", meshIdx);
      continue;
    }

    const ToolBary& inBary = *scene->barys()[mesh.relations().bary];

    if(mesh.relations().group < 0 || static_cast<size_t>(mesh.relations().group) >= inBary.groups().size())
    {
      LOGE("Error: The group for mesh %zu was out of range.\n", meshIdx);
      return false;
    }

    const bary::ContentView inBaryContent = inBary.groups()[mesh.relations().group];

    if(mesh.relations().mapOffset != 0)
    {
      LOGE("Error: Mesh %zu had a non-zero mapOffset, which is not currently supported in the optimizer.\n", meshIdx);
      return false;
    }

    Format     micromeshFormat{};
    FormatInfo micromeshFormatInfo{};
    if(!checkBaryRequirements(inBaryContent, micromeshFormat, micromeshFormatInfo))
    {
      return false;
    }

    // Uncompressed input for the encoder and its underlying storage.
    baryutils::BaryBasicData encoderInput;
    encoderInput.groups.push_back(bary::Group{});
    // Note that we don't use inBaryContent.basic.triangleCount; the bary
    // view splitting in ToolScene documents that this isn't set correctly,
    // because it can't change the group first/length values.
    encoderInput.groups[0].triangleCount = inBaryContent.basic.groups[0].triangleCount;
    encoderInput.triangles.resize(encoderInput.groups[0].triangleCount);

    // We perform a few steps here.
    // 1. We determine the subdiv level of each output triangle.
    // 2. Determine a global scale and bias so all values fit inside [0,1].
    // 3. We convert the input data (trimming it along the way) to uncompressed
    // unorm11 (inside a uint16_t array).
    // 4. We compress the data, using mesh topology information from the .bary
    // file itself (we assume we don't need to handle glTF input at the moment).
    // 5. Compute new topology edge flags from step 1.
    // 6. Combine the compressed values, other essential properties from the
    // original file, and the new properties into the BaryContentData.

    if(!passAnalyze(meshopsContext, args, inBaryContent, micromeshFormat, micromeshFormatInfo,  //
                    encoderInput))
    {
      return false;
    }

    if(!passTrimQuantize(meshopsContext, args, inBaryContent, micromeshFormat, micromeshFormatInfo,  //
                         encoderInput))
    {
      return false;
    }

    meshops::MeshTopologyData topology;
    {
      // Build the topology. Note that we don't want to have the topology
      // weld vertices here; internal cracks in the mesh can be used as
      // a way to tell the compressor to match values exactly along an edge.
      // (In particular, welding vertices will produce differences in the
      // baker cracks test case).
      // So, we go down to the meshops layer and invoke
      // meshopsOpBuildTopology() instead of micromesh_tool::buildTopologyData().
      meshops::OpBuildTopology_input opInput;
      opInput.meshView                    = meshView;
      opInput.triangleUniqueVertexIndices = meshops::ArrayView<Vector_uint32_3>(meshView.triangleVertices);
      meshops::OpBuildTopology_output opOutput;
      opOutput.meshTopology = &topology;
      if(micromesh::Result::eSuccess != meshops::meshopsOpBuildTopology(context.meshopsContext(), 1, &opInput, &opOutput))
      {
        LOGE("Error: failed to build mesh topology\n");
        return false;
      }
    }

    // Create a new compressed bary output.
    outBaryContents.push_back({});
    outBaryGroupToMeshIndex.push_back(meshIdx);
    baryutils::BaryContentData& outBaryContent = outBaryContents.back();

    // Compress encoderInput and store the result in outBaryContent.
    {
      // Hack: baryBasicDataCompressedAppend() doesn't require min/maxes when
      // our format is eR11_unorm_pack16, but meshopsOpCompressDisplacementMicromaps()
      // checks that they exist in any case. For now, create this data but
      // fill it with 0s. This is OK, since it will be recreated after
      // displacement in any case.
      encoderInput.triangleMinMaxs.resize(2 * encoderInput.triangles.size());
      encoderInput.triangleMinMaxsInfo.elementCount         = uint32_t(encoderInput.triangleMinMaxs.size());
      encoderInput.triangleMinMaxsInfo.elementFormat        = bary::Format::eR11_unorm_pack16;
      encoderInput.triangleMinMaxsInfo.elementByteSize      = sizeof(uint16_t);
      encoderInput.triangleMinMaxsInfo.elementByteAlignment = alignof(uint16_t*);

      bary::BasicView encoderInputView = encoderInput.getView();

      meshops::OpCompressDisplacementMicromap_input opInput;
      opInput.meshTopology            = topology;
      opInput.meshView                = meshView;
      opInput.settings.minimumPSNR    = args.psnr;
      opInput.settings.validateInputs = opInput.settings.validateOutputs = args.validateEdges;
      opInput.uncompressedDisplacement                                   = &encoderInputView;
      opInput.uncompressedDisplacementGroupIndex                         = 0;

      meshops::OpCompressDisplacementMicromap_output opOutput;
      opOutput.compressedDisplacement           = &outBaryContent.basic;
      opOutput.compressedDisplacementRasterMips = &outBaryContent.misc;

      if(micromesh::Result::eSuccess
         != meshops::meshopsOpCompressDisplacementMicromaps(context.meshopsContext(), 1, &opInput, &opOutput))
      {
        LOGE("Error: Compressing mesh %zu failed.\n", meshIdx);
        return false;
      }
    }

    // Subdivision levels are now in the bary data, so clear subdiv levels from this mesh.
    meshView.resize(meshops::MeshAttributeFlagBits::eMeshAttributeTriangleSubdivLevelsBit, 0, 0);

    // Finally, recompute primitive flags.
    if(outBaryContent.basic.minSubdivLevel == outBaryContent.basic.maxSubdivLevel)
    {
      // All subdiv levels are the same, so we can remove primitive flags.
      meshView.resize(meshops::MeshAttributeFlagBits::eMeshAttributeTrianglePrimitiveFlagsBit, 0, 0);
    }
    else
    {
      OpBuildPrimitiveFlags_input opInput;
      opInput.meshTopo                       = topology;
      opInput.meshTriangleSubdivLevels.count = outBaryContent.basic.triangles.size();
      // We use an interesting byte stride here!
      opInput.meshTriangleSubdivLevels.data       = &outBaryContent.basic.triangles[0].subdivLevel;
      opInput.meshTriangleSubdivLevels.byteStride = sizeof(outBaryContent.basic.triangles[0]);

      meshView.resize(meshops::MeshAttributeFlagBits::eMeshAttributeTrianglePrimitiveFlagsBit, meshView.triangleCount(),
                      meshView.vertexCount());

      OpBuildPrimitiveFlags_output opOutput;
      arraySetDataVec(opOutput.meshTrianglePrimitiveFlags, meshView.trianglePrimitiveFlags);

      if(micromesh::Result::eSuccess
         != micromeshOpBuildPrimitiveFlags(context.meshopsContext()->m_micromeshContext, &opInput, &opOutput))
      {
        LOGE("Error: Computing primitive flags for mesh %zu failed.\n", meshIdx);
        return false;
      }
    }
  }

  // Now replace the scene's bary data with the data from outBaryContent:
  if(outBaryContents.empty())
  {
    LOGW("Warning: The optimized scene had no micromap data. Did the input to the optimizer have micromeshes?\n");
    // We need to avoid linking a ToolBary if there's no bary data;
    // otherwise, we'll get an assertion error during saving. See issue #107.
    scene->clearBarys();
  }
  else
  {
    std::unique_ptr<ToolBary> outBary = ToolBary::create(std::move(outBaryContents));
    if(!outBary)
    {
      return false;
    }
    const size_t outBaryIndex = scene->replaceBarys(std::move(outBary));

    // Link the meshes with the bary groups in the micromap file.
    for(size_t groupIndex = 0; groupIndex < outBaryGroupToMeshIndex.size(); ++groupIndex)
    {
      const size_t meshIndex = outBaryGroupToMeshIndex[groupIndex];
      scene->linkBary(outBaryIndex, groupIndex, meshIndex);
    }
  }

  return true;
}

void toolOptimizeAddRequirements(meshops::ContextConfig& contextConfig)
{
  // Nothing special needed
}

}  // namespace tool_optimize
