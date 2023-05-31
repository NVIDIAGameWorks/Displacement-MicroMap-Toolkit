//
// Copyright 2016 Pixar
//
// Licensed under the Apache License, Version 2.0 (the "Apache License")
// with the following modification; you may not use this file except in
// compliance with the Apache License and the following modification to it:
// Section 6. Trademarks. is deleted and replaced with:
//
// 6. Trademarks. This License does not grant permission to use the trade
//    names, trademarks, service marks, or product names of the Licensor
//    and its affiliates, except as required to comply with Section 4(c) of
//    the License and to reproduce the content of the NOTICE file.
//
// You may obtain a copy of the Apache License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the Apache License with the above modification is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the Apache License for the specific
// language governing permissions and limitations under the Apache License.
//
#include "./tokens.h"

PXR_NAMESPACE_OPEN_SCOPE

NvMicromeshTokensType::NvMicromeshTokensType() :
    primvarsMicromeshCompressed("primvars:micromesh:compressed", TfToken::Immortal),
    primvarsMicromeshDirectionBounds("primvars:micromesh:directionBounds", TfToken::Immortal),
    primvarsMicromeshDirections("primvars:micromesh:directions", TfToken::Immortal),
    primvarsMicromeshFloatBias("primvars:micromesh:floatBias", TfToken::Immortal),
    primvarsMicromeshFloatScale("primvars:micromesh:floatScale", TfToken::Immortal),
    primvarsMicromeshHistogramBlockFormats("primvars:micromesh:histogramBlockFormats", TfToken::Immortal),
    primvarsMicromeshHistogramCounts("primvars:micromesh:histogramCounts", TfToken::Immortal),
    primvarsMicromeshHistogramSubdivLevels("primvars:micromesh:histogramSubdivLevels", TfToken::Immortal),
    primvarsMicromeshMaxSubdivLevel("primvars:micromesh:maxSubdivLevel", TfToken::Immortal),
    primvarsMicromeshMinSubdivLevel("primvars:micromesh:minSubdivLevel", TfToken::Immortal),
    primvarsMicromeshOffsetMap("primvars:micromesh:offsetMap", TfToken::Immortal),
    primvarsMicromeshQuaternionMap("primvars:micromesh:quaternionMap", TfToken::Immortal),
    primvarsMicromeshTriangleBlockFormats("primvars:micromesh:triangleBlockFormats", TfToken::Immortal),
    primvarsMicromeshTriangleFlags("primvars:micromesh:triangleFlags", TfToken::Immortal),
    primvarsMicromeshTriangleFlagsByteSize("primvars:micromesh:triangleFlagsByteSize", TfToken::Immortal),
    primvarsMicromeshTriangleFlagsCount("primvars:micromesh:triangleFlagsCount", TfToken::Immortal),
    primvarsMicromeshTriangleFlagsFormat("primvars:micromesh:triangleFlagsFormat", TfToken::Immortal),
    primvarsMicromeshTriangleMappings("primvars:micromesh:triangleMappings", TfToken::Immortal),
    primvarsMicromeshTriangleMappingsByteSize("primvars:micromesh:triangleMappingsByteSize", TfToken::Immortal),
    primvarsMicromeshTriangleMappingsCount("primvars:micromesh:triangleMappingsCount", TfToken::Immortal),
    primvarsMicromeshTriangleMappingsFormat("primvars:micromesh:triangleMappingsFormat", TfToken::Immortal),
    primvarsMicromeshTriangleMinMaxs("primvars:micromesh:triangleMinMaxs", TfToken::Immortal),
    primvarsMicromeshTriangleMinMaxsByteSize("primvars:micromesh:triangleMinMaxsByteSize", TfToken::Immortal),
    primvarsMicromeshTriangleMinMaxsCount("primvars:micromesh:triangleMinMaxsCount", TfToken::Immortal),
    primvarsMicromeshTriangleMinMaxsFormat("primvars:micromesh:triangleMinMaxsFormat", TfToken::Immortal),
    primvarsMicromeshTriangleSubdivLevels("primvars:micromesh:triangleSubdivLevels", TfToken::Immortal),
    primvarsMicromeshTriangleValueOffsets("primvars:micromesh:triangleValueOffsets", TfToken::Immortal),
    primvarsMicromeshValueByteSize("primvars:micromesh:valueByteSize", TfToken::Immortal),
    primvarsMicromeshValueCount("primvars:micromesh:valueCount", TfToken::Immortal),
    primvarsMicromeshValueFormat("primvars:micromesh:valueFormat", TfToken::Immortal),
    primvarsMicromeshValueFrequency("primvars:micromesh:valueFrequency", TfToken::Immortal),
    primvarsMicromeshValueLayout("primvars:micromesh:valueLayout", TfToken::Immortal),
    primvarsMicromeshValues("primvars:micromesh:values", TfToken::Immortal),
    primvarsMicromeshVersion("primvars:micromesh:version", TfToken::Immortal),
    allTokens({
        primvarsMicromeshCompressed,
        primvarsMicromeshDirectionBounds,
        primvarsMicromeshDirections,
        primvarsMicromeshFloatBias,
        primvarsMicromeshFloatScale,
        primvarsMicromeshHistogramBlockFormats,
        primvarsMicromeshHistogramCounts,
        primvarsMicromeshHistogramSubdivLevels,
        primvarsMicromeshMaxSubdivLevel,
        primvarsMicromeshMinSubdivLevel,
        primvarsMicromeshOffsetMap,
        primvarsMicromeshQuaternionMap,
        primvarsMicromeshTriangleBlockFormats,
        primvarsMicromeshTriangleFlags,
        primvarsMicromeshTriangleFlagsByteSize,
        primvarsMicromeshTriangleFlagsCount,
        primvarsMicromeshTriangleFlagsFormat,
        primvarsMicromeshTriangleMappings,
        primvarsMicromeshTriangleMappingsByteSize,
        primvarsMicromeshTriangleMappingsCount,
        primvarsMicromeshTriangleMappingsFormat,
        primvarsMicromeshTriangleMinMaxs,
        primvarsMicromeshTriangleMinMaxsByteSize,
        primvarsMicromeshTriangleMinMaxsCount,
        primvarsMicromeshTriangleMinMaxsFormat,
        primvarsMicromeshTriangleSubdivLevels,
        primvarsMicromeshTriangleValueOffsets,
        primvarsMicromeshValueByteSize,
        primvarsMicromeshValueCount,
        primvarsMicromeshValueFormat,
        primvarsMicromeshValueFrequency,
        primvarsMicromeshValueLayout,
        primvarsMicromeshValues,
        primvarsMicromeshVersion
    })
{
}

TfStaticData<NvMicromeshTokensType> NvMicromeshTokens;

PXR_NAMESPACE_CLOSE_SCOPE
