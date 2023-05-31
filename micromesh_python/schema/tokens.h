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
#ifndef NVMICROMESH_TOKENS_H
#define NVMICROMESH_TOKENS_H

/// \file nvMicromesh/tokens.h

// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
// 
// This is an automatically generated file (by usdGenSchema.py).
// Do not hand-edit!
// 
// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

#include "pxr/pxr.h"
#include "./api.h"
#include "pxr/base/tf/staticData.h"
#include "pxr/base/tf/token.h"
#include <vector>

PXR_NAMESPACE_OPEN_SCOPE


/// \class NvMicromeshTokensType
///
/// \link NvMicromeshTokens \endlink provides static, efficient
/// \link TfToken TfTokens\endlink for use in all public USD API.
///
/// These tokens are auto-generated from the module's schema, representing
/// property names, for when you need to fetch an attribute or relationship
/// directly by name, e.g. UsdPrim::GetAttribute(), in the most efficient
/// manner, and allow the compiler to verify that you spelled the name
/// correctly.
///
/// NvMicromeshTokens also contains all of the \em allowedTokens values
/// declared for schema builtin attributes of 'token' scene description type.
/// Use NvMicromeshTokens like so:
///
/// \code
///     gprim.GetMyTokenValuedAttr().Set(NvMicromeshTokens->primvarsMicromeshCompressed);
/// \endcode
struct NvMicromeshTokensType {
    NVMICROMESH_API NvMicromeshTokensType();
    /// \brief "primvars:micromesh:compressed"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshCompressed;
    /// \brief "primvars:micromesh:directionBounds"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshDirectionBounds;
    /// \brief "primvars:micromesh:directions"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshDirections;
    /// \brief "primvars:micromesh:floatBias"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshFloatBias;
    /// \brief "primvars:micromesh:floatScale"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshFloatScale;
    /// \brief "primvars:micromesh:histogramBlockFormats"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshHistogramBlockFormats;
    /// \brief "primvars:micromesh:histogramCounts"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshHistogramCounts;
    /// \brief "primvars:micromesh:histogramSubdivLevels"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshHistogramSubdivLevels;
    /// \brief "primvars:micromesh:maxSubdivLevel"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshMaxSubdivLevel;
    /// \brief "primvars:micromesh:minSubdivLevel"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshMinSubdivLevel;
    /// \brief "primvars:micromesh:offsetMap"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshOffsetMap;
    /// \brief "primvars:micromesh:quaternionMap"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshQuaternionMap;
    /// \brief "primvars:micromesh:triangleBlockFormats"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshTriangleBlockFormats;
    /// \brief "primvars:micromesh:triangleFlags"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshTriangleFlags;
    /// \brief "primvars:micromesh:triangleFlagsByteSize"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshTriangleFlagsByteSize;
    /// \brief "primvars:micromesh:triangleFlagsCount"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshTriangleFlagsCount;
    /// \brief "primvars:micromesh:triangleFlagsFormat"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshTriangleFlagsFormat;
    /// \brief "primvars:micromesh:triangleMappings"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshTriangleMappings;
    /// \brief "primvars:micromesh:triangleMappingsByteSize"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshTriangleMappingsByteSize;
    /// \brief "primvars:micromesh:triangleMappingsCount"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshTriangleMappingsCount;
    /// \brief "primvars:micromesh:triangleMappingsFormat"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshTriangleMappingsFormat;
    /// \brief "primvars:micromesh:triangleMinMaxs"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshTriangleMinMaxs;
    /// \brief "primvars:micromesh:triangleMinMaxsByteSize"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshTriangleMinMaxsByteSize;
    /// \brief "primvars:micromesh:triangleMinMaxsCount"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshTriangleMinMaxsCount;
    /// \brief "primvars:micromesh:triangleMinMaxsFormat"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshTriangleMinMaxsFormat;
    /// \brief "primvars:micromesh:triangleSubdivLevels"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshTriangleSubdivLevels;
    /// \brief "primvars:micromesh:triangleValueOffsets"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshTriangleValueOffsets;
    /// \brief "primvars:micromesh:valueByteSize"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshValueByteSize;
    /// \brief "primvars:micromesh:valueCount"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshValueCount;
    /// \brief "primvars:micromesh:valueFormat"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshValueFormat;
    /// \brief "primvars:micromesh:valueFrequency"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshValueFrequency;
    /// \brief "primvars:micromesh:valueLayout"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshValueLayout;
    /// \brief "primvars:micromesh:values"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshValues;
    /// \brief "primvars:micromesh:version"
    /// 
    /// NvMicromeshDisplacementMicromapAPI
    const TfToken primvarsMicromeshVersion;
    /// A vector of all of the tokens listed above.
    const std::vector<TfToken> allTokens;
};

/// \var NvMicromeshTokens
///
/// A global variable with static, efficient \link TfToken TfTokens\endlink
/// for use in all public USD API.  \sa NvMicromeshTokensType
extern NVMICROMESH_API TfStaticData<NvMicromeshTokensType> NvMicromeshTokens;

PXR_NAMESPACE_CLOSE_SCOPE

#endif
