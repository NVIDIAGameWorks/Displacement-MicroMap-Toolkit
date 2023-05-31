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
#include "./displacementMicromapAPI.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/typed.h"
#include "pxr/usd/usd/tokens.h"

#include "pxr/usd/sdf/types.h"
#include "pxr/usd/sdf/assetPath.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the schema with the TfType system.
TF_REGISTRY_FUNCTION(TfType)
{
    TfType::Define<NvMicromeshDisplacementMicromapAPI,
        TfType::Bases< UsdAPISchemaBase > >();
    
}

TF_DEFINE_PRIVATE_TOKENS(
    _schemaTokens,
    (DisplacementMicromapAPI)
);

/* virtual */
NvMicromeshDisplacementMicromapAPI::~NvMicromeshDisplacementMicromapAPI()
{
}

/* static */
NvMicromeshDisplacementMicromapAPI
NvMicromeshDisplacementMicromapAPI::Get(const UsdStagePtr &stage, const SdfPath &path)
{
    if (!stage) {
        TF_CODING_ERROR("Invalid stage");
        return NvMicromeshDisplacementMicromapAPI();
    }
    return NvMicromeshDisplacementMicromapAPI(stage->GetPrimAtPath(path));
}


/* virtual */
UsdSchemaKind NvMicromeshDisplacementMicromapAPI::_GetSchemaKind() const
{
    return NvMicromeshDisplacementMicromapAPI::schemaKind;
}

/* static */
bool
NvMicromeshDisplacementMicromapAPI::CanApply(
    const UsdPrim &prim, std::string *whyNot)
{
    return prim.CanApplyAPI<NvMicromeshDisplacementMicromapAPI>(whyNot);
}

/* static */
NvMicromeshDisplacementMicromapAPI
NvMicromeshDisplacementMicromapAPI::Apply(const UsdPrim &prim)
{
    if (prim.ApplyAPI<NvMicromeshDisplacementMicromapAPI>()) {
        return NvMicromeshDisplacementMicromapAPI(prim);
    }
    return NvMicromeshDisplacementMicromapAPI();
}

/* static */
const TfType &
NvMicromeshDisplacementMicromapAPI::_GetStaticTfType()
{
    static TfType tfType = TfType::Find<NvMicromeshDisplacementMicromapAPI>();
    return tfType;
}

/* static */
bool 
NvMicromeshDisplacementMicromapAPI::_IsTypedSchema()
{
    static bool isTyped = _GetStaticTfType().IsA<UsdTyped>();
    return isTyped;
}

/* virtual */
const TfType &
NvMicromeshDisplacementMicromapAPI::_GetTfType() const
{
    return _GetStaticTfType();
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshVersionAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshVersion);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshVersionAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshVersion,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshCompressedAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshCompressed);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshCompressedAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshCompressed,
                       SdfValueTypeNames->Bool,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshFloatScaleAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshFloatScale);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshFloatScaleAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshFloatScale,
                       SdfValueTypeNames->Float,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshFloatBiasAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshFloatBias);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshFloatBiasAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshFloatBias,
                       SdfValueTypeNames->Float,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshMinSubdivLevelAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshMinSubdivLevel);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshMinSubdivLevelAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshMinSubdivLevel,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshMaxSubdivLevelAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshMaxSubdivLevel);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshMaxSubdivLevelAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshMaxSubdivLevel,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshDirectionsAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshDirections);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshDirectionsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshDirections,
                       SdfValueTypeNames->Float3Array,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshDirectionBoundsAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshDirectionBounds);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshDirectionBoundsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshDirectionBounds,
                       SdfValueTypeNames->Float2Array,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshTriangleMappingsFormatAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshTriangleMappingsFormat);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshTriangleMappingsFormatAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshTriangleMappingsFormat,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshTriangleMappingsCountAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshTriangleMappingsCount);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshTriangleMappingsCountAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshTriangleMappingsCount,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshTriangleMappingsByteSizeAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshTriangleMappingsByteSize);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshTriangleMappingsByteSizeAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshTriangleMappingsByteSize,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshTriangleMappingsAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshTriangleMappings);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshTriangleMappingsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshTriangleMappings,
                       SdfValueTypeNames->UCharArray,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshValueLayoutAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshValueLayout);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshValueLayoutAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshValueLayout,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshValueFrequencyAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshValueFrequency);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshValueFrequencyAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshValueFrequency,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshValueFormatAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshValueFormat);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshValueFormatAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshValueFormat,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshValueCountAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshValueCount);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshValueCountAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshValueCount,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshValueByteSizeAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshValueByteSize);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshValueByteSizeAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshValueByteSize,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshValuesAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshValues);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshValuesAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshValues,
                       SdfValueTypeNames->UCharArray,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshTriangleFlagsFormatAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshTriangleFlagsFormat);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshTriangleFlagsFormatAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshTriangleFlagsFormat,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshTriangleFlagsCountAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshTriangleFlagsCount);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshTriangleFlagsCountAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshTriangleFlagsCount,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshTriangleFlagsByteSizeAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshTriangleFlagsByteSize);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshTriangleFlagsByteSizeAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshTriangleFlagsByteSize,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshTriangleFlagsAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshTriangleFlags);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshTriangleFlagsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshTriangleFlags,
                       SdfValueTypeNames->UCharArray,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshTriangleValueOffsetsAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshTriangleValueOffsets);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshTriangleValueOffsetsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshTriangleValueOffsets,
                       SdfValueTypeNames->UIntArray,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshTriangleSubdivLevelsAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshTriangleSubdivLevels);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshTriangleSubdivLevelsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshTriangleSubdivLevels,
                       SdfValueTypeNames->UIntArray,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshTriangleBlockFormatsAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshTriangleBlockFormats);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshTriangleBlockFormatsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshTriangleBlockFormats,
                       SdfValueTypeNames->UIntArray,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshHistogramCountsAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshHistogramCounts);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshHistogramCountsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshHistogramCounts,
                       SdfValueTypeNames->UIntArray,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshHistogramSubdivLevelsAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshHistogramSubdivLevels);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshHistogramSubdivLevelsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshHistogramSubdivLevels,
                       SdfValueTypeNames->UIntArray,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshHistogramBlockFormatsAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshHistogramBlockFormats);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshHistogramBlockFormatsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshHistogramBlockFormats,
                       SdfValueTypeNames->UIntArray,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshTriangleMinMaxsFormatAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshTriangleMinMaxsFormat);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshTriangleMinMaxsFormatAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshTriangleMinMaxsFormat,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshTriangleMinMaxsCountAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshTriangleMinMaxsCount);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshTriangleMinMaxsCountAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshTriangleMinMaxsCount,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshTriangleMinMaxsByteSizeAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshTriangleMinMaxsByteSize);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshTriangleMinMaxsByteSizeAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshTriangleMinMaxsByteSize,
                       SdfValueTypeNames->UInt,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshTriangleMinMaxsAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshTriangleMinMaxs);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshTriangleMinMaxsAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshTriangleMinMaxs,
                       SdfValueTypeNames->UCharArray,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshQuaternionMapAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshQuaternionMap);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshQuaternionMapAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshQuaternionMap,
                       SdfValueTypeNames->Asset,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::GetPrimvarsMicromeshOffsetMapAttr() const
{
    return GetPrim().GetAttribute(NvMicromeshTokens->primvarsMicromeshOffsetMap);
}

UsdAttribute
NvMicromeshDisplacementMicromapAPI::CreatePrimvarsMicromeshOffsetMapAttr(VtValue const &defaultValue, bool writeSparsely) const
{
    return UsdSchemaBase::_CreateAttr(NvMicromeshTokens->primvarsMicromeshOffsetMap,
                       SdfValueTypeNames->Asset,
                       /* custom = */ false,
                       SdfVariabilityVarying,
                       defaultValue,
                       writeSparsely);
}

namespace {
static inline TfTokenVector
_ConcatenateAttributeNames(const TfTokenVector& left,const TfTokenVector& right)
{
    TfTokenVector result;
    result.reserve(left.size() + right.size());
    result.insert(result.end(), left.begin(), left.end());
    result.insert(result.end(), right.begin(), right.end());
    return result;
}
}

/*static*/
const TfTokenVector&
NvMicromeshDisplacementMicromapAPI::GetSchemaAttributeNames(bool includeInherited)
{
    static TfTokenVector localNames = {
        NvMicromeshTokens->primvarsMicromeshVersion,
        NvMicromeshTokens->primvarsMicromeshCompressed,
        NvMicromeshTokens->primvarsMicromeshFloatScale,
        NvMicromeshTokens->primvarsMicromeshFloatBias,
        NvMicromeshTokens->primvarsMicromeshMinSubdivLevel,
        NvMicromeshTokens->primvarsMicromeshMaxSubdivLevel,
        NvMicromeshTokens->primvarsMicromeshDirections,
        NvMicromeshTokens->primvarsMicromeshDirectionBounds,
        NvMicromeshTokens->primvarsMicromeshTriangleMappingsFormat,
        NvMicromeshTokens->primvarsMicromeshTriangleMappingsCount,
        NvMicromeshTokens->primvarsMicromeshTriangleMappingsByteSize,
        NvMicromeshTokens->primvarsMicromeshTriangleMappings,
        NvMicromeshTokens->primvarsMicromeshValueLayout,
        NvMicromeshTokens->primvarsMicromeshValueFrequency,
        NvMicromeshTokens->primvarsMicromeshValueFormat,
        NvMicromeshTokens->primvarsMicromeshValueCount,
        NvMicromeshTokens->primvarsMicromeshValueByteSize,
        NvMicromeshTokens->primvarsMicromeshValues,
        NvMicromeshTokens->primvarsMicromeshTriangleFlagsFormat,
        NvMicromeshTokens->primvarsMicromeshTriangleFlagsCount,
        NvMicromeshTokens->primvarsMicromeshTriangleFlagsByteSize,
        NvMicromeshTokens->primvarsMicromeshTriangleFlags,
        NvMicromeshTokens->primvarsMicromeshTriangleValueOffsets,
        NvMicromeshTokens->primvarsMicromeshTriangleSubdivLevels,
        NvMicromeshTokens->primvarsMicromeshTriangleBlockFormats,
        NvMicromeshTokens->primvarsMicromeshHistogramCounts,
        NvMicromeshTokens->primvarsMicromeshHistogramSubdivLevels,
        NvMicromeshTokens->primvarsMicromeshHistogramBlockFormats,
        NvMicromeshTokens->primvarsMicromeshTriangleMinMaxsFormat,
        NvMicromeshTokens->primvarsMicromeshTriangleMinMaxsCount,
        NvMicromeshTokens->primvarsMicromeshTriangleMinMaxsByteSize,
        NvMicromeshTokens->primvarsMicromeshTriangleMinMaxs,
        NvMicromeshTokens->primvarsMicromeshQuaternionMap,
        NvMicromeshTokens->primvarsMicromeshOffsetMap,
    };
    static TfTokenVector allNames =
        _ConcatenateAttributeNames(
            UsdAPISchemaBase::GetSchemaAttributeNames(true),
            localNames);

    if (includeInherited)
        return allNames;
    else
        return localNames;
}

PXR_NAMESPACE_CLOSE_SCOPE

// ===================================================================== //
// Feel free to add custom code below this line. It will be preserved by
// the code generator.
//
// Just remember to wrap code in the appropriate delimiters:
// 'PXR_NAMESPACE_OPEN_SCOPE', 'PXR_NAMESPACE_CLOSE_SCOPE'.
// ===================================================================== //
// --(BEGIN CUSTOM CODE)--
