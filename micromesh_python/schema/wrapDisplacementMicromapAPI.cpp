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
#include "pxr/usd/usd/schemaBase.h"

#include "pxr/usd/sdf/primSpec.h"

#include "pxr/usd/usd/pyConversions.h"
#include "pxr/base/tf/pyAnnotatedBoolResult.h"
#include "pxr/base/tf/pyContainerConversions.h"
#include "pxr/base/tf/pyResultConversions.h"
#include "pxr/base/tf/pyUtils.h"
#include "pxr/base/tf/wrapTypeHelpers.h"

#include <boost/python.hpp>

#include <string>

using namespace boost::python;

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

#define WRAP_CUSTOM                                                     \
    template <class Cls> static void _CustomWrapCode(Cls &_class)

// fwd decl.
WRAP_CUSTOM;

        
static UsdAttribute
_CreatePrimvarsMicromeshVersionAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshVersionAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshCompressedAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshCompressedAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Bool), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshFloatScaleAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshFloatScaleAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Float), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshFloatBiasAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshFloatBiasAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Float), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshMinSubdivLevelAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshMinSubdivLevelAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshMaxSubdivLevelAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshMaxSubdivLevelAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshDirectionsAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshDirectionsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Float3Array), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshDirectionBoundsAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshDirectionBoundsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Float2Array), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshTriangleMappingsFormatAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshTriangleMappingsFormatAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshTriangleMappingsCountAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshTriangleMappingsCountAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshTriangleMappingsByteSizeAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshTriangleMappingsByteSizeAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshTriangleMappingsAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshTriangleMappingsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UCharArray), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshValueLayoutAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshValueLayoutAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshValueFrequencyAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshValueFrequencyAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshValueFormatAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshValueFormatAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshValueCountAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshValueCountAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshValueByteSizeAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshValueByteSizeAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshValuesAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshValuesAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UCharArray), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshTriangleFlagsFormatAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshTriangleFlagsFormatAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshTriangleFlagsCountAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshTriangleFlagsCountAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshTriangleFlagsByteSizeAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshTriangleFlagsByteSizeAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshTriangleFlagsAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshTriangleFlagsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UCharArray), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshTriangleValueOffsetsAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshTriangleValueOffsetsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UIntArray), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshTriangleSubdivLevelsAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshTriangleSubdivLevelsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UIntArray), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshTriangleBlockFormatsAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshTriangleBlockFormatsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UIntArray), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshHistogramCountsAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshHistogramCountsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UIntArray), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshHistogramSubdivLevelsAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshHistogramSubdivLevelsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UIntArray), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshHistogramBlockFormatsAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshHistogramBlockFormatsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UIntArray), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshTriangleMinMaxsFormatAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshTriangleMinMaxsFormatAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshTriangleMinMaxsCountAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshTriangleMinMaxsCountAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshTriangleMinMaxsByteSizeAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshTriangleMinMaxsByteSizeAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UInt), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshTriangleMinMaxsAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshTriangleMinMaxsAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->UCharArray), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshQuaternionMapAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshQuaternionMapAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Asset), writeSparsely);
}
        
static UsdAttribute
_CreatePrimvarsMicromeshOffsetMapAttr(NvMicromeshDisplacementMicromapAPI &self,
                                      object defaultVal, bool writeSparsely) {
    return self.CreatePrimvarsMicromeshOffsetMapAttr(
        UsdPythonToSdfType(defaultVal, SdfValueTypeNames->Asset), writeSparsely);
}

static std::string
_Repr(const NvMicromeshDisplacementMicromapAPI &self)
{
    std::string primRepr = TfPyRepr(self.GetPrim());
    return TfStringPrintf(
        "NvMicromesh.DisplacementMicromapAPI(%s)",
        primRepr.c_str());
}

struct NvMicromeshDisplacementMicromapAPI_CanApplyResult : 
    public TfPyAnnotatedBoolResult<std::string>
{
    NvMicromeshDisplacementMicromapAPI_CanApplyResult(bool val, std::string const &msg) :
        TfPyAnnotatedBoolResult<std::string>(val, msg) {}
};

static NvMicromeshDisplacementMicromapAPI_CanApplyResult
_WrapCanApply(const UsdPrim& prim)
{
    std::string whyNot;
    bool result = NvMicromeshDisplacementMicromapAPI::CanApply(prim, &whyNot);
    return NvMicromeshDisplacementMicromapAPI_CanApplyResult(result, whyNot);
}

} // anonymous namespace

void wrapNvMicromeshDisplacementMicromapAPI()
{
    typedef NvMicromeshDisplacementMicromapAPI This;

    NvMicromeshDisplacementMicromapAPI_CanApplyResult::Wrap<NvMicromeshDisplacementMicromapAPI_CanApplyResult>(
        "_CanApplyResult", "whyNot");

    class_<This, bases<UsdAPISchemaBase> >
        cls("DisplacementMicromapAPI");

    cls
        .def(init<UsdPrim>(arg("prim")))
        .def(init<UsdSchemaBase const&>(arg("schemaObj")))
        .def(TfTypePythonClass())

        .def("Get", &This::Get, (arg("stage"), arg("path")))
        .staticmethod("Get")

        .def("CanApply", &_WrapCanApply, (arg("prim")))
        .staticmethod("CanApply")

        .def("Apply", &This::Apply, (arg("prim")))
        .staticmethod("Apply")

        .def("GetSchemaAttributeNames",
             &This::GetSchemaAttributeNames,
             arg("includeInherited")=true,
             return_value_policy<TfPySequenceToList>())
        .staticmethod("GetSchemaAttributeNames")

        .def("_GetStaticTfType", (TfType const &(*)()) TfType::Find<This>,
             return_value_policy<return_by_value>())
        .staticmethod("_GetStaticTfType")

        .def(!self)

        
        .def("GetPrimvarsMicromeshVersionAttr",
             &This::GetPrimvarsMicromeshVersionAttr)
        .def("CreatePrimvarsMicromeshVersionAttr",
             &_CreatePrimvarsMicromeshVersionAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshCompressedAttr",
             &This::GetPrimvarsMicromeshCompressedAttr)
        .def("CreatePrimvarsMicromeshCompressedAttr",
             &_CreatePrimvarsMicromeshCompressedAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshFloatScaleAttr",
             &This::GetPrimvarsMicromeshFloatScaleAttr)
        .def("CreatePrimvarsMicromeshFloatScaleAttr",
             &_CreatePrimvarsMicromeshFloatScaleAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshFloatBiasAttr",
             &This::GetPrimvarsMicromeshFloatBiasAttr)
        .def("CreatePrimvarsMicromeshFloatBiasAttr",
             &_CreatePrimvarsMicromeshFloatBiasAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshMinSubdivLevelAttr",
             &This::GetPrimvarsMicromeshMinSubdivLevelAttr)
        .def("CreatePrimvarsMicromeshMinSubdivLevelAttr",
             &_CreatePrimvarsMicromeshMinSubdivLevelAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshMaxSubdivLevelAttr",
             &This::GetPrimvarsMicromeshMaxSubdivLevelAttr)
        .def("CreatePrimvarsMicromeshMaxSubdivLevelAttr",
             &_CreatePrimvarsMicromeshMaxSubdivLevelAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshDirectionsAttr",
             &This::GetPrimvarsMicromeshDirectionsAttr)
        .def("CreatePrimvarsMicromeshDirectionsAttr",
             &_CreatePrimvarsMicromeshDirectionsAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshDirectionBoundsAttr",
             &This::GetPrimvarsMicromeshDirectionBoundsAttr)
        .def("CreatePrimvarsMicromeshDirectionBoundsAttr",
             &_CreatePrimvarsMicromeshDirectionBoundsAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshTriangleMappingsFormatAttr",
             &This::GetPrimvarsMicromeshTriangleMappingsFormatAttr)
        .def("CreatePrimvarsMicromeshTriangleMappingsFormatAttr",
             &_CreatePrimvarsMicromeshTriangleMappingsFormatAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshTriangleMappingsCountAttr",
             &This::GetPrimvarsMicromeshTriangleMappingsCountAttr)
        .def("CreatePrimvarsMicromeshTriangleMappingsCountAttr",
             &_CreatePrimvarsMicromeshTriangleMappingsCountAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshTriangleMappingsByteSizeAttr",
             &This::GetPrimvarsMicromeshTriangleMappingsByteSizeAttr)
        .def("CreatePrimvarsMicromeshTriangleMappingsByteSizeAttr",
             &_CreatePrimvarsMicromeshTriangleMappingsByteSizeAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshTriangleMappingsAttr",
             &This::GetPrimvarsMicromeshTriangleMappingsAttr)
        .def("CreatePrimvarsMicromeshTriangleMappingsAttr",
             &_CreatePrimvarsMicromeshTriangleMappingsAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshValueLayoutAttr",
             &This::GetPrimvarsMicromeshValueLayoutAttr)
        .def("CreatePrimvarsMicromeshValueLayoutAttr",
             &_CreatePrimvarsMicromeshValueLayoutAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshValueFrequencyAttr",
             &This::GetPrimvarsMicromeshValueFrequencyAttr)
        .def("CreatePrimvarsMicromeshValueFrequencyAttr",
             &_CreatePrimvarsMicromeshValueFrequencyAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshValueFormatAttr",
             &This::GetPrimvarsMicromeshValueFormatAttr)
        .def("CreatePrimvarsMicromeshValueFormatAttr",
             &_CreatePrimvarsMicromeshValueFormatAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshValueCountAttr",
             &This::GetPrimvarsMicromeshValueCountAttr)
        .def("CreatePrimvarsMicromeshValueCountAttr",
             &_CreatePrimvarsMicromeshValueCountAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshValueByteSizeAttr",
             &This::GetPrimvarsMicromeshValueByteSizeAttr)
        .def("CreatePrimvarsMicromeshValueByteSizeAttr",
             &_CreatePrimvarsMicromeshValueByteSizeAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshValuesAttr",
             &This::GetPrimvarsMicromeshValuesAttr)
        .def("CreatePrimvarsMicromeshValuesAttr",
             &_CreatePrimvarsMicromeshValuesAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshTriangleFlagsFormatAttr",
             &This::GetPrimvarsMicromeshTriangleFlagsFormatAttr)
        .def("CreatePrimvarsMicromeshTriangleFlagsFormatAttr",
             &_CreatePrimvarsMicromeshTriangleFlagsFormatAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshTriangleFlagsCountAttr",
             &This::GetPrimvarsMicromeshTriangleFlagsCountAttr)
        .def("CreatePrimvarsMicromeshTriangleFlagsCountAttr",
             &_CreatePrimvarsMicromeshTriangleFlagsCountAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshTriangleFlagsByteSizeAttr",
             &This::GetPrimvarsMicromeshTriangleFlagsByteSizeAttr)
        .def("CreatePrimvarsMicromeshTriangleFlagsByteSizeAttr",
             &_CreatePrimvarsMicromeshTriangleFlagsByteSizeAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshTriangleFlagsAttr",
             &This::GetPrimvarsMicromeshTriangleFlagsAttr)
        .def("CreatePrimvarsMicromeshTriangleFlagsAttr",
             &_CreatePrimvarsMicromeshTriangleFlagsAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshTriangleValueOffsetsAttr",
             &This::GetPrimvarsMicromeshTriangleValueOffsetsAttr)
        .def("CreatePrimvarsMicromeshTriangleValueOffsetsAttr",
             &_CreatePrimvarsMicromeshTriangleValueOffsetsAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshTriangleSubdivLevelsAttr",
             &This::GetPrimvarsMicromeshTriangleSubdivLevelsAttr)
        .def("CreatePrimvarsMicromeshTriangleSubdivLevelsAttr",
             &_CreatePrimvarsMicromeshTriangleSubdivLevelsAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshTriangleBlockFormatsAttr",
             &This::GetPrimvarsMicromeshTriangleBlockFormatsAttr)
        .def("CreatePrimvarsMicromeshTriangleBlockFormatsAttr",
             &_CreatePrimvarsMicromeshTriangleBlockFormatsAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshHistogramCountsAttr",
             &This::GetPrimvarsMicromeshHistogramCountsAttr)
        .def("CreatePrimvarsMicromeshHistogramCountsAttr",
             &_CreatePrimvarsMicromeshHistogramCountsAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshHistogramSubdivLevelsAttr",
             &This::GetPrimvarsMicromeshHistogramSubdivLevelsAttr)
        .def("CreatePrimvarsMicromeshHistogramSubdivLevelsAttr",
             &_CreatePrimvarsMicromeshHistogramSubdivLevelsAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshHistogramBlockFormatsAttr",
             &This::GetPrimvarsMicromeshHistogramBlockFormatsAttr)
        .def("CreatePrimvarsMicromeshHistogramBlockFormatsAttr",
             &_CreatePrimvarsMicromeshHistogramBlockFormatsAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshTriangleMinMaxsFormatAttr",
             &This::GetPrimvarsMicromeshTriangleMinMaxsFormatAttr)
        .def("CreatePrimvarsMicromeshTriangleMinMaxsFormatAttr",
             &_CreatePrimvarsMicromeshTriangleMinMaxsFormatAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshTriangleMinMaxsCountAttr",
             &This::GetPrimvarsMicromeshTriangleMinMaxsCountAttr)
        .def("CreatePrimvarsMicromeshTriangleMinMaxsCountAttr",
             &_CreatePrimvarsMicromeshTriangleMinMaxsCountAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshTriangleMinMaxsByteSizeAttr",
             &This::GetPrimvarsMicromeshTriangleMinMaxsByteSizeAttr)
        .def("CreatePrimvarsMicromeshTriangleMinMaxsByteSizeAttr",
             &_CreatePrimvarsMicromeshTriangleMinMaxsByteSizeAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshTriangleMinMaxsAttr",
             &This::GetPrimvarsMicromeshTriangleMinMaxsAttr)
        .def("CreatePrimvarsMicromeshTriangleMinMaxsAttr",
             &_CreatePrimvarsMicromeshTriangleMinMaxsAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshQuaternionMapAttr",
             &This::GetPrimvarsMicromeshQuaternionMapAttr)
        .def("CreatePrimvarsMicromeshQuaternionMapAttr",
             &_CreatePrimvarsMicromeshQuaternionMapAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))
        
        .def("GetPrimvarsMicromeshOffsetMapAttr",
             &This::GetPrimvarsMicromeshOffsetMapAttr)
        .def("CreatePrimvarsMicromeshOffsetMapAttr",
             &_CreatePrimvarsMicromeshOffsetMapAttr,
             (arg("defaultValue")=object(),
              arg("writeSparsely")=false))

        .def("__repr__", ::_Repr)
    ;

    _CustomWrapCode(cls);
}

// ===================================================================== //
// Feel free to add custom code below this line, it will be preserved by 
// the code generator.  The entry point for your custom code should look
// minimally like the following:
//
// WRAP_CUSTOM {
//     _class
//         .def("MyCustomMethod", ...)
//     ;
// }
//
// Of course any other ancillary or support code may be provided.
// 
// Just remember to wrap code in the appropriate delimiters:
// 'namespace {', '}'.
//
// ===================================================================== //
// --(BEGIN CUSTOM CODE)--

namespace {

WRAP_CUSTOM {


}

}
