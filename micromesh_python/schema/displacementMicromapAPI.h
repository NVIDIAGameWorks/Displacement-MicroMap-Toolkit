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
#ifndef NVMICROMESH_GENERATED_DISPLACEMENTMICROMAPAPI_H
#define NVMICROMESH_GENERATED_DISPLACEMENTMICROMAPAPI_H

/// \file nvMicromesh/displacementMicromapAPI.h

#include "pxr/pxr.h"
#include "./api.h"
#include "pxr/usd/usd/apiSchemaBase.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/stage.h"
#include "./tokens.h"

#include "pxr/base/vt/value.h"

#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/matrix4d.h"

#include "pxr/base/tf/token.h"
#include "pxr/base/tf/type.h"

PXR_NAMESPACE_OPEN_SCOPE

class SdfAssetPath;

// -------------------------------------------------------------------------- //
// DISPLACEMENTMICROMAPAPI                                                    //
// -------------------------------------------------------------------------- //

/// \class NvMicromeshDisplacementMicromapAPI
///
/// DisplacementMicromapAPI Schema
///
class
#if defined(ARCH_COMPILER_GCC) && ARCH_COMPILER_GCC_MAJOR >= 4 || defined(ARCH_COMPILER_CLANG)
  NVMICROMESH_API
#endif
NvMicromeshDisplacementMicromapAPI : public UsdAPISchemaBase
{
public:
    /// Compile time constant representing what kind of schema this class is.
    ///
    /// \sa UsdSchemaKind
    static const UsdSchemaKind schemaKind = UsdSchemaKind::SingleApplyAPI;

    /// Construct a NvMicromeshDisplacementMicromapAPI on UsdPrim \p prim .
    /// Equivalent to NvMicromeshDisplacementMicromapAPI::Get(prim.GetStage(), prim.GetPath())
    /// for a \em valid \p prim, but will not immediately throw an error for
    /// an invalid \p prim
    explicit NvMicromeshDisplacementMicromapAPI(const UsdPrim& prim=UsdPrim())
        : UsdAPISchemaBase(prim)
    {
    }

    /// Construct a NvMicromeshDisplacementMicromapAPI on the prim held by \p schemaObj .
    /// Should be preferred over NvMicromeshDisplacementMicromapAPI(schemaObj.GetPrim()),
    /// as it preserves SchemaBase state.
    explicit NvMicromeshDisplacementMicromapAPI(const UsdSchemaBase& schemaObj)
        : UsdAPISchemaBase(schemaObj)
    {
    }

    /// Destructor.
    NVMICROMESH_API
    virtual ~NvMicromeshDisplacementMicromapAPI();

    /// Return a vector of names of all pre-declared attributes for this schema
    /// class and all its ancestor classes.  Does not include attributes that
    /// may be authored by custom/extended methods of the schemas involved.
    NVMICROMESH_API
    static const TfTokenVector &
    GetSchemaAttributeNames(bool includeInherited=true);

    /// Return a NvMicromeshDisplacementMicromapAPI holding the prim adhering to this
    /// schema at \p path on \p stage.  If no prim exists at \p path on
    /// \p stage, or if the prim at that path does not adhere to this schema,
    /// return an invalid schema object.  This is shorthand for the following:
    ///
    /// \code
    /// NvMicromeshDisplacementMicromapAPI(stage->GetPrimAtPath(path));
    /// \endcode
    ///
    NVMICROMESH_API
    static NvMicromeshDisplacementMicromapAPI
    Get(const UsdStagePtr &stage, const SdfPath &path);


    /// Returns true if this <b>single-apply</b> API schema can be applied to 
    /// the given \p prim. If this schema can not be a applied to the prim, 
    /// this returns false and, if provided, populates \p whyNot with the 
    /// reason it can not be applied.
    /// 
    /// Note that if CanApply returns false, that does not necessarily imply
    /// that calling Apply will fail. Callers are expected to call CanApply
    /// before calling Apply if they want to ensure that it is valid to 
    /// apply a schema.
    /// 
    /// \sa UsdPrim::GetAppliedSchemas()
    /// \sa UsdPrim::HasAPI()
    /// \sa UsdPrim::CanApplyAPI()
    /// \sa UsdPrim::ApplyAPI()
    /// \sa UsdPrim::RemoveAPI()
    ///
    NVMICROMESH_API
    static bool 
    CanApply(const UsdPrim &prim, std::string *whyNot=nullptr);

    /// Applies this <b>single-apply</b> API schema to the given \p prim.
    /// This information is stored by adding "DisplacementMicromapAPI" to the 
    /// token-valued, listOp metadata \em apiSchemas on the prim.
    /// 
    /// \return A valid NvMicromeshDisplacementMicromapAPI object is returned upon success. 
    /// An invalid (or empty) NvMicromeshDisplacementMicromapAPI object is returned upon 
    /// failure. See \ref UsdPrim::ApplyAPI() for conditions 
    /// resulting in failure. 
    /// 
    /// \sa UsdPrim::GetAppliedSchemas()
    /// \sa UsdPrim::HasAPI()
    /// \sa UsdPrim::CanApplyAPI()
    /// \sa UsdPrim::ApplyAPI()
    /// \sa UsdPrim::RemoveAPI()
    ///
    NVMICROMESH_API
    static NvMicromeshDisplacementMicromapAPI 
    Apply(const UsdPrim &prim);

protected:
    /// Returns the kind of schema this class belongs to.
    ///
    /// \sa UsdSchemaKind
    NVMICROMESH_API
    UsdSchemaKind _GetSchemaKind() const override;

private:
    // needs to invoke _GetStaticTfType.
    friend class UsdSchemaRegistry;
    NVMICROMESH_API
    static const TfType &_GetStaticTfType();

    static bool _IsTypedSchema();

    // override SchemaBase virtuals.
    NVMICROMESH_API
    const TfType &_GetTfType() const override;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHVERSION 
    // --------------------------------------------------------------------- //
    /// 
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:version` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshVersionAttr() const;

    /// See GetPrimvarsMicromeshVersionAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshVersionAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHCOMPRESSED 
    // --------------------------------------------------------------------- //
    /// true if micromap dat is compressed
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `bool primvars:micromesh:compressed` |
    /// | C++ Type | bool |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Bool |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshCompressedAttr() const;

    /// See GetPrimvarsMicromeshCompressedAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshCompressedAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHFLOATSCALE 
    // --------------------------------------------------------------------- //
    /// global scale
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `float primvars:micromesh:floatScale` |
    /// | C++ Type | float |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Float |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshFloatScaleAttr() const;

    /// See GetPrimvarsMicromeshFloatScaleAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshFloatScaleAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHFLOATBIAS 
    // --------------------------------------------------------------------- //
    /// global bias
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `float primvars:micromesh:floatBias` |
    /// | C++ Type | float |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Float |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshFloatBiasAttr() const;

    /// See GetPrimvarsMicromeshFloatBiasAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshFloatBiasAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHMINSUBDIVLEVEL 
    // --------------------------------------------------------------------- //
    /// minimum subdivision level in the micromap
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:minSubdivLevel` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshMinSubdivLevelAttr() const;

    /// See GetPrimvarsMicromeshMinSubdivLevelAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshMinSubdivLevelAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHMAXSUBDIVLEVEL 
    // --------------------------------------------------------------------- //
    /// maximum subdivision level in the micromap
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:maxSubdivLevel` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshMaxSubdivLevelAttr() const;

    /// See GetPrimvarsMicromeshMaxSubdivLevelAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshMaxSubdivLevelAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHDIRECTIONS 
    // --------------------------------------------------------------------- //
    /// per-vertex displacement directions
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `float3[] primvars:micromesh:directions` |
    /// | C++ Type | VtArray<GfVec3f> |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Float3Array |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshDirectionsAttr() const;

    /// See GetPrimvarsMicromeshDirectionsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshDirectionsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHDIRECTIONBOUNDS 
    // --------------------------------------------------------------------- //
    /// per-vertex displacement direction bounds
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `float2[] primvars:micromesh:directionBounds` |
    /// | C++ Type | VtArray<GfVec2f> |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Float2Array |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshDirectionBoundsAttr() const;

    /// See GetPrimvarsMicromeshDirectionBoundsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshDirectionBoundsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHTRIANGLEMAPPINGSFORMAT 
    // --------------------------------------------------------------------- //
    /// 
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:triangleMappingsFormat` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshTriangleMappingsFormatAttr() const;

    /// See GetPrimvarsMicromeshTriangleMappingsFormatAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshTriangleMappingsFormatAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHTRIANGLEMAPPINGSCOUNT 
    // --------------------------------------------------------------------- //
    /// 
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:triangleMappingsCount` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshTriangleMappingsCountAttr() const;

    /// See GetPrimvarsMicromeshTriangleMappingsCountAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshTriangleMappingsCountAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHTRIANGLEMAPPINGSBYTESIZE 
    // --------------------------------------------------------------------- //
    /// 
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:triangleMappingsByteSize` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshTriangleMappingsByteSizeAttr() const;

    /// See GetPrimvarsMicromeshTriangleMappingsByteSizeAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshTriangleMappingsByteSizeAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHTRIANGLEMAPPINGS 
    // --------------------------------------------------------------------- //
    /// maps triangle ID to base primitive (unpack per format)
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uchar[] primvars:micromesh:triangleMappings` |
    /// | C++ Type | VtArray<unsigned char> |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UCharArray |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshTriangleMappingsAttr() const;

    /// See GetPrimvarsMicromeshTriangleMappingsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshTriangleMappingsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHVALUELAYOUT 
    // --------------------------------------------------------------------- //
    /// 
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:valueLayout` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshValueLayoutAttr() const;

    /// See GetPrimvarsMicromeshValueLayoutAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshValueLayoutAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHVALUEFREQUENCY 
    // --------------------------------------------------------------------- //
    /// 
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:valueFrequency` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshValueFrequencyAttr() const;

    /// See GetPrimvarsMicromeshValueFrequencyAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshValueFrequencyAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHVALUEFORMAT 
    // --------------------------------------------------------------------- //
    /// 
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:valueFormat` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshValueFormatAttr() const;

    /// See GetPrimvarsMicromeshValueFormatAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshValueFormatAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHVALUECOUNT 
    // --------------------------------------------------------------------- //
    /// 
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:valueCount` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshValueCountAttr() const;

    /// See GetPrimvarsMicromeshValueCountAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshValueCountAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHVALUEBYTESIZE 
    // --------------------------------------------------------------------- //
    /// 
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:valueByteSize` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshValueByteSizeAttr() const;

    /// See GetPrimvarsMicromeshValueByteSizeAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshValueByteSizeAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHVALUES 
    // --------------------------------------------------------------------- //
    /// per-vertex displacement values
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uchar[] primvars:micromesh:values` |
    /// | C++ Type | VtArray<unsigned char> |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UCharArray |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshValuesAttr() const;

    /// See GetPrimvarsMicromeshValuesAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshValuesAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHTRIANGLEFLAGSFORMAT 
    // --------------------------------------------------------------------- //
    /// 
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:triangleFlagsFormat` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshTriangleFlagsFormatAttr() const;

    /// See GetPrimvarsMicromeshTriangleFlagsFormatAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshTriangleFlagsFormatAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHTRIANGLEFLAGSCOUNT 
    // --------------------------------------------------------------------- //
    /// 
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:triangleFlagsCount` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshTriangleFlagsCountAttr() const;

    /// See GetPrimvarsMicromeshTriangleFlagsCountAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshTriangleFlagsCountAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHTRIANGLEFLAGSBYTESIZE 
    // --------------------------------------------------------------------- //
    /// 
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:triangleFlagsByteSize` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshTriangleFlagsByteSizeAttr() const;

    /// See GetPrimvarsMicromeshTriangleFlagsByteSizeAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshTriangleFlagsByteSizeAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHTRIANGLEFLAGS 
    // --------------------------------------------------------------------- //
    /// per-triangle edge flags
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uchar[] primvars:micromesh:triangleFlags` |
    /// | C++ Type | VtArray<unsigned char> |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UCharArray |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshTriangleFlagsAttr() const;

    /// See GetPrimvarsMicromeshTriangleFlagsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshTriangleFlagsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHTRIANGLEVALUEOFFSETS 
    // --------------------------------------------------------------------- //
    /// 
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint[] primvars:micromesh:triangleValueOffsets` |
    /// | C++ Type | VtArray<unsigned int> |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UIntArray |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshTriangleValueOffsetsAttr() const;

    /// See GetPrimvarsMicromeshTriangleValueOffsetsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshTriangleValueOffsetsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHTRIANGLESUBDIVLEVELS 
    // --------------------------------------------------------------------- //
    /// 
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint[] primvars:micromesh:triangleSubdivLevels` |
    /// | C++ Type | VtArray<unsigned int> |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UIntArray |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshTriangleSubdivLevelsAttr() const;

    /// See GetPrimvarsMicromeshTriangleSubdivLevelsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshTriangleSubdivLevelsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHTRIANGLEBLOCKFORMATS 
    // --------------------------------------------------------------------- //
    /// 
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint[] primvars:micromesh:triangleBlockFormats` |
    /// | C++ Type | VtArray<unsigned int> |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UIntArray |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshTriangleBlockFormatsAttr() const;

    /// See GetPrimvarsMicromeshTriangleBlockFormatsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshTriangleBlockFormatsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHHISTOGRAMCOUNTS 
    // --------------------------------------------------------------------- //
    /// required for compressed data
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint[] primvars:micromesh:histogramCounts` |
    /// | C++ Type | VtArray<unsigned int> |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UIntArray |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshHistogramCountsAttr() const;

    /// See GetPrimvarsMicromeshHistogramCountsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshHistogramCountsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHHISTOGRAMSUBDIVLEVELS 
    // --------------------------------------------------------------------- //
    /// required for compressed data
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint[] primvars:micromesh:histogramSubdivLevels` |
    /// | C++ Type | VtArray<unsigned int> |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UIntArray |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshHistogramSubdivLevelsAttr() const;

    /// See GetPrimvarsMicromeshHistogramSubdivLevelsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshHistogramSubdivLevelsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHHISTOGRAMBLOCKFORMATS 
    // --------------------------------------------------------------------- //
    /// required for compressed data
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint[] primvars:micromesh:histogramBlockFormats` |
    /// | C++ Type | VtArray<unsigned int> |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UIntArray |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshHistogramBlockFormatsAttr() const;

    /// See GetPrimvarsMicromeshHistogramBlockFormatsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshHistogramBlockFormatsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHTRIANGLEMINMAXSFORMAT 
    // --------------------------------------------------------------------- //
    /// optional
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:triangleMinMaxsFormat` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshTriangleMinMaxsFormatAttr() const;

    /// See GetPrimvarsMicromeshTriangleMinMaxsFormatAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshTriangleMinMaxsFormatAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHTRIANGLEMINMAXSCOUNT 
    // --------------------------------------------------------------------- //
    /// optional
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:triangleMinMaxsCount` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshTriangleMinMaxsCountAttr() const;

    /// See GetPrimvarsMicromeshTriangleMinMaxsCountAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshTriangleMinMaxsCountAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHTRIANGLEMINMAXSBYTESIZE 
    // --------------------------------------------------------------------- //
    /// optional
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uint primvars:micromesh:triangleMinMaxsByteSize` |
    /// | C++ Type | unsigned int |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UInt |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshTriangleMinMaxsByteSizeAttr() const;

    /// See GetPrimvarsMicromeshTriangleMinMaxsByteSizeAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshTriangleMinMaxsByteSizeAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHTRIANGLEMINMAXS 
    // --------------------------------------------------------------------- //
    /// optional
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `uchar[] primvars:micromesh:triangleMinMaxs` |
    /// | C++ Type | VtArray<unsigned char> |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->UCharArray |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshTriangleMinMaxsAttr() const;

    /// See GetPrimvarsMicromeshTriangleMinMaxsAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshTriangleMinMaxsAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHQUATERNIONMAP 
    // --------------------------------------------------------------------- //
    /// quaternion map texture
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `asset primvars:micromesh:quaternionMap` |
    /// | C++ Type | SdfAssetPath |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Asset |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshQuaternionMapAttr() const;

    /// See GetPrimvarsMicromeshQuaternionMapAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshQuaternionMapAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // --------------------------------------------------------------------- //
    // PRIMVARSMICROMESHOFFSETMAP 
    // --------------------------------------------------------------------- //
    /// offset map texture
    ///
    /// | ||
    /// | -- | -- |
    /// | Declaration | `asset primvars:micromesh:offsetMap` |
    /// | C++ Type | SdfAssetPath |
    /// | \ref Usd_Datatypes "Usd Type" | SdfValueTypeNames->Asset |
    NVMICROMESH_API
    UsdAttribute GetPrimvarsMicromeshOffsetMapAttr() const;

    /// See GetPrimvarsMicromeshOffsetMapAttr(), and also 
    /// \ref Usd_Create_Or_Get_Property for when to use Get vs Create.
    /// If specified, author \p defaultValue as the attribute's default,
    /// sparsely (when it makes sense to do so) if \p writeSparsely is \c true -
    /// the default for \p writeSparsely is \c false.
    NVMICROMESH_API
    UsdAttribute CreatePrimvarsMicromeshOffsetMapAttr(VtValue const &defaultValue = VtValue(), bool writeSparsely=false) const;

public:
    // ===================================================================== //
    // Feel free to add custom code below this line, it will be preserved by 
    // the code generator. 
    //
    // Just remember to: 
    //  - Close the class declaration with }; 
    //  - Close the namespace with PXR_NAMESPACE_CLOSE_SCOPE
    //  - Close the include guard with #endif
    // ===================================================================== //
    // --(BEGIN CUSTOM CODE)--
};

PXR_NAMESPACE_CLOSE_SCOPE

#endif
