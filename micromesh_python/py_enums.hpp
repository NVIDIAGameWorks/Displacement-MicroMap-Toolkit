#pragma once

// TODO: use public meshops API for attrib generation
#include <meshops_internal/umesh_util.hpp>

#include <pybind11/pybind11.h>

#include <vulkan/vulkan_core.h>

#include <meshops/meshops_operations.h>

#include <nvh/nvprint.hpp>

namespace py = pybind11;

enum PyVerbosity
{
    eErrors = LOGBITS_ERRORS,
    eWarnings = LOGBITS_WARNINGS,
    eInfo = LOGBITS_INFO
};

enum PySubdivMethod : int
{
  // Subdiv levels from file, if any
  //eCustomOrUniform,

  // Use the target subdiv level
  eUniform,

  // Generate subdiv levels
  eAdaptive3D,
  eAdaptiveUV,

  // Use subdiv levels from the file, error out if missing
  eCustom
};

enum PyTextureFormat
{
  eRGBA8Unorm  = VK_FORMAT_R8G8B8A8_UNORM,
  eRGBA16Unorm = VK_FORMAT_R16G16B16A16_UNORM,
  eR16Unorm    = VK_FORMAT_R16_UNORM
};

enum PyRemesherCurvatureMaxDistanceMode
{
    eSceneFraction,
    eWorldSpace
};

template<typename M>
void registerEnums(M& m)
{
    py::enum_<PyVerbosity>(m, "Verbosity", py::arithmetic())
        .value("Errors", PyVerbosity::eErrors)
        .value("Warnings", PyVerbosity::eWarnings)
        .value("Info", PyVerbosity::eInfo);

    py::enum_<PyRemesherCurvatureMaxDistanceMode>(m, "RemesherMaxDistanceMode", py::arithmetic())
        .value("SceneFraction", PyRemesherCurvatureMaxDistanceMode::eSceneFraction)
        .value("WorldSpace", PyRemesherCurvatureMaxDistanceMode::eWorldSpace);   

    py::enum_<PyTextureFormat>(m, "TextureFormat", py::arithmetic())
        .value("RGBA8Unorm", PyTextureFormat::eRGBA8Unorm)
        .value("RGBA16Unorm", PyTextureFormat::eRGBA16Unorm)
        .value("R16Unorm", PyTextureFormat::eR16Unorm);

    py::enum_<meshops::TextureType>(m, "TextureType", py::arithmetic())
        .value("Generic", meshops::TextureType::eGeneric)
        .value("NormalMap", meshops::TextureType::eNormalMap)
        .value("QuaternionMap", meshops::TextureType::eQuaternionMap)
        .value("OffsetMap", meshops::TextureType::eOffsetMap)
        .value("HeightMap", meshops::TextureType::eHeightMap);

    py::enum_<PySubdivMethod>(m, "SubdivMethod", py::arithmetic())
        .value("Uniform", PySubdivMethod::eUniform)
        .value("Adaptive3D", PySubdivMethod::eAdaptive3D)
        .value("AdaptiveUV", PySubdivMethod::eAdaptiveUV)
        .value("Custom", PySubdivMethod::eCustom);

    py::enum_<NormalReduceOp>(m, "NormalReduceOp", py::arithmetic())
        .value("NormalReduceLinear", NormalReduceOp::eNormalReduceLinear)
        .value("NormalReduceNormalizedLinear", NormalReduceOp::eNormalReduceNormalizedLinear)
        .value("NormalReduceTangent", NormalReduceOp::eNormalReduceTangent);

    py::enum_<meshops::TangentSpaceAlgorithm>(m, "TangentSpaceAlgorithm", py::arithmetic())
        .value("Invalid", meshops::TangentSpaceAlgorithm::eInvalid)
        .value("Lengyel", meshops::TangentSpaceAlgorithm::eLengyel)
        .value("Liani", meshops::TangentSpaceAlgorithm::eLiani)
        .value("MikkTSpace", meshops::TangentSpaceAlgorithm::eMikkTSpace)
        .value("Default", meshops::TangentSpaceAlgorithm::eDefault);

    py::enum_<micromesh::Result>(m, "MicromeshResult", py::arithmetic())
        .value("Success", micromesh::Result::eSuccess)
        .value("Failure", micromesh::Result::eFailure)
        .value("Continue", micromesh::Result::eContinue)
        .value("InvalidFrequency", micromesh::Result::eInvalidFrequency)
        .value("InvalidFormat", micromesh::Result::eInvalidFormat)
        .value("InvalidBlockFormat", micromesh::Result::eInvalidBlockFormat)
        .value("InvalidRange", micromesh::Result::eInvalidRange)
        .value("InvalidValue", micromesh::Result::eInvalidValue)
        .value("InvalidLayout", micromesh::Result::eInvalidLayout)
        .value("InvalidOperationOrder", micromesh::Result::eInvalidOperationOrder)
        .value("MismatchingInputEdgeValues", micromesh::Result::eMismatchingInputEdgeValues)
        .value("MismatchingOutputEdgeValues", micromesh::Result::eMismatchingOutputEdgeValues)
        .value("UnsupportedVersion", micromesh::Result::eUnsupportedVersion)
        .value("UnsupportedShaderCodeType", micromesh::Result::eUnsupportedShaderCodeType);    

    py::enum_<micromesh::Format>(m, "MicromeshFormat", py::arithmetic())
        .value("Undefined", micromesh::Format::eUndefined)
        .value("R8_unorm", micromesh::Format::eR8_unorm)
        .value("R8_snorm", micromesh::Format::eR8_snorm)
        .value("R8_uint", micromesh::Format::eR8_uint)
        .value("R8_sint", micromesh::Format::eR8_sint)
        .value("RG8_unorm", micromesh::Format::eRG8_unorm)
        .value("RG8_snorm", micromesh::Format::eRG8_snorm)
        .value("RG8_uint", micromesh::Format::eRG8_uint)
        .value("RG8_sint", micromesh::Format::eRG8_sint)
        .value("RGB8_unorm", micromesh::Format::eRGB8_unorm)
        .value("RGB8_snorm", micromesh::Format::eRGB8_snorm)
        .value("RGB8_uint", micromesh::Format::eRGB8_uint)
        .value("RGB8_sint", micromesh::Format::eRGB8_sint)
        .value("RGBA8_unorm", micromesh::Format::eRGBA8_unorm)
        .value("RGBA8_snorm", micromesh::Format::eRGBA8_snorm)
        .value("RGBA8_uint", micromesh::Format::eRGBA8_uint)
        .value("RGBA8_sint", micromesh::Format::eRGBA8_sint)
        .value("R16_unorm", micromesh::Format::eR16_unorm)
        .value("R16_snorm", micromesh::Format::eR16_snorm)
        .value("R16_uint", micromesh::Format::eR16_uint)
        .value("R16_sint", micromesh::Format::eR16_sint)
        .value("R16_sfloat", micromesh::Format::eR16_sfloat)
        .value("RG16_unorm", micromesh::Format::eRG16_unorm)
        .value("RG16_snorm", micromesh::Format::eRG16_snorm)
        .value("RG16_uint", micromesh::Format::eRG16_uint)
        .value("RG16_sint", micromesh::Format::eRG16_sint)
        .value("RG16_sfloat", micromesh::Format::eRG16_sfloat)
        .value("RGB16_unorm", micromesh::Format::eRGB16_unorm)
        .value("RGB16_snorm", micromesh::Format::eRGB16_snorm)
        .value("RGB16_uint", micromesh::Format::eRGB16_uint)
        .value("RGB16_sint", micromesh::Format::eRGB16_sint)
        .value("RGB16_sfloat", micromesh::Format::eRGB16_sfloat)
        .value("RGBA16_unorm", micromesh::Format::eRGBA16_unorm)
        .value("RGBA16_snorm", micromesh::Format::eRGBA16_snorm)
        .value("RGBA16_uint", micromesh::Format::eRGBA16_uint)
        .value("RGBA16_sint", micromesh::Format::eRGBA16_sint)
        .value("RGBA16_sfloat", micromesh::Format::eRGBA16_sfloat)
        .value("R32_uint", micromesh::Format::eR32_uint)
        .value("R32_sint", micromesh::Format::eR32_sint)
        .value("R32_sfloat", micromesh::Format::eR32_sfloat)
        .value("RG32_uint", micromesh::Format::eRG32_uint)
        .value("RG32_sint", micromesh::Format::eRG32_sint)
        .value("RG32_sfloat", micromesh::Format::eRG32_sfloat)
        .value("RGB32_uint", micromesh::Format::eRGB32_uint)
        .value("RGB32_sint", micromesh::Format::eRGB32_sint)
        .value("RGB32_sfloat", micromesh::Format::eRGB32_sfloat)
        .value("RGBA32_uint", micromesh::Format::eRGBA32_uint)
        .value("RGBA32_sint", micromesh::Format::eRGBA32_sint)
        .value("RGBA32_sfloat", micromesh::Format::eRGBA32_sfloat)
        .value("R64_uint", micromesh::Format::eR64_uint)
        .value("R64_sint", micromesh::Format::eR64_sint)
        .value("R64_sfloat", micromesh::Format::eR64_sfloat)
        .value("RG64_uint", micromesh::Format::eRG64_uint)
        .value("RG64_sint", micromesh::Format::eRG64_sint)
        .value("RG64_sfloat", micromesh::Format::eRG64_sfloat)
        .value("RGB64_uint", micromesh::Format::eRGB64_uint)
        .value("RGB64_sint", micromesh::Format::eRGB64_sint)
        .value("RGB64_sfloat", micromesh::Format::eRGB64_sfloat)
        .value("RGBA64_uint", micromesh::Format::eRGBA64_uint)
        .value("RGBA64_sint", micromesh::Format::eRGBA64_sint)
        .value("RGBA64_sfloat", micromesh::Format::eRGBA64_sfloat)
        .value("OpaC1_rx_uint_block", micromesh::Format::eOpaC1_rx_uint_block)
        .value("DispC1_r11_unorm_block", micromesh::Format::eDispC1_r11_unorm_block)
        .value("R11_unorm_pack16", micromesh::Format::eR11_unorm_pack16)
        .value("R11_unorm_packed_align32", micromesh::Format::eR11_unorm_packed_align32);

    py::enum_<bary::ValueLayout>(m, "ValueLayout", py::arithmetic())
        .value("Undefined", bary::ValueLayout::eUndefined)
        .value("TriangleUmajor", bary::ValueLayout::eTriangleUmajor)
        .value("TriangleBirdCurve", bary::ValueLayout::eTriangleBirdCurve);

    py::enum_<bary::ValueFrequency>(m, "ValueFrequency", py::arithmetic())
        .value("Undefined", bary::ValueFrequency::eUndefined)
        .value("PerVertex", bary::ValueFrequency::ePerVertex)
        .value("PerTriangle", bary::ValueFrequency::ePerTriangle);

    py::enum_<bary::Format>(m, "BaryFormat", py::arithmetic()) // Same as micromesh::Format it appears
        .value("Undefined", bary::Format::eUndefined)
        .value("R8_unorm", bary::Format::eR8_unorm)
        .value("R8_snorm", bary::Format::eR8_snorm)
        .value("R8_uint", bary::Format::eR8_uint)
        .value("R8_sint", bary::Format::eR8_sint)
        .value("RG8_unorm", bary::Format::eRG8_unorm)
        .value("RG8_snorm", bary::Format::eRG8_snorm)
        .value("RG8_uint", bary::Format::eRG8_uint)
        .value("RG8_sint", bary::Format::eRG8_sint)
        .value("RGB8_unorm", bary::Format::eRGB8_unorm)
        .value("RGB8_snorm", bary::Format::eRGB8_snorm)
        .value("RGB8_uint", bary::Format::eRGB8_uint)
        .value("RGB8_sint", bary::Format::eRGB8_sint)
        .value("RGBA8_unorm", bary::Format::eRGBA8_unorm)
        .value("RGBA8_snorm", bary::Format::eRGBA8_snorm)
        .value("RGBA8_uint", bary::Format::eRGBA8_uint)
        .value("RGBA8_sint", bary::Format::eRGBA8_sint)
        .value("R16_unorm", bary::Format::eR16_unorm)
        .value("R16_snorm", bary::Format::eR16_snorm)
        .value("R16_uint", bary::Format::eR16_uint)
        .value("R16_sint", bary::Format::eR16_sint)
        .value("R16_sfloat", bary::Format::eR16_sfloat)
        .value("RG16_unorm", bary::Format::eRG16_unorm)
        .value("RG16_snorm", bary::Format::eRG16_snorm)
        .value("RG16_uint", bary::Format::eRG16_uint)
        .value("RG16_sint", bary::Format::eRG16_sint)
        .value("RG16_sfloat", bary::Format::eRG16_sfloat)
        .value("RGB16_unorm", bary::Format::eRGB16_unorm)
        .value("RGB16_snorm", bary::Format::eRGB16_snorm)
        .value("RGB16_uint", bary::Format::eRGB16_uint)
        .value("RGB16_sint", bary::Format::eRGB16_sint)
        .value("RGB16_sfloat", bary::Format::eRGB16_sfloat)
        .value("RGBA16_unorm", bary::Format::eRGBA16_unorm)
        .value("RGBA16_snorm", bary::Format::eRGBA16_snorm)
        .value("RGBA16_uint", bary::Format::eRGBA16_uint)
        .value("RGBA16_sint", bary::Format::eRGBA16_sint)
        .value("RGBA16_sfloat", bary::Format::eRGBA16_sfloat)
        .value("R32_uint", bary::Format::eR32_uint)
        .value("R32_sint", bary::Format::eR32_sint)
        .value("R32_sfloat", bary::Format::eR32_sfloat)
        .value("RG32_uint", bary::Format::eRG32_uint)
        .value("RG32_sint", bary::Format::eRG32_sint)
        .value("RG32_sfloat", bary::Format::eRG32_sfloat)
        .value("RGB32_uint", bary::Format::eRGB32_uint)
        .value("RGB32_sint", bary::Format::eRGB32_sint)
        .value("RGB32_sfloat", bary::Format::eRGB32_sfloat)
        .value("RGBA32_uint", bary::Format::eRGBA32_uint)
        .value("RGBA32_sint", bary::Format::eRGBA32_sint)
        .value("RGBA32_sfloat", bary::Format::eRGBA32_sfloat)
        .value("R64_uint", bary::Format::eR64_uint)
        .value("R64_sint", bary::Format::eR64_sint)
        .value("R64_sfloat", bary::Format::eR64_sfloat)
        .value("RG64_uint", bary::Format::eRG64_uint)
        .value("RG64_sint", bary::Format::eRG64_sint)
        .value("RG64_sfloat", bary::Format::eRG64_sfloat)
        .value("RGB64_uint", bary::Format::eRGB64_uint)
        .value("RGB64_sint", bary::Format::eRGB64_sint)
        .value("RGB64_sfloat", bary::Format::eRGB64_sfloat)
        .value("RGBA64_uint", bary::Format::eRGBA64_uint)
        .value("RGBA64_sint", bary::Format::eRGBA64_sint)
        .value("RGBA64_sfloat", bary::Format::eRGBA64_sfloat)
        .value("OpaC1_rx_uint_block", bary::Format::eOpaC1_rx_uint_block)
        .value("DispC1_r11_unorm_block", bary::Format::eDispC1_r11_unorm_block)
        .value("R11_unorm_pack16", bary::Format::eR11_unorm_pack16)
        .value("R11_unorm_packed_align32", bary::Format::eR11_unorm_packed_align32);
}