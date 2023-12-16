//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

#pragma once

#include "meshops/meshops_mesh_view.h"
#include "nvmath/nvmath_types.h"
#include "tiny_gltf.h"
#include <gltf/NV_micromesh_extension_types.hpp>
#include <gltf/micromesh_util.hpp>
#include <cstdint>
#include <limits>
#include <meshops_internal/mesh_view.hpp>
#include <nvh/gltfscene.hpp>
#include <nvh/nvprint.hpp>
#include <bary/bary_types.h>

// Adaptors for ArrayView and gltf type info
// clang-format off
template <class T> struct tinygltfTypeInfo;
template <> struct tinygltfTypeInfo<int8_t>   { static const int componentType = TINYGLTF_COMPONENT_TYPE_BYTE;           static const int type = TINYGLTF_TYPE_SCALAR; };
template <> struct tinygltfTypeInfo<uint8_t>  { static const int componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;  static const int type = TINYGLTF_TYPE_SCALAR; };
template <> struct tinygltfTypeInfo<int16_t>  { static const int componentType = TINYGLTF_COMPONENT_TYPE_SHORT;          static const int type = TINYGLTF_TYPE_SCALAR; };
template <> struct tinygltfTypeInfo<uint16_t> { static const int componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT; static const int type = TINYGLTF_TYPE_SCALAR; };
template <> struct tinygltfTypeInfo<int32_t>  { static const int componentType = TINYGLTF_COMPONENT_TYPE_INT;            static const int type = TINYGLTF_TYPE_SCALAR; };
template <> struct tinygltfTypeInfo<uint32_t> { static const int componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;   static const int type = TINYGLTF_TYPE_SCALAR; };
template <> struct tinygltfTypeInfo<float>    { static const int componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;          static const int type = TINYGLTF_TYPE_SCALAR; };
template <> struct tinygltfTypeInfo<double>   { static const int componentType = TINYGLTF_COMPONENT_TYPE_DOUBLE;         static const int type = TINYGLTF_TYPE_SCALAR; };
template <> struct tinygltfTypeInfo<nvmath::vec2f>  { static const int componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;        static const int type = TINYGLTF_TYPE_VEC2; };
template <> struct tinygltfTypeInfo<nvmath::vec2i>  { static const int componentType = TINYGLTF_COMPONENT_TYPE_INT;          static const int type = TINYGLTF_TYPE_VEC2; };
template <> struct tinygltfTypeInfo<nvmath::vec2ui> { static const int componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT; static const int type = TINYGLTF_TYPE_VEC2; };
template <> struct tinygltfTypeInfo<nvmath::vec3f>  { static const int componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;        static const int type = TINYGLTF_TYPE_VEC3; };
template <> struct tinygltfTypeInfo<nvmath::vec3i>  { static const int componentType = TINYGLTF_COMPONENT_TYPE_INT;          static const int type = TINYGLTF_TYPE_VEC3; };
template <> struct tinygltfTypeInfo<nvmath::vec3ui> { static const int componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT; static const int type = TINYGLTF_TYPE_VEC3; };
template <> struct tinygltfTypeInfo<nvmath::vec4f>  { static const int componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;        static const int type = TINYGLTF_TYPE_VEC4; };
template <> struct tinygltfTypeInfo<nvmath::vec4i>  { static const int componentType = TINYGLTF_COMPONENT_TYPE_INT;          static const int type = TINYGLTF_TYPE_VEC4; };
template <> struct tinygltfTypeInfo<nvmath::vec4ui> { static const int componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT; static const int type = TINYGLTF_TYPE_VEC4; };
// clang-format on

constexpr const char* tinygltfTypeName(int type, int componentType);

template <class T>
constexpr const char* tinygltfTypeName()
{
  return tinygltfTypeName(T::type, T::componentType);
}

inline meshops::MeshSlice makeMeshSlice(const nvh::GltfPrimMesh& primMesh)
{
  meshops::MeshSlice result;
  result.triangleOffset = primMesh.firstIndex / 3;
  result.triangleCount  = primMesh.indexCount / 3;
  result.vertexOffset   = primMesh.vertexOffset;
  result.vertexCount    = primMesh.vertexCount;
  return result;
}

inline void writeMeshSlice(const meshops::MeshSlice& slice, nvh::GltfPrimMesh& primMesh)
{
  assert(slice.triangleOffset <= std::numeric_limits<decltype(primMesh.firstIndex)>::max());
  assert(slice.triangleCount <= std::numeric_limits<decltype(primMesh.indexCount)>::max());
  assert(slice.vertexOffset <= std::numeric_limits<decltype(primMesh.vertexOffset)>::max());
  assert(slice.vertexCount <= std::numeric_limits<decltype(primMesh.vertexCount)>::max());
  primMesh.firstIndex   = static_cast<uint32_t>(slice.triangleOffset * 3);
  primMesh.indexCount   = static_cast<uint32_t>(slice.triangleCount * 3);
  primMesh.vertexOffset = static_cast<uint32_t>(slice.vertexOffset);
  primMesh.vertexCount  = static_cast<uint32_t>(slice.vertexCount);
}

inline void writeMeshSetSlices(const std::vector<meshops::MeshSlice>& slices, nvh::GltfScene& gltfScene)
{
  for(size_t i = 0; i < slices.size(); ++i)
    writeMeshSlice(slices[i], gltfScene.m_primMeshes[i]);
}

// Create a MeshView with ArrayView pointers populated from the given bary
// ContentView. Bary MeshView pointers are const so a mutable view is not
// possible.
meshops::MeshView makeMeshView(const bary::ContentView* baryView, uint32_t groupIndex);

// Create a MeshView from a bary file and further slice it given
// NV_displacement_micromap offsets and the target mesh vertex and triangle
// count
meshops::MeshView makeMeshView(const bary::ContentView* baryView,
                               uint32_t                 groupIndex,
                               NV_displacement_micromap micromap,
                               size_t                   triangleCount,
                               size_t                   vertexCount);

// Create a MutableMeshView from a tinygltf::Model. This includes attributes
// from NV_micromap_tooling and NV_displacement_micromap. The fallback storage
// is required in case a MeshView cannot be created directly and some attribute
// conversion is needed. An optional (may be nullptr) bary ContentView can be
// given to fill missing attributes from its bary::MeshView.
meshops::MutableMeshView makeMutableMeshView(tinygltf::Model&           model,
                                             const tinygltf::Primitive& tinygltfPrim,
                                             meshops::DynamicMeshView&& fallbackStorage,
                                             const bary::ContentView*   baryView,
                                             uint32_t                   baryGroupIndex);

// Creates a MeshView from a tinygltf::Model. This includes attributes from
// NV_micromap_tooling and NV_displacement_micromap. The fallback storage is
// required in case a MeshView cannot be created directly and some attribute
// conversion is needed. An optional (may be nullptr) bary ContentView can be
// given to fill missing attributes from its bary::MeshView.
meshops::MeshView makeMeshView(const tinygltf::Model&     model,
                               const tinygltf::Primitive& tinygltfPrim,
                               meshops::DynamicMeshView&& fallbackStorage,
                               const bary::ContentView*   baryView,
                               uint32_t                   baryGroupIndex);

inline meshops::MutableMeshView makeMeshView(nvh::GltfScene& gltfScene)
{
  meshops::MutableMeshView result;
  // Two constructors for an explicit cast from glm to nvmath
  result.triangleVertices = meshops::ArrayView<nvmath::vec3ui>(meshops::ArrayView<uint32_t>(gltfScene.m_indices));
  result.vertexPositions  = meshops::ArrayView<nvmath::vec3f>(meshops::ArrayView<glm::vec3>(gltfScene.m_positions));
  result.vertexNormals    = meshops::ArrayView<nvmath::vec3f>(meshops::ArrayView<glm::vec3>(gltfScene.m_normals));
  result.vertexTexcoords0 = meshops::ArrayView<nvmath::vec2f>(meshops::ArrayView<glm::vec2>(gltfScene.m_texcoords0));
  result.vertexTangents   = meshops::ArrayView<nvmath::vec4f>(meshops::ArrayView<glm::vec4>(gltfScene.m_tangents));
  assert(result.consistent());
  return result;
}

inline meshops::MeshView makeMeshView(const nvh::GltfScene& gltfScene)
{
  meshops::MeshView result;
  // Two constructors for an explicit cast from glm to nvmath
  result.triangleVertices = meshops::ArrayView<const nvmath::vec3ui>(meshops::ArrayView<const uint32_t>(gltfScene.m_indices));
  result.vertexPositions = meshops::ArrayView<const nvmath::vec3f>(meshops::ArrayView<const glm::vec3>(gltfScene.m_positions));
  result.vertexNormals = meshops::ArrayView<const nvmath::vec3f>(meshops::ArrayView<const glm::vec3>(gltfScene.m_normals));
  result.vertexTexcoords0 = meshops::ArrayView<const nvmath::vec2f>(meshops::ArrayView<const glm::vec2>(gltfScene.m_texcoords0));
  result.vertexTangents = meshops::ArrayView<const nvmath::vec4f>(meshops::ArrayView<const glm::vec4>(gltfScene.m_tangents));
  assert(result.consistent());
  return result;
}

inline meshops::DynamicMeshView makeDynamicMeshView(nvh::GltfScene& gltfScene)
{
  meshops::DynamicMeshView result;
  // Two constructors for an explicit cast from glm to nvmath
  result.triangleVertices = meshops::DynamicArrayView<nvmath::vec3ui>(meshops::DynamicArrayView<uint32_t>(gltfScene.m_indices));
  result.vertexPositions = meshops::DynamicArrayView<nvmath::vec3f>(meshops::DynamicArrayView<glm::vec3>(gltfScene.m_positions));
  result.vertexNormals = meshops::DynamicArrayView<nvmath::vec3f>(meshops::DynamicArrayView<glm::vec3>(gltfScene.m_normals));
  result.vertexTexcoords0 =
      meshops::DynamicArrayView<nvmath::vec2f>(meshops::DynamicArrayView<glm::vec2>(gltfScene.m_texcoords0));
  result.vertexTangents = meshops::DynamicArrayView<nvmath::vec4f>(meshops::DynamicArrayView<glm::vec4>(gltfScene.m_tangents));
  assert(result.consistent());
  return result;
}

inline meshops::MutableMeshSetView makeMeshSetView(nvh::GltfScene& gltfScene)
{
  meshops::MutableMeshSetView result;
  result.flat = makeMeshView(gltfScene);
  result.slices.reserve(gltfScene.m_primMeshes.size());
  for(auto& primMesh : gltfScene.m_primMeshes)
  {
    result.slices.push_back(makeMeshSlice(primMesh));
  }
  return result;
}

inline meshops::MeshSetView makeMeshSetView(const nvh::GltfScene& gltfScene)
{
  meshops::MeshSetView result;
  result.flat = makeMeshView(gltfScene);
  result.slices.reserve(gltfScene.m_primMeshes.size());
  for(auto& primMesh : gltfScene.m_primMeshes)
  {
    result.slices.push_back(makeMeshSlice(primMesh));
  }
  return result;
}

inline meshops::DynamicMeshSetView makeDynamicMeshSetView(nvh::GltfScene& gltfScene)
{
  meshops::DynamicMeshSetView result;
  result.flat = makeDynamicMeshView(gltfScene);
  result.slices.reserve(gltfScene.m_primMeshes.size());
  for(auto& primMesh : gltfScene.m_primMeshes)
  {
    result.slices.push_back(makeMeshSlice(primMesh));
  }
  return result;
}
