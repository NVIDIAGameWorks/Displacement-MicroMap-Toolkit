/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "tool_mesh.hpp"
#include <nvh/gltfscene.hpp>
#include <mesh_view_conv.hpp>
#include <nvmath/nvmath_types.h>

namespace micromesh_tool {


ToolMesh::Relations::Relations(const tinygltf::Primitive& tinygltfPrimitive)
{
  // Copy any bary relations from the extension on the primitve. If baryMeshView
  // is null, assume the micromap references are invalid.
  NV_displacement_micromap displacement;
  if(getPrimitiveDisplacementMicromap(tinygltfPrimitive, displacement) && displacement.micromap != -1)
  {
    bary      = displacement.micromap;
    group     = displacement.groupIndex;
    mapOffset = displacement.mapOffset;
  }

  material = tinygltfPrimitive.material;
}

ToolMesh::Meta::Meta(const tinygltf::Mesh& tinygltfMesh)
    : name(tinygltfMesh.name)
{
}

ToolMesh::ToolMesh(tinygltf::Model*           model,
                   const Relations&           relations,
                   const Meta&                meta,
                   const tinygltf::Primitive* primitive,
                   const bary::ContentView*   baryMeshView)
    : m_aux()
    , m_view(makeMutableMeshView(*model, (const tinygltf::Primitive&)*primitive, meshops::DynamicMeshView(m_aux), baryMeshView, 0),
             makeResizableMeshViewCallback(m_aux))
    , m_gltfPrimitive(primitive)
    , m_relations(relations)
    , m_meta(meta)
{
  assert(m_gltfPrimitive);

  // The bary mesh view must be provided if relations refers to a bary file
  assert(relations.bary == -1 || baryMeshView);

  // ToolMicromap should provide individual ContentViews for each group.
  assert(!baryMeshView || baryMeshView->basic.groupsCount == 1);

  // HACK: isOriginalData() works by checking to see if a view().resize() has
  // changed its pointer from the tinygltf buffer to m_aux arrays. This also
  // works for the case when a resize is used to clear an array. However, to
  // handle the case where the inital arrays are empty a different value to
  // nullptr is needed.
  // TODO: this is dangerous because anything that blindly uses .data() might
  // check for a nullptr and assume it's valid without checking the size.
  char* notNull = nullptr;
  ++notNull;
  if(!m_view.triangleVertices.data())
    m_view.triangleVertices = meshops::ArrayView{reinterpret_cast<nvmath::vec3ui*>(notNull), 0};
  if(!m_view.vertexPositions.data())
    m_view.vertexPositions = meshops::ArrayView{reinterpret_cast<nvmath::vec3f*>(notNull), 0};
  if(!m_view.vertexNormals.data())
    m_view.vertexNormals = meshops::ArrayView{reinterpret_cast<nvmath::vec3f*>(notNull), 0};
  if(!m_view.vertexTexcoords0.data())
    m_view.vertexTexcoords0 = meshops::ArrayView{reinterpret_cast<nvmath::vec2f*>(notNull), 0};
  if(!m_view.vertexTangents.data())
    m_view.vertexTangents = meshops::ArrayView{reinterpret_cast<nvmath::vec4f*>(notNull), 0};
  if(!m_view.vertexDirections.data())
    m_view.vertexDirections = meshops::ArrayView{reinterpret_cast<nvmath::vec3f*>(notNull), 0};
  if(!m_view.vertexDirectionBounds.data())
    m_view.vertexDirectionBounds = meshops::ArrayView{reinterpret_cast<nvmath::vec2f*>(notNull), 0};
  if(!m_view.vertexImportance.data())
    m_view.vertexImportance = meshops::ArrayView{reinterpret_cast<float*>(notNull), 0};
  if(!m_view.triangleSubdivisionLevels.data())
    m_view.triangleSubdivisionLevels = meshops::ArrayView{reinterpret_cast<uint16_t*>(notNull), 0};
  if(!m_view.trianglePrimitiveFlags.data())
    m_view.trianglePrimitiveFlags = meshops::ArrayView{reinterpret_cast<uint8_t*>(notNull), 0};
}

bool ToolMesh::isOriginalData() const
{
  // If any view attributes are pointing to m_aux, the data is not from the
  // original source. Empty arrays will have a pointer to 0x1 to differentiate
  // themselves from empty m_aux vectors. See ToolMesh().
  if(m_view.triangleVertices.data() == m_aux.triangleVertices.data())
    return false;
  if(m_view.vertexPositions.data() == m_aux.vertexPositions.data())
    return false;
  if(m_view.vertexNormals.data() == m_aux.vertexNormals.data())
    return false;
  if(m_view.vertexTexcoords0.data() == m_aux.vertexTexcoords0.data())
    return false;
  if(m_view.vertexTangents.data() == m_aux.vertexTangents.data())
    return false;
  if(m_view.vertexDirections.data() == m_aux.vertexDirections.data())
    return false;
  if(m_view.vertexDirectionBounds.data() == m_aux.vertexDirectionBounds.data())
    return false;
  if(m_view.vertexImportance.data() == m_aux.vertexImportance.data())
    return false;
  if(m_view.triangleSubdivisionLevels.data() == m_aux.triangleSubdivisionLevels.data())
    return false;
  if(m_view.trianglePrimitiveFlags.data() == m_aux.trianglePrimitiveFlags.data())
    return false;
  return true;
}

}  // namespace micromesh_tool
