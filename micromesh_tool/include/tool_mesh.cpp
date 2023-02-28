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

namespace micromesh_tool {


ToolMesh::ToolMesh(tinygltf::Model*         model,
                   tinygltf::Mesh*          mesh,
                   tinygltf::Primitive*     primitive,
                   nvmath::mat4f            firstInstanceTransform,
                   const bary::ContentView* baryMeshView)
    : m_aux()
    , m_view(makeMutableMeshView(*model, (const tinygltf::Primitive&)*primitive, meshops::DynamicMeshView(m_aux), baryMeshView, 0),
             makeResizableMeshViewCallback(m_aux))
    , m_firstInstanceTransform(firstInstanceTransform)
    , m_gltfPrimitive(primitive)
{
  assert(mesh);
  assert(m_gltfPrimitive);

  // ToolMicromap should provide individual ContentViews for each group.
  assert(!baryMeshView || baryMeshView->basic.groupsCount == 1);

  // Copy any bary relations from the extension on the primitve
  NV_displacement_micromap displacement;
  if(getPrimitiveDisplacementMicromap(*m_gltfPrimitive, displacement) && displacement.micromap != -1)
  {
    m_relations.bary      = displacement.micromap;
    m_relations.group     = displacement.groupIndex;
    m_relations.mapOffset = displacement.mapOffset;
  }

  m_relations.material = primitive->material;
  m_meta.name          = mesh->name;
}

bool ToolMesh::isOriginalData() const
{
  // If any view attributes are pointing to m_aux, the data is not from the original source.
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

nvmath::mat4f ToolMesh::firstInstanceTransform() const
{
  return m_firstInstanceTransform;
}

}  // namespace micromesh_tool
