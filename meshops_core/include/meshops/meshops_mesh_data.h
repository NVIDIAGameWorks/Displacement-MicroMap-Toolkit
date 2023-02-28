//
// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

/**
 * @file mesh_data.hpp
 * @brief MeshData instantiates MeshAttributes from meshops_mesh_view.hpp with a
 * concrete std::vector backed mesh with its own data
 */

#pragma once

#include <meshops/meshops_mesh_view.h>

namespace meshops {

// Define a std::vector<T> that uses the default allocator, only taking one template argument
template <class T>
using DefaultAllocVector = std::vector<T>;

class MeshData : public MeshAttributes<DefaultAllocVector>
{
public:
  MeshData()                      = default;
  MeshData(const MeshData& other) = default;

  // Initialize vectors with a copy of the data pointed to by a mesh class
  // TODO: ideally this would be in the MeshAttributes constructor
  template <class OtherMeshType>
  MeshData(const OtherMeshType& mesh)
  {
    triangleVertices      = std::vector<nvmath::vec3ui>(mesh.triangleVertices.begin(), mesh.triangleVertices.end());
    vertexPositions       = std::vector<nvmath::vec3f>(mesh.vertexPositions.begin(), mesh.vertexPositions.end());
    vertexNormals         = std::vector<nvmath::vec3f>(mesh.vertexNormals.begin(), mesh.vertexNormals.end());
    vertexTexcoords0      = std::vector<nvmath::vec2f>(mesh.vertexTexcoords0.begin(), mesh.vertexTexcoords0.end());
    vertexTangents        = std::vector<nvmath::vec4f>(mesh.vertexTangents.begin(), mesh.vertexTangents.end());
    vertexDirections      = std::vector<nvmath::vec3f>(mesh.vertexDirections.begin(), mesh.vertexDirections.end());
    vertexDirectionBounds = std::vector<nvmath::vec2f>(mesh.vertexDirectionBounds.begin(), mesh.vertexDirectionBounds.end());
    triangleSubdivisionLevels =
        std::vector<uint16_t>(mesh.triangleSubdivisionLevels.begin(), mesh.triangleSubdivisionLevels.end());
    trianglePrimitiveFlags = std::vector<uint8_t>(mesh.trianglePrimitiveFlags.begin(), mesh.trianglePrimitiveFlags.end());
  }

  void resize(MeshAttributeFlags attribFlags, size_t triangleCount, size_t vertexCount)
  {
    if((attribFlags & eMeshAttributeTriangleVerticesBit) != 0)
    {
      triangleVertices.resize(triangleCount);
    }
    if((attribFlags & eMeshAttributeVertexPositionBit) != 0)
    {
      vertexPositions.resize(vertexCount);
    }
    if((attribFlags & eMeshAttributeVertexNormalBit) != 0)
    {
      vertexNormals.resize(vertexCount);
    }
    if((attribFlags & eMeshAttributeVertexTexcoordBit) != 0)
    {
      vertexTexcoords0.resize(vertexCount);
    }
    if((attribFlags & eMeshAttributeVertexTangentBit) != 0)
    {
      vertexTangents.resize(vertexCount);
    }
    if((attribFlags & eMeshAttributeVertexDirectionBit) != 0)
    {
      vertexDirections.resize(vertexCount);
    }
    if((attribFlags & eMeshAttributeVertexDirectionBoundsBit) != 0)
    {
      vertexDirectionBounds.resize(vertexCount);
    }
    if((attribFlags & eMeshAttributeVertexImportanceBit) != 0)
    {
      vertexImportance.resize(vertexCount);
    }
    if((attribFlags & eMeshAttributeTriangleSubdivLevelsBit) != 0)
    {
      triangleSubdivisionLevels.resize(triangleCount);
    }
    if((attribFlags & eMeshAttributeTrianglePrimitiveFlagsBit) != 0)
    {
      trianglePrimitiveFlags.resize(triangleCount);
    }
    assert(consistent());
  }
};

static_assert(std::is_constructible_v<MeshView, MeshData>);  // view of data
static_assert(std::is_constructible_v<MeshData, MeshView>);  // copy from view
static_assert(std::is_copy_constructible_v<MeshData>);

inline ResizableMeshView::resize_callback makeResizableMeshViewCallback(MeshData& resizableMesh)
{
  return [&resizableMesh](ResizableMeshView& meshView, MeshAttributeFlags attribFlags, size_t triangleCount, size_t vertexCount) {
    resizableMesh.resize(attribFlags, triangleCount, vertexCount);
    meshView.replace(resizableMesh, attribFlags);
  };
}

}  // namespace meshops
