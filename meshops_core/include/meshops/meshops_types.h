//
// SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: LicenseRef-NvidiaProprietary
//
// NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
// property and proprietary rights in and to this material, related
// documentation and any modifications thereto. Any use, reproduction,
// disclosure or distribution of this material and related documentation
// without an express license agreement from NVIDIA CORPORATION or
// its affiliates is strictly prohibited.
//

#pragma once

#include "meshops_api.h"
#include "meshops_mesh_view.h"
#include <micromesh/micromesh_types.h>

namespace meshops {

template <typename Tinfo, typename Tview>
void arrayInfoTypedFromView(Tinfo& info, Tview& view)
{
  static_assert(sizeof(typename Tinfo::value_type) == sizeof(typename Tview::value_type), "value_type size mismatch");
  info.data       = const_cast<void*>(reinterpret_cast<const void*>(view.data()));
  info.byteStride = uint32_t(view.stride());
  info.count      = view.size();
}

// Simplified C++ class for allocating and building a MeshTopology from an
// index buffer.
struct MeshTopologyData
{
  micromesh::MeshTopology topology;

  std::vector<micromesh::Vector_uint32_3> triangleVertices;
  std::vector<micromesh::Vector_uint32_3> triangleEdges;
  std::vector<micromesh::Range32>         vertexEdgeRanges;
  std::vector<micromesh::Range32>         vertexTriangleRanges;
  std::vector<uint32_t>                   vertexTriangleConnections;
  std::vector<uint32_t>                   vertexEdgeConnections;
  std::vector<uint32_t>                   edgeVertices;
  std::vector<micromesh::Range32>         edgeTriangleRanges;
  std::vector<uint32_t>                   edgeTriangleConnections;

  // Returns an array of triangle indices that reference the given vertex index
  ArrayView<uint32_t> getVertexTriangles(uint32_t vertIdx)
  {
    auto range = vertexTriangleRanges[vertIdx];
    return ArrayView(vertexTriangleConnections).slice(range.first, range.count);
  }

  // Returns an array of edge indices that reference the given vertex index
  ArrayView<uint32_t> getVertexEdges(uint32_t vertIdx)
  {
    auto range = vertexEdgeRanges[vertIdx];
    return ArrayView(vertexEdgeConnections).slice(range.first, range.count);
  }

  // Returns an array of triangle indices that reference the given edge index
  ArrayView<uint32_t> getEdgeTriangles(uint32_t vertIdx)
  {
    auto range = edgeTriangleRanges[vertIdx];
    return ArrayView(edgeTriangleConnections).slice(range.first, range.count);
  }


  // Generates unique vertex indices based on positions before passing them into buildFromUnique.
  // Positions are expected to perfectly match in binary representation, otherwise watertightness
  // cannot be ensured for operations depending on this topology data.
  micromesh::Result buildFindingWatertightIndices(micromesh::OpContext             ctx,
                                                  size_t                           numIndices,
                                                  const uint32_t*                  indices,
                                                  size_t                           numVertices,
                                                  const micromesh::Vector_float_3* vertices,
                                                  uint32_t verticesStride = uint32_t(sizeof(micromesh::Vector_float_3)));

  // result may not be watertight if indices contain split vertices (same position, but other different attributes)
  micromesh::Result buildFromIndicesAsIs(micromesh::OpContext ctx, size_t numIndices, const uint32_t* indices, size_t numVertices);

  uint64_t getVertexCount() const { return vertexEdgeRanges.size(); }
  uint64_t getTriangleCount() const { return triangleVertices.size(); }

  operator const micromesh::MeshTopology*() const { return &topology; }
};

// Tangent generation algorithm.
enum class TangentSpaceAlgorithm : uint32_t
{
  eInvalid,
  eLengyel,     // Uses Lengyel's tangent algorithm from FGED volume 2; this is fast and the default in nvpro_core.
  eLiani,       // Uses a new tangent generation developed by Max Liani; see liani_tangents.hpp for more info.
  eMikkTSpace,  // Uses glTF's recommended tangent generation algorithm, but can be slow.
  eDefault = eLiani
};

// Functions for converting between a TangentSpaceAlgorithm and its string
// representation, e.g. "mikktspace" for TangentSpaceAlgorithm::eMikkTSpace.
MESHOPS_API TangentSpaceAlgorithm MESHOPS_CALL tangentAlgorithmFromName(const char* name);
MESHOPS_API const char* MESHOPS_CALL           getTangentAlgorithmName(TangentSpaceAlgorithm algorithm);

}  // namespace meshops
