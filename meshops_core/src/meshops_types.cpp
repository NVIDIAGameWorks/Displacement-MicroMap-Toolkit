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

#include <meshops/meshops_types.h>
#include <micromesh/micromesh_operations.h>
#include <micromesh/micromesh_utils.h>

namespace meshops {
//////////////////////////////////////////////////////////////////////////

micromesh::Result MeshTopologyData::buildFromIndicesAsIs(micromesh::OpContext ctx, size_t numIndices, const uint32_t* indices, size_t numVertices)
{
  const size_t numTriangles = numIndices / 3;

  // Allocate space for triangleEdges, vertexEdgeRanges, and vertexTriangleRanges
  topology = micromesh::MeshTopology{};

  // Make a copy of the index buffer - otherwise we run into lifetime issues
  triangleVertices.resize(numTriangles);
  memcpy(triangleVertices.data(), indices, triangleVertices.size() * sizeof(micromesh::Vector_uint32_3));
  micromesh::arraySetDataVec(topology.triangleVertices, triangleVertices);

  triangleEdges.resize(numTriangles);
  micromesh::arraySetDataVec(topology.triangleEdges, triangleEdges);

  topology.vertexEdgeRanges.count = numVertices;
  vertexEdgeRanges.resize(numVertices);
  topology.vertexEdgeRanges.data = vertexEdgeRanges.data();

  topology.vertexTriangleRanges.count = numVertices;
  vertexTriangleRanges.resize(numVertices);
  topology.vertexTriangleRanges.data = vertexTriangleRanges.data();

  // Fill those 3 arrays and get sizes for remaining MeshTopology arrays
  micromesh::Result result = micromesh::micromeshOpBuildMeshTopologyBegin(ctx, &topology);
  if(result != micromesh::Result::eSuccess)
    return result;

  // Allocate remaining output.
  vertexTriangleConnections.resize(topology.vertexTriangleConnections.count);
  topology.vertexTriangleConnections.data = vertexTriangleConnections.data();

  vertexEdgeConnections.resize(topology.vertexEdgeConnections.count);
  topology.vertexEdgeConnections.data = vertexEdgeConnections.data();

  edgeVertices.resize(topology.edgeVertices.count * 2);
  topology.edgeVertices.data = edgeVertices.data();

  edgeTriangleRanges.resize(topology.edgeTriangleRanges.count);
  topology.edgeTriangleRanges.data = edgeTriangleRanges.data();

  edgeTriangleConnections.resize(topology.edgeTriangleConnections.count);
  topology.edgeTriangleConnections.data = edgeTriangleConnections.data();

  // Okay, now build the topology!
  return micromesh::micromeshOpBuildMeshTopologyEnd(ctx, &topology);
}

micromesh::Result MeshTopologyData::buildFindingWatertightIndices(micromesh::OpContext             ctx,
                                                                  size_t                           numIndices,
                                                                  const uint32_t*                  indices,
                                                                  size_t                           numVertices,
                                                                  const micromesh::Vector_float_3* vertices,
                                                                  uint32_t                         verticesStride)
{
  std::vector<micromesh::Vector_uint32_3> uniqueTriangleVertices(numIndices / 3);

  micromesh::OpBuildMeshTopologyIndices_input input;
  micromesh::arraySetData(input.meshTriangleVertices, indices, numIndices / 3);
  micromesh::arraySetData(input.meshVertexPositions, vertices, numVertices);
  input.meshVertexPositions.byteStride = verticesStride;

  micromesh::OpBuildMeshTopologyIndices_output output;
  micromesh::arraySetDataVec(output.meshTopologyTriangleVertices, uniqueTriangleVertices);

  micromesh::Result result = micromesh::micromeshOpBuildMeshTopologyIndices(ctx, &input, &output);

  if(result != micromesh::Result::eSuccess)
  {
    return result;
  }

  return buildFromIndicesAsIs(ctx, uniqueTriangleVertices.size() * 3,
                              reinterpret_cast<uint32_t*>(uniqueTriangleVertices.data()), numVertices);
}
}  // namespace meshops