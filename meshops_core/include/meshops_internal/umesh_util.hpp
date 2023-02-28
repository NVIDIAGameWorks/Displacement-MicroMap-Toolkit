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

#include "nvmath/nvmath_types.h"
#include <micromesh/micromesh_operations.h>
#include <micromesh/micromesh_types.h>
#include <micromesh/micromesh_utils.h>
#include <nvmath/nvmath.h>
#include <vector>
#include <meshops/meshops_operations.h>
#include <meshops/meshops_mesh_view.h>
#include <meshops_internal/mesh_view.hpp>
#include <bary/bary_types.h>

template <class T>
T baryInterp(const T& a, const T& b, const T& c, const glm::vec3& baryCoord)
{
  return a * baryCoord.x + b * baryCoord.y + c * baryCoord.z;
}

template <class Vector>
typename Vector::value_type baryInterp(const Vector& attr, const nvmath::vec3ui& tri, const nvmath::vec3f& baryCoord)
{
  return baryInterp(attr[tri.x], attr[tri.y], attr[tri.z], baryCoord);
}

// Sort a triangle's vertex indices so that any rounding in baryInterp is consistent along tessellated edges
inline void stabilizeTriangleVerticesOrder(nvmath::vec3ui& triVertices, nvmath::vec3f& baryCoord)
{
  if(triVertices.y < triVertices.x)
  {
    std::swap(triVertices.y, triVertices.x);
    std::swap(baryCoord.y, baryCoord.x);
  }
  if(triVertices.z < triVertices.y)
  {
    std::swap(triVertices.z, triVertices.y);
    std::swap(baryCoord.z, baryCoord.y);
  }
  if(triVertices.y < triVertices.x)
  {
    std::swap(triVertices.y, triVertices.x);
    std::swap(baryCoord.y, baryCoord.x);
  }
}

// For both micromesh and heightmap displacement, we may generate smooth normals to use as displacement vectors. This
// option affects the length of the generated vectors at seams, where a new direction is computed. When used for
// heightmap displacement the geometry may change, if we don't normalize after interpolation anyway. When used for
// micromesh displacement the baking bounds may change but the target highres geometry is fixed.
enum NormalReduceOp
{
  // Dull/cut corners by linearly interpolating normalized normals
  eNormalReduceLinear,

  // Rounded corners by normalizing again, after interpolating
  eNormalReduceNormalizedLinear,

  // Sharp corners - preserves heightmap heights relative to surfaces at seams at the cost of stretching the geometry.
  // Affects direction too.
  eNormalReduceTangent,
};

void makeDisplacementDirections(const meshops::MeshView&          meshView,
                                const micromesh::MeshTopology&    topology,
                                meshops::ArrayView<nvmath::vec3f> outDisplacementDirections,
                                NormalReduceOp                    normalReduceOp);

// Compute per-triangle tessellation factor based on UV edge length in heightmap texels
void computeSubdivisionLevelsMatchingHeightmap(const meshops::MeshView&        meshView,
                                               nvmath::vec2ui                  heightmapSize,
                                               int32_t                         levelBias,
                                               uint32_t                        maxSubdivLevel,
                                               const meshops::MutableMeshView& result);

micromesh::Result sanitizeSubdivisionLevels(micromesh::OpContext           context,
                                            const micromesh::MeshTopology& topology,
                                            const meshops::MeshView&       meshView,
                                            meshops::ArrayView<uint16_t>   outSubdivisionLevels,
                                            meshops::ArrayView<uint8_t>    outEdgeFlags,
                                            uint32_t                       maxSubdivLevel);


bool tessellateQuads(int targetSubdivisionDiff, const std::vector<nvmath::vec2ui>& meshHeightmapSizes, meshops::DynamicMeshSetView& meshSet);

micromesh::Result tessellateMesh(micromesh::OpContext      context,
                                 const meshops::MeshView&  meshView,
                                 uint32_t                  maxSubdivLevel,
                                 meshops::DynamicMeshView& result);

// Creates a mesh line primitives for debugging displacement values
micromesh::Result generateDisplacementLines(micromesh::OpContext        context,
                                            const meshops::MeshView&    meshView,
                                            const bary::BasicView&      basic,
                                            const bary::Group&          baryGroup,
                                            std::vector<uint32_t>&      indices,
                                            std::vector<nvmath::vec3f>& positions,
                                            const float*                displacements);

// Not all operations in generateMeshAttributes need a topology, which is
// expensive to create. This method returns the required attributes that do not
// already exist that also need topology. If any are non-zero, topology must be
// provided to generateMeshAttributes.
meshops::MeshAttributeFlags generationRequiresTopology(meshops::MeshAttributeFlags existing, meshops::MeshAttributeFlags required);

// TODO: clean up inconsistent IO such as subdivisionLevelSettings and maxSubdivLevel
micromesh::Result generateMeshAttributes(meshops::Context                           context,
                                         meshops::MeshAttributeFlags                meshAttrFlags,
                                         meshops::OpGenerateSubdivisionLevel_input* subdivisionLevelSettings,
                                         const micromesh::MeshTopology*             topology,
                                         meshops::ResizableMeshView&                meshView,
                                         uint32_t&                                  maxSubdivLevel,
                                         NormalReduceOp                             directionsGenOp,
                                         meshops::TangentSpaceAlgorithm             tangentAlgorithm);
