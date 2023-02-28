// Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#pragma once

#include "meshops_internal/meshops_context.h"
#include <nvmath/nvmath_types.h>
#include <stdint.h>
#include <vector>

namespace meshops {

// Uses Liani's algorithm to generate tangents, as used in Omniverse. This
// produces results similar to MikkTSpace with sometimes higher quality, and is
// much faster and supports polygons with arbitrary numbers of sides.
// TODO: Integrate more closely with MeshView
//
// Arguments (the varying/facevarying terminology is from USD):
// Inputs:
//   varyingIndices: Indices of points to define the topology. We don't need to
// know how many indices per face.
//   facevaryingIndices: Optional triangulated indices; if not null, then this
// is an indirection buffer to read from varyingIndices. If null, then
// varyingIndices are of a topology already made of triangles.
//   uniformIndices: Optional triangulated face indices. If set, it allows us
// to know which authored face a triangle belongs to. This is only required to
// safely partition workload across threads.
//   numVarying: In short, the number of points, or any other primvar defined
// with "varying" interpolation.
//   numFacevarying: The number of values in facevarying primvars.
//   numTriangles: The number of triangles.
//   inPosition: Vertex positions. Indexed by varyingIndices.
//   inNormal: Vertex normals. Indexed by facevaryingIndices or varyingIndices,
// depending on facevaryingN.
//   inUvs: Vertex texture coordinates. Indexed by facevaryingIndices or
// varyingIndices, depending on facevaryingTx.
// Outputs:
//   adjacencyMap: Varying-to-facevarying inverse map, which can be used to
// weld vertices later on. adjacencyMap[0] contains the max vertex valence; the
// next numVarying elements contain the prefix sum of the vertex valences; the
// next numVarying elements contain the elements contain the vertex valences,
// and the final numFacevarying elements contain, for each vertex, the indices
// within varyingIndices that pointed to it, concatenated. The total output
// length will be numVarying * 2 + numFacevarying + 1.
//   tangent: Tangent buffer, length numFacevarying. The w component is
// the bitangent sign, as defined in the glTF spec: B = cross(N, T) * sign
void createLianiTangents(Context         context,
                         const uint32_t* varyingIndices,
                         const uint32_t* facevaryingIndices,  //< optional
                         const uint32_t* uniformIndices,      //< optional

                         uint32_t numVarying,      // Limited to signed int size!
                         uint32_t numFacevarying,  // Limited to signed int size!
                         uint32_t numTriangles,    // Limited to signed int size!

                         const nvmath::vec3f* inNormal,
                         const nvmath::vec3f* inPosition,
                         const nvmath::vec2f* inUvs,

                         const bool facevaryingN,
                         const bool facevaryingTx,

                         // Results:
                         std::vector<uint32_t>& adjacencyMap,
                         nvmath::vec4f*         tangent);
}  // namespace meshops