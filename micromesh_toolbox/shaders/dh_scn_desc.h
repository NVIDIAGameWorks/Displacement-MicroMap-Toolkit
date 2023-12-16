/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef DH_SCN_DESC_H
#define DH_SCN_DESC_H 1

#ifdef __cplusplus
#include <nvmath/nvmath.h>
namespace shaders {
using mat4  = nvmath::mat4f;
using vec4  = nvmath::vec4f;
using vec3  = nvmath::vec3f;
using ivec3 = nvmath::vec3i;
using ivec2 = nvmath::vec2i;
#endif  // __cplusplus


struct InstanceInfo
{
  mat4 objectToWorld;
  mat4 worldToObject;
  int  materialID;
};

// Must be kept in sync with meshops::MeshAttributeFlags
// clang-format off
#if !defined(__cplusplus)
#define eMeshAttributeTriangleVerticesBit       (uint64_t(1) << 0)
#define eMeshAttributeTriangleSubdivLevelsBit   (uint64_t(1) << 1)
#define eMeshAttributeTrianglePrimitiveFlagsBit (uint64_t(1) << 2)
#define eMeshAttributeVertexPositionBit         (uint64_t(1) << 8)
#define eMeshAttributeVertexNormalBit           (uint64_t(1) << 9)
#define eMeshAttributeVertexTangentBit          (uint64_t(1) << 10)
#define eMeshAttributeVertexDirectionBit        (uint64_t(1) << 12)
#define eMeshAttributeVertexDirectionBoundsBit  (uint64_t(1) << 13)
#define eMeshAttributeVertexImportanceBit       (uint64_t(1) << 14)
#define eMeshAttributeVertexTexcoordBit         (uint64_t(1) << 16)
#endif
// clang-format on

struct DeviceMeshInfo
{
  uint64_t triangleVertexIndexBuffer;
  uint64_t triangleAttributesBuffer;
  uint64_t vertexPositionNormalBuffer;
  uint64_t vertexTangentSpaceBuffer;
  uint64_t vertexTexcoordBuffer;
  uint64_t vertexDirectionsBuffer;
  uint64_t vertexDirectionBoundsBuffer;
  uint64_t vertexImportanceBuffer;

  // WatertightIndices
  uint64_t triangleWatertightIndicesBuffer;

  // meshops::MeshAttributeFlags indicating which attributes are real or
  // generated/default-initialized.
  uint64_t sourceAttribFlags;

  // meshops::MeshAttributeFlags flags indicating which buffers are valid.
  uint64_t deviceAttribFlags;
};

struct DeviceBaryInfo
{
  // TODO: add something like this here
  //MicromeshData meshData;

  // DeviceMicromap
  // TODO: remove
  uint64_t baryValuesBuffer;

  // TODO: remove - made redundant by 'RBuffer baseTriangles'
  uint64_t baryTrianglesBuffer;

  // MicromeshSetCompressedVK::meshDatas[0].binding in DeviceMicromap::Raster
  // TODO: remove indirection
  uint64_t rasterMeshDataBindingBuffer;
};

#define WATERTIGHT_INDICES_INVALID_VERTEX -1

// Per-triangle indices to position-unique vertices, facilitating rendering
// heightmap displacement without cracks. This fixes cracks due to heightmap
// value differences across UV seams, but not cracks caused by normals pointing
// in different directions.
struct WatertightIndices
{
  // When tessellating and displacing a mesh with a heightmap, the bilinearly
  // interpolated values across a UV seam will necessarily be slightly
  // different. To avoid cracks, we fetch displacements along these edges too
  // and take the average. Values of -1 indicate the edge is shared and not to
  // take an average.
  // Edge ordering: {v0,v1}, {v1,v2}, {v2,v0}
  //
  ivec2 seamEdge[3];

  // Stop the cpp compiler tightly packing the next member
  ivec2 padding_;

  // Similarly, we need to average the displacements at shared vertices, however
  // there maybe a variable, and indeed unlimited number (e.g. tip of a cone).
  // We instead pick one vertex and sample height at only its UV coordinate.
  // Values of -1 indicate either the vertex is shared and not to use its UV, or
  // this is the vertex that others will snap to.
  ivec3 watertightCornerVertex;

  // When computing edge decimation for heightmap LODs, we need to know the LOD
  // of adjacent triangles.
  ivec3 adjacentTriangles;
};

struct SceneDescription
{
  uint64_t materialAddress;
  uint64_t instInfoAddress;
  uint64_t deviceMeshInfoAddress;
  uint64_t deviceBaryInfoAddress;
  uint64_t splitPartsVerticesAddress;
  uint64_t splitPartsIndicesAddress;
};

struct GltfShadeMaterial
{
  vec4 pbrBaseColorFactor;
  vec3 emissiveFactor;
  int  pbrBaseColorTexture;

  int   normalTexture;
  float normalTextureScale;
  int   shadingModel;
  float pbrRoughnessFactor;

  float pbrMetallicFactor;
  int   pbrMetallicRoughnessTexture;
  int   khrSpecularGlossinessTexture;
  int   khrDiffuseTexture;
  int   khrDisplacementTexture;

  vec4  khrDiffuseFactor;
  vec3  khrSpecularFactor;
  float khrGlossinessFactor;
  float khrDisplacementFactor;
  float khrDisplacementOffset;

  int   emissiveTexture;
  int   alphaMode;
  float alphaCutoff;
};

#ifdef __cplusplus
}  // namespace shaders
#endif

#endif
