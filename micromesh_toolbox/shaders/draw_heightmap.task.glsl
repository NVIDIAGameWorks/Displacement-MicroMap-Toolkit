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

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : require

#extension GL_NV_mesh_shader : enable

#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_8bit_storage : enable
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_shader_atomic_int64 : enable

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_NV_shader_subgroup_partitioned : require

#include "dh_scn_desc.h"
#include "dh_bindings.h"
#include "device_host.h"
#include "config.h"
#include "common_micromesh.h"
#include "common_micromesh_compressed.h"
#include "common_barymap.h"

layout(local_size_x = MICRO_GROUP_SIZE) in;

layout(push_constant) uniform RasterPushConstant_
{
  PushConstant pc;
};

/////////////////////////////////////
// UNIFORMS

// clang-format off
layout(buffer_reference, scalar) readonly buffer Vertices         { vec4 v[]; };
layout(buffer_reference, scalar) readonly buffer TexCoords        { vec2 v[]; };
layout(buffer_reference, scalar) readonly buffer Tangents         { vec2 v[]; };
layout(buffer_reference, scalar) readonly buffer Indices          { uvec3 i[]; };
layout(buffer_reference, scalar) readonly buffer Directions       { f16vec4 v[]; };
layout(buffer_reference, scalar) readonly buffer DirectionBounds  { vec2 v[]; };
layout(buffer_reference, scalar) readonly buffer TriWTIndices     { WatertightIndices v[]; };
layout(buffer_reference, scalar) readonly buffer DeviceMeshInfos  { DeviceMeshInfo i[]; };
layout(buffer_reference, scalar) readonly buffer DeviceBaryInfos  { DeviceBaryInfo i[]; };
layout(buffer_reference, scalar) readonly buffer MicromeshDataPtr { MicromeshData microdata; };
layout(buffer_reference, scalar) readonly buffer Materials        { GltfShadeMaterial m[]; };
layout(buffer_reference, scalar) readonly buffer InstanceInfos    { InstanceInfo i[]; };

layout(set = 0, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 0, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; } ;
layout(set = 0, binding = eTextures) uniform sampler2D[] texturesMap;
// clang-format on

//////////////////////////////////////////////////////////////////////////
// OUTPUT

taskNV out Task
{
  // The task shader processes 32 triangles at once collaboratively. This is the
  // starting index for a range of 32.
  uint baseTriangle;

  // Some base triangles may produce multiple mesh shaders. This array can be
  // used to work out how many mesh shaders were created per input triangle and
  // in turn the index of the mesh shader within each triangle's group.
  // - subdivLevel:   4 bits, 0-4
  // - edge0Decimate: 3 bits, 4-7   - relative to subdivLevel
  // - edge1Decimate: 3 bits, 7-10  - relative to subdivLevel
  // - edge2Decimate: 3 bits, 10-13 - relative to subdivLevel
  // - valid:         1 bits, 13-14
  uint16_t subdivLevelBits[MICRO_GROUP_SIZE];
}
OUT;

//////////////////////////////////////////////////////////////////////////

// Returns t along the edge {a, b} of the point closest to the near plane.
// E.g: 'x' on the following lines.
//       _.~|        /  _.~|
//  _.~'  / |      _/~'    |
// |     /  |     |/       |
//  '~._x   |     x'~._    |
//      '~._|    /     '~._|
float clipMinT(vec4 a, vec4 b)
{
  vec4  d              = b - a;
  float tNearPlane     = -(a.z + a.w) / (d.w + d.z);
  bool  intersectsNear = tNearPlane > 0.0 && tNearPlane < 1.0;
  float tNearest       = a.w < b.w ? 0.0 : 1.0;
  return intersectsNear ? tNearPlane : tNearest;
}

// Given clip space points a, b and interpolation t, returns the rate of change
// of x and y in NDC with respect to t. For example, point t on the following
// line will be nearer the viewer and have a greater pixel frequency due to
// perspective (as opposed to affine) interpolation. Imagine lines on a road
// going into the distance.
//       _.~|
//  _.~'  s |
// |     /  |
//  '~._t   |
//      '~._|
vec2 clipSpaceDNDCxyDt(vec4 a, vec4 b, float t)
{
  vec4  d = b - a;
  float r = a.w + t * d.w;
  return (d.xy * a.w - d.w * a.xy) / (r * r);
}

// Returns true if all points are on the outside of any clipping plane. This
// guarantees anything drawn between the points will not be on-screen.
bool earlyRejectPoints(in vec4[6] points)
{
  bvec3 allLow = bvec3(true, true, true);
  bvec3 allHigh = bvec3(true, true, true);
  for(int i = 0; i < 6; ++i)
  {
    // Cohen-Sutherland outcodes
    bvec3 low  = lessThan(points[i].xyz, vec3(-points[i].w));
    bvec3 high = greaterThan(points[i].xyz, vec3(points[i].w));
    allLow = bvec3(uvec3(allLow) & uvec3(low));
    allHigh = bvec3(uvec3(allHigh) & uvec3(high));
  }
  return any(allLow) || any(allHigh);
}

// Takes the closest point to the near plane (clipped) and use the derivative of
// edge interpolation with respect to pixels to compute the subdivision level.
// See clipMinT() and clipSpaceDNDCxyDt() respectively.
int triangleLOD(vec4 csPos0, vec4 csPos1, vec4 csPos2, float pixelsPerMicroEdge)
{
#if 1
  // Compute pixel derivatives in x and y at the nearest (most conservative
  // tessellation) point on each edge
  // Note: since there is no clipping to the frustum sides, the nearest point on
  // each edge may be outside the viewing volume and result in a large
  // subdivision over-estimate.
  vec2 e0PixelsXY = clipSpaceDNDCxyDt(csPos0, csPos1, clipMinT(csPos0, csPos1)) * 0.5 * frameInfo.resolution;
  vec2 e1PixelsXY = clipSpaceDNDCxyDt(csPos1, csPos2, clipMinT(csPos1, csPos2)) * 0.5 * frameInfo.resolution;
  vec2 e2PixelsXY = clipSpaceDNDCxyDt(csPos2, csPos0, clipMinT(csPos2, csPos0)) * 0.5 * frameInfo.resolution;
#else
  // Naive affine screen space delta without any clipping and ignoring w=0
  vec2 isPos0     = (csPos0.xy / csPos0.w) * 0.5 * frameInfo.resolution;
  vec2 isPos1     = (csPos1.xy / csPos1.w) * 0.5 * frameInfo.resolution;
  vec2 isPos2     = (csPos2.xy / csPos2.w) * 0.5 * frameInfo.resolution;
  vec2 e0PixelsXY = isPos1 - isPos0;
  vec2 e1PixelsXY = isPos2 - isPos1;
  vec2 e2PixelsXY = isPos0 - isPos2;
#endif

  // Compute per edge subdiv level and take the maximum
  int e0Subdiv = int(0.5 + log2(length(e0PixelsXY) / pixelsPerMicroEdge));
  int e1Subdiv = int(0.5 + log2(length(e1PixelsXY) / pixelsPerMicroEdge));
  int e2Subdiv = int(0.5 + log2(length(e2PixelsXY) / pixelsPerMicroEdge));
  return clamp(max(e0Subdiv, max(e1Subdiv, e2Subdiv)), 0, frameInfo.heightmapSubdivLevel);
}

void loadTriangle(in uint triangle, out vec3 wsPos0, out vec3 wsPos1, out vec3 wsPos2, out vec3 wsDir0, out vec3 wsDir1, out vec3 wsDir2)
{
  // Instances
  InstanceInfos instances = InstanceInfos(sceneDesc.instInfoAddress);
  InstanceInfo  instinfo  = instances.i[pc.instanceID];

  // Primitive meshes
  DeviceMeshInfos pInfos = DeviceMeshInfos(sceneDesc.deviceMeshInfoAddress);
  DeviceMeshInfo  pinfo  = pInfos.i[pc.primMeshID];

  Indices indices       = Indices(pinfo.triangleVertexIndexBuffer);
  uvec3   triangleIndex = indices.i[triangle];

  Vertices vertices = Vertices(pinfo.vertexPositionNormalBuffer);
  wsPos0            = mat4x3(instinfo.objectToWorld) * vec4(vertices.v[triangleIndex.x].xyz, 1.0);
  wsPos1            = mat4x3(instinfo.objectToWorld) * vec4(vertices.v[triangleIndex.y].xyz, 1.0);
  wsPos2            = mat4x3(instinfo.objectToWorld) * vec4(vertices.v[triangleIndex.z].xyz, 1.0);
  wsDir0            = mat3(instinfo.objectToWorld) * oct32_to_vec(floatBitsToUint(vertices.v[triangleIndex.x].w));
  wsDir1            = mat3(instinfo.objectToWorld) * oct32_to_vec(floatBitsToUint(vertices.v[triangleIndex.y].w));
  wsDir2            = mat3(instinfo.objectToWorld) * oct32_to_vec(floatBitsToUint(vertices.v[triangleIndex.z].w));
}

// Returns the subdivision level (rounded to nearest) for the given triangle
// index to guarantee microtriangle edges are at least pixelsPerMicroEdge.
int triangleIndexLOD(in uint triangle, in mat4 mvp, float pixelsPerMicroEdge)
{
  vec3 wsPos0, wsPos1, wsPos2;
  vec3 wsDir0, wsDir1, wsDir2;
  loadTriangle(triangle, wsPos0, wsPos1, wsPos2, wsDir0, wsDir1, wsDir2);
  vec4 csPos0 = mvp * vec4(wsPos0, 1.0);
  vec4 csPos1 = mvp * vec4(wsPos1, 1.0);
  vec4 csPos2 = mvp * vec4(wsPos2, 1.0);
  return triangleLOD(csPos0, csPos1, csPos2, pixelsPerMicroEdge);
}

// Computes clip space points at the minimum and maximum displacement and culls
// the triangle if all points are on the outside of any clipping plane. If not
// culled, computes the triangle's subdivision level to match
// pixelsPerMicroEdge.
bool triangleIndexCullAndLOD(in uint triangle, in mat4 mvp, float pixelsPerMicroEdge, out int subdivLevel)
{
  vec3 wsPos0, wsPos1, wsPos2;
  vec3 wsDir0, wsDir1, wsDir2;
  loadTriangle(triangle, wsPos0, wsPos1, wsPos2, wsDir0, wsDir1, wsDir2);

  // Create points at the minimum and maximum displacement
  Materials         mats           = Materials(sceneDesc.materialAddress);
  GltfShadeMaterial mat            = mats.m[pc.materialID];
  vec2              heightmapRange = vec2(0.0, 1.0);
  heightmapRange                   = heightmapRange * mat.khrDisplacementFactor + mat.khrDisplacementOffset;
  heightmapRange                   = heightmapRange * frameInfo.heightmapScale + frameInfo.heightmapOffset;
  vec4 points[6] = vec4[6](
    mvp * vec4(wsPos0 + wsDir0 * heightmapRange.x, 1.0),
    mvp * vec4(wsPos1 + wsDir1 * heightmapRange.x, 1.0),
    mvp * vec4(wsPos2 + wsDir2 * heightmapRange.x, 1.0),
    mvp * vec4(wsPos0 + wsDir0 * heightmapRange.y, 1.0),
    mvp * vec4(wsPos1 + wsDir1 * heightmapRange.y, 1.0),
    mvp * vec4(wsPos2 + wsDir2 * heightmapRange.y, 1.0)
  );

  // Cull the triangle if all points are on the same side of any clipping plane
  if(earlyRejectPoints(points))
  {
    return false;
  }

  // Match triangleIndexLOD()
  vec4 csPos0 = mvp * vec4(wsPos0, 1.0);
  vec4 csPos1 = mvp * vec4(wsPos1, 1.0);
  vec4 csPos2 = mvp * vec4(wsPos2, 1.0);
  subdivLevel = triangleLOD(csPos0, csPos1, csPos2, pixelsPerMicroEdge);

  // Don't cull the triangle
  return true;
}

void main()
{
  uint baseID     = gl_WorkGroupID.x * MICRO_GROUP_SIZE;
  uint laneID     = gl_SubgroupInvocationID;
  bool valid      = baseID + laneID < pc.triangleCount;
  uint relativeID = valid ? laneID : 0;
  uint triID      = baseID + relativeID;
  mat4 mvp        = frameInfo.proj * frameInfo.view;

  // Compute base triangle subdiv level
  const float pixelsPerMicroEdge = 3.0;
  int         subdivLevel        = 0;
  valid                          = valid && triangleIndexCullAndLOD(triID, mvp, pixelsPerMicroEdge, subdivLevel);
  uint subdivLevelBits           = bitfieldInsert(0, subdivLevel, 0, 4);

  // Compute adjacent triangle subdiv levels for edge decimation
  DeviceMeshInfos   pInfos    = DeviceMeshInfos(sceneDesc.deviceMeshInfoAddress);
  DeviceMeshInfo    pinfo     = pInfos.i[pc.primMeshID];
  WatertightIndices wtIndices = TriWTIndices(pinfo.triangleWatertightIndicesBuffer).v[triID];
  for(int edgeIdx = 0; edgeIdx < 3; ++edgeIdx)
  {
    int adjTri = wtIndices.adjacentTriangles[edgeIdx];
    if(adjTri == WATERTIGHT_INDICES_INVALID_VERTEX)
      continue;
    int  edgeSubdiv     = triangleIndexLOD(adjTri, mvp, pixelsPerMicroEdge);
    uint edgeSubdivDiff = subdivLevel - min(subdivLevel, edgeSubdiv);
    subdivLevelBits     = bitfieldInsert(subdivLevelBits, edgeSubdivDiff, 4 + edgeIdx * 3, 3);
  }

  // Set a special bit for zero-triangle output.
  // TODO: this is probably not needed as gl_TaskCountNV would not include them.
  subdivLevelBits = bitfieldInsert(subdivLevelBits, valid ? 1 : 0, 13, 1);

  OUT.baseTriangle            = baseID;
  OUT.subdivLevelBits[laneID] = uint16_t(subdivLevelBits);

  // Maximum mesh shader output is 64 triangles, which is subdiv 3. Higher
  // levels are created with multiple mesh shaders per triangle.
  uint outerSubdiv = subdivLevel > 3 ? subdivLevel - 3 : 0;
  uint subTriCount = valid ? (1 << (2 * outerSubdiv)) : 0;

  uint subgroupMeshShaderCount = subgroupAdd(subTriCount);
  if(laneID == 0)
  {
    gl_TaskCountNV = subgroupMeshShaderCount;
  }
}
