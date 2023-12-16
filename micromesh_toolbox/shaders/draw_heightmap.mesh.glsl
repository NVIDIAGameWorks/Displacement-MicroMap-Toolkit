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
#extension GL_EXT_debug_printf : enable

#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_nonuniform_qualifier : enable
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
#extension GL_KHR_shader_subgroup_shuffle_relative : require

#include "device_host.h"
#include "dh_bindings.h"
#include "dh_scn_desc.h"
#include "config.h"
#include "common_micromesh.h"
#include "common_micromesh_compressed.h"
#include "common_barymap.h"
#include "raster_anisotropy.glsl"
#include "nvvkhl/shaders/func.glsl"

layout(local_size_x = MICRO_GROUP_SIZE) in;
layout(max_primitives = MICRO_MESHLET_PRIMITIVES, max_vertices = MICRO_MESHLET_VERTICES, triangles) out;

layout(push_constant) uniform RasterPushConstant_
{
  PushConstant pc;
};

layout(constant_id = 0) const int CONST_SHADE_MODE = 0;

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

layout(buffer_reference, scalar) readonly buffer MicroVertices { MicromeshSTriVertex v[]; };
layout(buffer_reference, scalar) readonly buffer MicroIndices  { u8vec4 v[]; };

layout(set = 0, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 0, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; };
layout(set = 0, binding = eTextures) uniform sampler2D[] texturesMap;
// clang-format on

//////////////////////////////////////////////////////////////////////////
// INPUT

taskNV in Task
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
IN;

//////////////////////////////////////////////////////////////////////////
// OUTPUT

layout(location = 0) out Interpolants
{
  // world-space vertex posiiton
  vec3 pos;
  vec3 bary;
}
OUT[];

layout(location = 4) perprimitiveNV out PrimInterpolants
{
  uint shadeModeValue;
}
POUT[];

//////////////////////////////////////////////////////////////////////////

#include "micromesh_utils.glsl"

//////////////////////////////////////////////////////////////////////////

// Return the number of edges vertWUV lies on, i.e. count(vertWUV.* == 0)
// Will be 2 at corners, 1 in the middle of an edge and 0 otherwise
int baryOnEdgeCount(uvec3 vertWUV)
{
  ivec3 onEdge = ivec3(equal(uvec3(0), vertWUV));
  return onEdge.x + onEdge.y + onEdge.z;
}

int baryOnEdgeIndex(uvec3 vertWUV)
{
  ivec3 bNZero = ivec3(equal(uvec3(0), vertWUV));
  // edge ordering 0={v0,v1}, 1={v1,v2}, 2={v2,v0}
  // return the edge opposite the vertWUV.* == 0 vertex
  // w=0 => edge 1, u=0 => edge 2, v=0 => edge 0
  return bNZero.x * 1 + bNZero.y * 2 + bNZero.z * 0;
}

// Returns the value for mix(edgeVert0, edgeVert1, value)
float baryOnEdgeInterp(int edgeIndex, vec3 baryCoord)
{
  // edge ordering 0={v0,v1}, 1={v1,v2}, 2={v2,v0}
  // Return the baryCoord of the edge's second vertex
  // edge 0 (v=0) => u, edge 1 (w=0) => v, edge 2 (u=0) => w
  return baryCoord[(edgeIndex + 1) % 3];
}

bool baryIsCorner(uvec3 vertWUV)
{
  return baryOnEdgeCount(vertWUV) == 2;
}

bool baryIsMidEdge(uvec3 vertWUV)
{
  return baryOnEdgeCount(vertWUV) == 1;
}

int baryCornerIndex(uvec3 vertWUV)
{
  ivec2 bNZero = ivec2(notEqual(uvec2(0), vertWUV.yz));
  return bNZero.x * 1 + bNZero.y * 2;
}

// Returns the subdivision level for a UV edge that is at least matching the frequency of heightmap texels.
// C++ reference: computeSubdivisionLevelsMatchingHeightmap()
uint edgeSubdivLevel(vec2 edge, vec2 heightmapSize)
{
  edge            = abs(edge * heightmapSize);
  float maxTexels = max(1.0, max(edge.x, edge.y));
  return uint(ceil(log2(maxTexels)));
}

#define MAX_MESH_SUBDIV 3

struct Job
{
  // Base mesh triangle index
  uint triangle;

  // Triangle subdiv level
  uint subdivLevel;

  // Sub-triangle index for when multiple mesh shaders tessellate one triangle
  uint subTriangle;

  // Level above triangle subdiv 3
  uint outerSubdivLevel;

  // Subdivision level along each edge for edge decimation
  uint edge0Subdiv;
  uint edge1Subdiv;
  uint edge2Subdiv;
};

// The task shader has produced some number of mesh shaders for MICRO_GROUP_SIZE
// base triangles. Some triangles may need multiple mesh shader instances
// (called sub-triangles) to be fully tessellated, i.e. when greater than subdiv
// level 3. We need to find the original base mesh's triangle index for this
// mesh shader and the sub-triangle index. The following uses subgroup threads
// to do so collaboratively.
Job computeJob(in uint taskMesh)
{
  Job  result;
  uint sgSubdivLevelBits = IN.subdivLevelBits[gl_SubgroupInvocationID];
  uint sgTargetSubdiv    = bitfieldExtract(sgSubdivLevelBits, 0, 4);
  uint sgEdge0Subdiv     = sgTargetSubdiv - bitfieldExtract(sgSubdivLevelBits, 4, 3);
  uint sgEdge1Subdiv     = sgTargetSubdiv - bitfieldExtract(sgSubdivLevelBits, 7, 3);
  uint sgEdge2Subdiv     = sgTargetSubdiv - bitfieldExtract(sgSubdivLevelBits, 10, 3);
  bool sgValid           = bitfieldExtract(sgSubdivLevelBits, 13, 1) != 0;

  uint sgOuterSubdiv = sgTargetSubdiv > MAX_MESH_SUBDIV ? sgTargetSubdiv - MAX_MESH_SUBDIV : 0;

  // Maximum mesh shader output is 64 triangles, which is subdiv 3. Find the
  // expected sub triangle count for each triangle.
  // E.g. subdiv [4, 3, 2, 4] gives mesh counts [4, 1, 1, 4]
  uint sgSubTriCount = sgValid ? (1 << (2 * sgOuterSubdiv)) : 0;

  // Prefix sum scan to find the starts and ends of each base triangle's group
  // of sub triangles.
  // E.g. (cont.) count ends will be [0, 4, 5, 6]
  uint sgSubTriStart = subgroupExclusiveAdd(sgSubTriCount);

  // Compare taskMesh against the start counts for each triangle. This mesh
  // shader's sub triangle index will be the largest thread index where the
  // condition still holds.
  // E.g. (cont.) if taskMesh is 7 sgTriangle will be [true, true, true, false]
  bool sgTriangle = taskMesh >= sgSubTriStart;

  // The base triangle index offset from IN.baseTriangle is the largest thread
  // index where sgTriangle is true.
  // E.g. (cont.) returns 2 because 7 > 6 but < 10 in sgSubTriEnd
  uint triangleOffset = subgroupBallotFindMSB(subgroupBallot(sgTriangle));
  result.triangle     = IN.baseTriangle + triangleOffset;

  // Given the triangle offset, everything could be re-computed, but one thread
  // will already have the needed values.
  result.subTriangle      = taskMesh - subgroupShuffle(sgSubTriStart, triangleOffset);
  result.subdivLevel      = subgroupShuffle(sgTargetSubdiv, triangleOffset);
  result.outerSubdivLevel = subgroupShuffle(sgOuterSubdiv, triangleOffset);
  result.edge0Subdiv      = subgroupShuffle(sgEdge0Subdiv, triangleOffset);
  result.edge1Subdiv      = subgroupShuffle(sgEdge1Subdiv, triangleOffset);
  result.edge2Subdiv      = subgroupShuffle(sgEdge2Subdiv, triangleOffset);
  return result;
}

uint isqrt(uint n)
{
  return uint(sqrt(float(n)));
}

// Computes sub-triangle positions given a linear triangle index i.
// - corner: the W-closest barycentric {w, u, v} vertex of a triangle where W is
//           the vertex opposite the w barycentric area. See the 'x' below.
// - negU:   if false, produce a triangle {corner, corner + WU, corner + WV}
//           if true, produce a triangle {corner, corner - WU, corner + WV}
//           See +U and -U triangles below. Admittedly counter-intuitive as
//           increasing the u coordinate moves towards U.
//
//               V
//              / \ 
//             / 3 \ 
//            /_____\ 
//           / \-U / \ 
//          /   \ /+U \ 
//         W ___ x ___ U
//
void uMajorTriangle(uint i, uint subdiv, out ivec3 corner, out bool negU)
{
  uint baryMax = 1 << subdiv;

  // Compute the row of triangles at a fixed w
  uint row = isqrt(i);
  i -= row * row;

  // Make quads from every vertex position
  negU = (i & 1) != 0;
  i >>= 1;

  // The remainder of i iterates between u and v
  uint w = baryMax - row;
  uint v = i;
  uint u = baryMax - w - v;
  corner = ivec3(w, u, v);
}

// Updates barycentric vertex coordinates to be watertight given per-edge subdiv
// levels by snapping coordinates and producing degenerate triangles.
ivec3 decimateEdges(ivec3 ibary, uint triangleSubdiv, uint edge0Subdiv, uint edge1Subdiv, uint edge2Subdiv)
{
  int edgeSegments = 1 << triangleSubdiv;
  if(triangleSubdiv != edge0Subdiv && ibary.z == 0)
  {
    int decimateBits = (1 << (triangleSubdiv - edge0Subdiv)) - 1;
    if(ibary.y > edgeSegments / 2)
      ibary.y += decimateBits;
    ibary.y &= ~decimateBits;
    ibary.x = edgeSegments - ibary.y;
  }
  if(triangleSubdiv != edge1Subdiv && ibary.x == 0)
  {
    int decimateBits = (1 << (triangleSubdiv - edge1Subdiv)) - 1;
    if(ibary.y > edgeSegments / 2)
      ibary.y += decimateBits;
    ibary.y &= ~decimateBits;
    ibary.z = edgeSegments - ibary.y;
  }
  if(triangleSubdiv != edge2Subdiv && ibary.y == 0)
  {
    int decimateBits = (1 << (triangleSubdiv - edge2Subdiv)) - 1;
    if(ibary.z > edgeSegments / 2)
      ibary.z += decimateBits;
    ibary.z &= ~decimateBits;
    ibary.x = edgeSegments - ibary.z;
  }
  return ibary;
}

// Need higher precision than HW texture filtering to avoid staircases at high
// tesesllation.
vec4 sampleHeight(sampler2D tex, vec2 coord)
{
  ivec2 size  = textureSize(tex, 0);
  vec2  texel = coord * size - 0.5;
  ivec2 i     = ivec2(floor(texel)) % size;
  if(i.x < 0)
    i.x += size.x;
  if(i.y < 0)
    i.y += size.y;
  vec2  f  = fract(texel);
  ivec2 aa = (i + ivec2(0, 0)) % size;
  ivec2 ba = (i + ivec2(1, 0)) % size;
  ivec2 ab = (i + ivec2(0, 1)) % size;
  ivec2 bb = (i + ivec2(1, 1)) % size;
  return mix(mix(texelFetch(tex, aa, 0), texelFetch(tex, ba, 0), f.x),
             mix(texelFetch(tex, ab, 0), texelFetch(tex, bb, 0), f.x), f.y);
}

void main()
{
  Job job = computeJob(gl_WorkGroupID.x);

  // Subdiv level of this mesh shader's geometry
  // meshSubdivLevel + job.outerSubdivLevel = job.subdivLevel
  uint meshSubdivLevel = min(job.subdivLevel, MAX_MESH_SUBDIV);

  // Compute subtriangle position in case of subdivision greater than 3
  ivec3 subTriangleCorner;
  bool  subTriangleNegU;
  uMajorTriangle(job.subTriangle, job.outerSubdivLevel, subTriangleCorner, subTriangleNegU);

  // Set offsets for the pre-cojputed meshlet geometry. The sub-triangle tiling
  // and edge decimation is done dynamically in this shader, so the only offsets
  // used is for meshSubdivLevel. See micromesh_decoder_subtri.glsl and
  // microdisp::MicroSplitParts.
  uint partOffset           = 0;
  uint edgeFlags            = 0;
  uint subTopo              = 0;
  uint splitPartsBase       = MICRO_MESHLET_LOD_PRIMS * meshSubdivLevel;
  uint splitPartsPrimOffset = splitPartsBase + subTopo * MICRO_PART_MAX_PRIMITIVES + edgeFlags * MICRO_MESHLET_PRIMS;
  uint splitPartsVertOffset = partOffset * MICRO_PART_VERTICES_STRIDE;

  // Instances
  InstanceInfos instances = InstanceInfos(sceneDesc.instInfoAddress);
  InstanceInfo  instinfo  = instances.i[pc.instanceID];

  // Instance geometry
  DeviceMeshInfos pInfos = DeviceMeshInfos(sceneDesc.deviceMeshInfoAddress);
  DeviceMeshInfo  pinfo  = pInfos.i[pc.primMeshID];

  Indices indices       = Indices(pinfo.triangleVertexIndexBuffer);
  uvec3   triangleIndex = indices.i[job.triangle];

  // Base triangle positions before subdivision and interpolation.
  Vertices   vertices = Vertices(pinfo.vertexPositionNormalBuffer);
  const vec3 v0       = vertices.v[triangleIndex.x].xyz;
  const vec3 v1       = vertices.v[triangleIndex.y].xyz;
  const vec3 v2       = vertices.v[triangleIndex.z].xyz;

  // Direction vectors are mesh normals. Note that if configured the baker can
  // generate smooth direction vectors which can be different.
  const vec3 d0 = oct32_to_vec(floatBitsToInt(vertices.v[triangleIndex.x].w));
  const vec3 d1 = oct32_to_vec(floatBitsToInt(vertices.v[triangleIndex.y].w));
  const vec3 d2 = oct32_to_vec(floatBitsToInt(vertices.v[triangleIndex.z].w));

  // Texture coordinates for sampling the heightmap
  vec2 tex0;
  vec2 tex1;
  vec2 tex2;
  if(pinfo.vertexTexcoordBuffer != 0)
  {
    tex0 = TexCoords(pinfo.vertexTexcoordBuffer).v[triangleIndex.x];
    tex1 = TexCoords(pinfo.vertexTexcoordBuffer).v[triangleIndex.y];
    tex2 = TexCoords(pinfo.vertexTexcoordBuffer).v[triangleIndex.z];
  }

  // Material of the object
  Materials         mats = Materials(sceneDesc.materialAddress);
  GltfShadeMaterial mat  = mats.m[pc.materialID];

  // subdiv 3 is the maximum per mesh shader. Use a constant to allow unrolling.
  const uint maxVertices = subdiv_getNumVerts(MAX_MESH_SUBDIV);
  uint       numVertices = subdiv_getNumVerts(meshSubdivLevel);
  for(uint vertIter = gl_SubgroupInvocationID; vertIter < numVertices && vertIter < maxVertices; vertIter += MICRO_GROUP_SIZE)
  {
    // Load meshlet vertices. See smicrodec_subgroupGetVertex().
    MicromeshSTriVertex localVtx = MicroVertices(sceneDesc.splitPartsVerticesAddress).v[splitPartsVertOffset + vertIter];
    ivec2 subTriUV = ivec2(bitfieldExtract(localVtx.packed, MICRO_STRI_VTX_U_SHIFT, MICRO_STRI_VTX_UV_WIDTH),
                           bitfieldExtract(localVtx.packed, MICRO_STRI_VTX_V_SHIFT, MICRO_STRI_VTX_UV_WIDTH));

    // MicroSplitParts generates integer coordinates relative to subdiv 3 even
    // for lower levels. This is incompatible with decimateEdges() so a division
    // is needed.
    subTriUV >>= MAX_MESH_SUBDIV - meshSubdivLevel;

    // Flip the sub triangle around U (the WV vertex direction). See uMajorTriangle.
    if(subTriangleNegU)
    {
      subTriUV.y += subTriUV.x;  // "-u" shoud go towards +v, not +w
      subTriUV.x = -subTriUV.x;
    }

    // Add the sub triangle offset (based on job.subTriangle index)
    subTriUV += subTriangleCorner.yz << MAX_MESH_SUBDIV;

    // Convert integer to normalized floating point barycentric coordinates
    int   baryMax = 1 << job.subdivLevel;
    ivec3 ibary   = ivec3(baryMax - subTriUV.x - subTriUV.y, subTriUV);
    ibary         = decimateEdges(ibary, job.subdivLevel, job.edge0Subdiv, job.edge1Subdiv, job.edge2Subdiv);
    vec3 bary     = vec3(ibary) / float(baryMax);

    vec3 dispDirection = mixBary(d0, d1, d2, bary);
    vec3 pos           = mixBary(v0, v1, v2, bary);

    float dispDistance = 0.0;
    if(mat.khrDisplacementTexture > -1 && pinfo.vertexTexcoordBuffer != 0)
    {
      vec2 tex = mixBary(tex0, tex1, tex2, bary);

      // Snap to displacements at watertight vertices. See WatertightIndices.
      WatertightIndices wtIndices = TriWTIndices(pinfo.triangleWatertightIndicesBuffer).v[job.triangle];
      if(baryIsCorner(ibary))
      {
        int cornerIdx = baryCornerIndex(ibary);
        if(wtIndices.watertightCornerVertex[cornerIdx] != WATERTIGHT_INDICES_INVALID_VERTEX)
        {
          dispDirection = oct32_to_vec(floatBitsToInt(vertices.v[wtIndices.watertightCornerVertex[cornerIdx]].w));
          tex           = TexCoords(pinfo.vertexTexcoordBuffer).v[wtIndices.watertightCornerVertex[cornerIdx]];
        }
      }

      dispDistance = sampleHeight(texturesMap[nonuniformEXT(mat.khrDisplacementTexture)], tex).x;

      // Take the average of displacements along seams. See WatertightIndices.
      if(baryIsMidEdge(ibary))
      {
        int   edgeIdx   = baryOnEdgeIndex(ibary);
        ivec2 edgeVerts = wtIndices.seamEdge[edgeIdx];
        if(edgeVerts.x != WATERTIGHT_INDICES_INVALID_VERTEX && edgeVerts.y != WATERTIGHT_INDICES_INVALID_VERTEX)
        {
          // TODO: interpolation inputs should be sorted for stability
          vec3 edgeDir0 = oct32_to_vec(floatBitsToInt(vertices.v[edgeVerts.x].w));
          vec3 edgeDir1 = oct32_to_vec(floatBitsToInt(vertices.v[edgeVerts.y].w));
          vec3 edgeDir  = mix(edgeDir0, edgeDir1, baryOnEdgeInterp(edgeIdx, bary));
          dispDirection = mix(dispDirection, edgeDir, 0.5);
          vec2 edgeTex0 = TexCoords(pinfo.vertexTexcoordBuffer).v[edgeVerts.x];
          vec2 edgeTex1 = TexCoords(pinfo.vertexTexcoordBuffer).v[edgeVerts.y];
          vec2 edgeTex  = mix(edgeTex0, edgeTex1, baryOnEdgeInterp(edgeIdx, bary));
          dispDistance = mix(dispDistance, sampleHeight(texturesMap[nonuniformEXT(mat.khrDisplacementTexture)], edgeTex).x, 0.5);
        }
      }

      dispDistance = dispDistance * mat.khrDisplacementFactor + mat.khrDisplacementOffset;
      dispDistance = dispDistance * frameInfo.heightmapScale + frameInfo.heightmapOffset;

      // Heightmaps are typically displaced by normalized interpolated normals
      dispDirection = normalize(dispDirection);
    }

    pos                                     = pos + dispDirection * dispDistance;
    vec4 wPos                               = instinfo.objectToWorld * vec4(pos, 1);
    gl_MeshVerticesNV[vertIter].gl_Position = frameInfo.proj * frameInfo.view * wPos;

    OUT[vertIter].pos  = wPos.xyz;
    OUT[vertIter].bary = bary;
  }

  // subdiv 3 is the maximum per mesh shader. Use a constant to allow unrolling.
  const uint maxTriangles = subdiv_getNumTriangles(MAX_MESH_SUBDIV);
  uint       numTriangles = subdiv_getNumTriangles(meshSubdivLevel);
  for(uint32_t primIter = gl_SubgroupInvocationID; primIter < numTriangles && primIter < maxTriangles; primIter += MICRO_GROUP_SIZE)
  {
    // Load meshlet indices. See smicrodec_getTriangle().
    u8vec4 indices = MicroIndices(sceneDesc.splitPartsIndicesAddress).v[splitPartsPrimOffset + primIter];
    gl_PrimitiveIndicesNV[primIter * 3 + 0]      = indices.x;
    gl_PrimitiveIndicesNV[primIter * 3 + 1]      = subTriangleNegU ? indices.z : indices.y;
    gl_PrimitiveIndicesNV[primIter * 3 + 2]      = subTriangleNegU ? indices.y : indices.z;
    gl_MeshPrimitivesNV[primIter].gl_PrimitiveID = int(job.triangle);

    // Visualization data for the fragment shader
    if(CONST_SHADE_MODE == eRenderShading_anisotropy)
    {
      vec3  a                       = OUT[indices.x].pos;
      vec3  b                       = OUT[indices.y].pos;
      vec3  c                       = OUT[indices.z].pos;
      float anisotropy              = anisotropyMetric(a, b, c);
      uint  value                   = floatBitsToInt(anisotropy);
      POUT[primIter].shadeModeValue = value;
    }
    else if(CONST_SHADE_MODE == eRenderShading_baseTriangleIndex)
    {
      uint id                       = primIter + job.subTriangle * numTriangles;
      POUT[primIter].shadeModeValue = id | (job.triangle << 10);
    }
    else if(CONST_SHADE_MODE == eRenderShading_subdivLevel)
    {
      // Send the triangle subdiv level and insert some fake edge flags in the
      // last few bits to visualize decimation.
      uint bits                     = job.subdivLevel;
      bits                          = bitfieldInsert(bits, job.edge0Subdiv == job.subdivLevel ? 0 : 1, 29, 1);
      bits                          = bitfieldInsert(bits, job.edge1Subdiv == job.subdivLevel ? 0 : 1, 30, 1);
      bits                          = bitfieldInsert(bits, job.edge2Subdiv == job.subdivLevel ? 0 : 1, 31, 1);
      POUT[primIter].shadeModeValue = bits;
    }
  }

  if(gl_SubgroupInvocationID == 0)
  {
    gl_PrimitiveCountNV = numTriangles;
  }
}
