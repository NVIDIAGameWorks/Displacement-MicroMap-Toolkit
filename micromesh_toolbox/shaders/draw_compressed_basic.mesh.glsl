/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// PLEASE READ:
// The rasterization of micromeshes, especially compressed is a bit of
// a more complex topic on its own. Therefore there will be a future
// dedicated sample that goes into details how it works
// and showcases more features, such as dynamic level-of-detail.
// We recommend to wait for this, rather than attempt to
// embed the code from the toolkit. The future sample will also
// provide more performant solutions and address compute-based
// rasterization as well.

#version 460

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_control_flow_attributes : require

#extension GL_NV_mesh_shader : enable

#extension GL_EXT_buffer_reference2 : enable
#extension GL_EXT_scalar_block_layout : enable
//#extension GL_EXT_shader_8bit_storage : enable
//#extension GL_EXT_shader_16bit_storage : enable
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

//#undef  MICRO_SUPPORTED_FORMAT_BITS
//#define MICRO_SUPPORTED_FORMAT_BITS   ((1<<MICRO_FORMAT_64T_512B) | (1<<MICRO_FORMAT_256T_1024B))

#include "device_host.h"
#include "dh_bindings.h"
#include "dh_scn_desc.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "common.h"
//#include "common_mesh.h"
#include "common_barymap.h"
#include "common_micromesh_compressed.h"
#include "raster_anisotropy.glsl"

//////////////////////////////////////////////////////////////////////////

layout(local_size_x = MICRO_GROUP_SIZE) in;
layout(max_primitives = MICRO_MESHLET_PRIMITIVES, max_vertices = MICRO_MESHLET_VERTICES, triangles) out;

layout(constant_id = 0) const int CONST_SHADE_MODE = 0;


////////////////////////////////////////////////////////////////
// BINDINGS

//layout(scalar, binding = DRAWCOMPRESSED_UBO_COMPRESSED) uniform microBuffer {
//  MicromeshData microdata;
//};

// set in main()
MicromeshData microdata;

layout(push_constant) uniform RasterPushConstant_
{
  PushConstant pc;
};

// clang-format off
layout(buffer_reference, scalar) readonly buffer Vertices         { vec4 v[]; };
layout(buffer_reference, scalar) readonly buffer TexCoords        { vec2 v[]; };
layout(buffer_reference, scalar) readonly buffer Tangents         { vec2 v[]; };
layout(buffer_reference, scalar) readonly buffer Indices          { uvec3 i[]; };
layout(buffer_reference, scalar) readonly buffer Directions       { f16vec4 v[]; };
layout(buffer_reference, scalar) readonly buffer DirectionBounds  { vec2 v[]; };
layout(buffer_reference, scalar) readonly buffer DeviceMeshInfos  { DeviceMeshInfo i[]; };
layout(buffer_reference, scalar) readonly buffer DeviceBaryInfos  { DeviceBaryInfo i[]; };
layout(buffer_reference, scalar) readonly buffer MicromeshDataPtr { MicromeshData microdata; };
layout(buffer_reference, scalar) readonly buffer Materials        { GltfShadeMaterial m[]; };
layout(buffer_reference, scalar) readonly buffer InstanceInfos    { InstanceInfo i[]; };

layout(set = 0, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 0, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; } ;
layout(set = 0, binding = eTextures) uniform sampler2D[] texturesMap;

layout(set = 1, binding = 0) uniform sampler2D   u_GGXLUT; // lookup table
layout(set = 1, binding = 1) uniform samplerCube u_LambertianEnvSampler; // 
layout(set = 1, binding = 2) uniform samplerCube u_GGXEnvSampler;  //

layout(set = 2, binding = eSkyParam) uniform SkyInfo_ { ProceduralSkyShaderParameters skyInfo; };
// clang-format on

//////////////////////////////////////////////////////////////////////////
// INPUT

taskNV in Task
{
  uint     wgroupID;
  uint16_t prefix[MICRO_TRI_PER_TASK];
}
IN;

//////////////////////////////////////////////////////////////////////////
// OUTPUT

#if SURFACEVIS == SURFACEVIS_SHADING
layout(location = 0) out Interpolants
{
  vec3 pos;
  vec3 bary;
}
OUT[];
#endif

layout(location = 3) perprimitiveNV out uint shadeModeValue[];


//////////////////////////////////////////////////////////////////////////

#include "micromesh_utils.glsl"
#include "micromesh_decoder.glsl"

#if USE_PRIMITIVE_CULLING
#include "draw_culling.glsl"
#endif

//////////////////////////////////////////////////////////////////////////

// This shader does not do any packing of micromeshes with low subdivision
// into a single mesh-shader invocation. This yields less performance as we
// unterutilize the hardware this way (an entire mesh workgroup may generate
// only a single triangle).
// Look at the draw_micromesh_lod shaders which use a more complex setup
// that does this.

void main()
{
  //////////////////////////////////////
  // Decoder Configuration Phase

  // task emits N meshlets for multiple micromeshes
  // find original microSubTri we are from
  uint  wgroupID = gl_WorkGroupID.x;
  uint  laneID   = gl_SubgroupInvocationID;
  uint  prefix   = IN.prefix[laneID];
  uvec4 voteID   = subgroupBallot(wgroupID >= prefix);
  uint  subID    = subgroupBallotFindMSB(voteID);
  uint  microID  = IN.wgroupID + subID;
  uint  partID   = wgroupID - subgroupShuffle(prefix, subID);

  DeviceBaryInfos  baryInfos    = DeviceBaryInfos(sceneDesc.deviceBaryInfoAddress);
  DeviceBaryInfo   baryInfo     = baryInfos.i[pc.baryInfoID];
  MicromeshDataPtr microdataPtr = MicromeshDataPtr(baryInfo.rasterMeshDataBindingBuffer);
  microdata                     = microdataPtr.microdata;

#if 0
  // debugging
  if (partID != scene.dbgUint) {
    gl_PrimitiveCountNV = 0;
    return;
  }
#endif

#if MICRO_USE_BASETRIANGLES
  MicromeshBaseTri microTri = microdata.basetriangles.d[microID];
#if MICRO_DECODER == MICRO_DECODER_MICROTRI_THREAD || MICRO_DECODER == MICRO_DECODER_BASETRI_MIP_SHUFFLE
  uint subTriangleIndex = 0;
  uint microSubdiv      = micromesh_getBaseSubdiv(microTri);
#else
  uint microSubdiv      = micromesh_getSubdiv(microTri);
  uint numPartMeshlets  = subdiv_getNumMeshlets(micromesh_getSubdiv(microTri));
  uint subTriangleIndex = partID / numPartMeshlets;
  // adjust partID to be meshlet part within sub-triangle
  partID &= numPartMeshlets - 1;
#endif
#else
  // Load microSubTri
  MicromeshSubTri microTri         = microdata.subtriangles.d[microID];
  uint            subTriangleIndex = 0;
  uint            microSubdiv      = micromesh_getSubdiv(microTri);
#endif

  MicroDecoderConfig cfg;
  cfg.microID          = microID;
  cfg.partID           = partID;
  cfg.subTriangleIndex = subTriangleIndex;
  cfg.targetSubdiv     = microSubdiv;
  cfg.partSubdiv       = min(3, microSubdiv);
  cfg.packThreadID     = laneID;
  cfg.packThreads      = SUBGROUP_SIZE;
  cfg.packID           = 0;
  cfg.valid            = true;

  //////////////////////////////////////
  // Initial Decoding Phase

  SubgroupMicromeshDecoder sdec;
  smicrodec_subgroupInit(sdec, cfg, microTri, 0, 0, 0);

  //////////////////////////////////////
  // Mesh Preparation Phase
  DeviceMeshInfos pInfo_ = DeviceMeshInfos(sceneDesc.deviceMeshInfoAddress);
  DeviceMeshInfo  pinfo  = pInfo_.i[pc.primMeshID];

  InstanceInfos instInfos = InstanceInfos(sceneDesc.instInfoAddress);
  InstanceInfo  instInfo  = instInfos.i[pc.instanceID];

  Vertices        positions       = Vertices(pinfo.vertexPositionNormalBuffer);
  Indices         indices         = Indices(pinfo.triangleVertexIndexBuffer);
  TexCoords       texcoords       = TexCoords(pinfo.vertexTexcoordBuffer);
  Directions      directions      = Directions(pinfo.vertexDirectionsBuffer);
  DirectionBounds directionBounds = DirectionBounds(pinfo.vertexDirectionBoundsBuffer);

  uint  triLocal   = smicrodec_getMeshTriangle(sdec);
  uint  tri        = triLocal;
  uvec3 triIndices = indices.i[tri];

  mat4 worldMatrix   = instInfo.objectToWorld;
  mat4 worldMatrixIT = instInfo.worldToObject;

  // Generate vertices
  vec3 v0 = positions.v[triIndices.x].xyz;
  vec3 v1 = positions.v[triIndices.y].xyz;
  vec3 v2 = positions.v[triIndices.z].xyz;

  f16vec3 d0 = directions.v[triIndices.x].xyz;
  f16vec3 d1 = directions.v[triIndices.y].xyz;
  f16vec3 d2 = directions.v[triIndices.z].xyz;

#if USE_DIRECTION_BOUNDS
  boundsVec2 bounds0 = directionBounds.v[triIndices.x];
  boundsVec2 bounds1 = directionBounds.v[triIndices.y];
  boundsVec2 bounds2 = directionBounds.v[triIndices.z];

  v0 = v0 + d0 * bounds0.x;
  v1 = v1 + d1 * bounds1.x;
  v2 = v2 + d2 * bounds2.x;

  d0 = d0 * float16_t(bounds0.y);
  d1 = d1 * float16_t(bounds1.y);
  d2 = d2 * float16_t(bounds2.y);
#endif

  //////////////////////////////////////
  // Vertex Iteration Phase

#if USE_MICROVERTEX_NORMALS
  uint firstValue = microdata.attrTriangleOffsets.d[triLocal];
#endif

  //////////////////////////////////////
  // Primitive Iteration Phase

  float valueRange = 0;

  if(CONST_SHADE_MODE == eRenderShading_minMax)
  {
    // FIXME tri index can actually be wrong here
    // okay for the baking app
    valueRange = getValueRange(float(microdata.basetriangleMinMaxs.d[tri * 2 + 0]) / float(0x7FF),
                               float(microdata.basetriangleMinMaxs.d[tri * 2 + 1]) / float(0x7FF));
  }

  uint baseSubdiv   = smicrodec_getBaseSubdiv(sdec);
  uint numTriangles = smicrodec_getNumTriangles(sdec);
  uint formatIdx    = smicrodec_getFormatIdx(sdec);

  for(uint vertIter = 0; vertIter < smicrodec_getIterationCount(); vertIter++)
  {
    MicroDecodedVertex decoded = smicrodec_subgroupGetVertex(sdec, vertIter);
    uint               vertOut = decoded.outIndex;

    // safe to early out post shuffle
    // This thread may not be valid, but a valid one before it may need to acces its data for shuffle in
    // smicrodec_subgroupGetLocalVertex

    if(!decoded.valid)
      continue;

    float dispDistance = micromesh_getFloatDisplacement(decoded.displacement, f16vec2(pc.microScaleBias));

    // for tweakable scaling (not compatible with RT)
    //dispDistance = dispDistance * scene.disp_scale + scene.disp_bias;

    // Compute interpolation
    vec3 dispDirection = getInterpolated(d0, d1, d2, decoded.bary);

    vec3 pos = getInterpolated(v0, v1, v2, decoded.bary) + vec3(dispDirection) * dispDistance;
    pos      = vec3(worldMatrix * vec4(pos, 1.0));

    gl_MeshVerticesNV[vertOut].gl_Position = frameInfo.proj * frameInfo.view * vec4(pos, 1);


#if SURFACEVIS == SURFACEVIS_SHADING
    OUT[vertOut].pos  = pos;
    OUT[vertOut].bary = decoded.bary;


#if USE_MICROVERTEX_NORMALS
    uint valueIdx = umajorUV_toLinear(subdiv_getNumVertsPerEdge(baseSubdiv), decoded.uv);
#if !SHADING_UMAJOR
    valueIdx = microdata.umajor2bmap[baseSubdiv].d[valueIdx];
#endif
    OUTvtx[vertOut].vidx = firstValue + valueIdx;
#endif
#endif
  }


  // iterate primitives
#if USE_PRIMITIVE_CULLING
  uint numTrianglesOut = 0;
#else
  uint            numTrianglesOut  = numTriangles;
#endif
  for(uint primIter = 0; primIter < smicrodec_getIterationCount(); primIter++)
  {
    MicroDecodedTriangle decoded = smicrodec_getTriangle(sdec, primIter);
    bool                 visible = decoded.valid;
#if USE_PRIMITIVE_CULLING
    if(visible)
    {
      RasterVertex a = getRasterVertex(gl_MeshVerticesNV[decoded.indices.x].gl_Position);
      RasterVertex b = getRasterVertex(gl_MeshVerticesNV[decoded.indices.y].gl_Position);
      RasterVertex c = getRasterVertex(gl_MeshVerticesNV[decoded.indices.z].gl_Position);

      visible = testTriangle(a, b, c, 1.0);
    }
    uvec4 voteVis = subgroupBallot(visible);
    uint  primOut = numTrianglesOut + subgroupBallotExclusiveBitCount(voteVis);

    numTrianglesOut += subgroupBallotBitCount(voteVis);
#else
    uint primOut = decoded.outIndex;
#endif

    if(visible)
    {
      gl_PrimitiveIndicesNV[primOut * 3 + 0] = decoded.indices.x;
      gl_PrimitiveIndicesNV[primOut * 3 + 1] = decoded.indices.y;
      gl_PrimitiveIndicesNV[primOut * 3 + 2] = decoded.indices.z;

      // Set the primitive ID for the fragment shader to load and interpolate
      // based on OUT[].bary
      gl_MeshPrimitivesNV[primOut].gl_PrimitiveID = int(tri);

      if(CONST_SHADE_MODE == eRenderShading_anisotropy)
      {
        vec3  a                 = OUT[decoded.indices.x].pos;
        vec3  b                 = OUT[decoded.indices.y].pos;
        vec3  c                 = OUT[decoded.indices.z].pos;
        float anisotropy        = anisotropyMetric(a, b, c);
        uint  value             = floatBitsToInt(anisotropy);
        shadeModeValue[primOut] = value;
      }
      else if(CONST_SHADE_MODE == eRenderShading_baseTriangleIndex)
      {
        uint id                 = decoded.localIndex + cfg.partID * numTriangles;
        shadeModeValue[primOut] = int(id | (tri << 10));
      }
      else if(CONST_SHADE_MODE == eRenderShading_subdivLevel)
      {
        shadeModeValue[primOut] = int(baseSubdiv);
      }
      else if(CONST_SHADE_MODE == eRenderShading_compressionFormat)
      {
        shadeModeValue[primOut] = formatIdx;
      }
      else if(CONST_SHADE_MODE == eRenderShading_minMax)
      {
        shadeModeValue[primOut] = floatBitsToInt(valueRange);
      }
    }
  }

  if(laneID == 0)
  {
    gl_PrimitiveCountNV = numTrianglesOut;
#if USE_STATS
    atomicAdd(stats.triangles, numTrianglesOut);
#endif
  }
}
