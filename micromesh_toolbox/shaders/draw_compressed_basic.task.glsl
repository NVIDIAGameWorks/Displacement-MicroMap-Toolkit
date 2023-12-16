/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

/*
 * NOTE: This file's content is based on older versions of code in
 *       https://github.com/nvpro-samples/vk_displacement_micromaps. Please
 *       follow micromesh rasterization examples there instead.
 */

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

#include "device_host.h"
#include "dh_bindings.h"
#include "dh_scn_desc.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "common.h"
//#include "common_mesh.h"
#include "common_barymap.h"
#include "common_micromesh_compressed.h"

//////////////////////////////////////////////////////////////////////////

layout(local_size_x = MICRO_GROUP_SIZE) in;

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
layout(buffer_reference, scalar) readonly buffer Vertices { vec4 v[]; };
layout(buffer_reference, scalar) readonly buffer TexCoords { vec2 v[]; };
layout(buffer_reference, scalar) readonly buffer Tangents { vec2 v[]; };
layout(buffer_reference, scalar) readonly buffer Indices { uvec3 i[]; };
layout(buffer_reference, scalar) readonly buffer DeviceMeshInfos { DeviceMeshInfo i[]; };
layout(buffer_reference, scalar) readonly buffer DeviceBaryInfos { DeviceBaryInfo i[]; };
layout(buffer_reference, scalar) readonly buffer MicromeshDataPtr { MicromeshData microdata; };

layout(set = 0, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 0, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; } ;
layout(set = 0, binding = eTextures) uniform sampler2D[] texturesMap;

layout(set = 1, binding = 0) uniform sampler2D   u_GGXLUT; // lookup table
layout(set = 1, binding = 1) uniform samplerCube u_LambertianEnvSampler; // 
layout(set = 1, binding = 2) uniform samplerCube u_GGXEnvSampler;  //

layout(set = 2, binding = eSkyParam) uniform SkyInfo_ { ProceduralSkyShaderParameters skyInfo; };
// clang-format on

//////////////////////////////////////////////////////////////////////////
// OUTPUT

taskNV out Task
{
  uint     baseID;
  uint16_t prefix[MICRO_TRI_PER_TASK];
}
OUT;


//////////////////////////////////////////////////////////////////////////

#include "micromesh_utils.glsl"

//////////////////////////////////////////////////////////////////////////

// This shader does not do any packing of micromeshes with low subdivision
// into a single mesh-shader invocation. This yields less performance as we
// unterutilize the hardware this way (an entire mesh workgroup may generate
// only a single triangle).
// Look at the draw_micromesh_lod shaders which use a more complex setup
// that does this.

void main()
{
  uint baseID = gl_WorkGroupID.x * MICRO_TRI_PER_TASK;
  uint laneID = gl_SubgroupInvocationID;

  OUT.baseID = baseID;

  uint microID = baseID + laneID;

  DeviceBaryInfos  baryInfos    = DeviceBaryInfos(sceneDesc.deviceBaryInfoAddress);
  DeviceBaryInfo   baryInfo     = baryInfos.i[pc.baryInfoID];
  MicromeshDataPtr microdataPtr = MicromeshDataPtr(baryInfo.rasterMeshDataBindingBuffer);
  microdata                     = microdataPtr.microdata;

#if MICRO_USE_BASETRIANGLES
  MicromeshBaseTri microBaseTri = microdata.basetriangles.d[min(microID, pc.microMax)];
  uint             microSubdiv  = micromesh_getBaseSubdiv(microBaseTri);
#else
  MicromeshSubTri microSubTri = microdata.subtriangles.d[min(microID, pc.microMax)];
  uint            microSubdiv = micromesh_getSubdiv(microSubTri);
#endif

  uint partMicroMeshlets = subdiv_getNumMeshlets(microSubdiv);
  uint microMax          = pc.microMax;

  bool valid = true;

  // debugging
  //valid = microID == 0;
  //partMicroMeshlets = 1;

  uint meshletCount = microID <= microMax && valid ? partMicroMeshlets : 0;

  uint prefix        = subgroupExclusiveAdd(meshletCount);
  OUT.prefix[laneID] = uint16_t(prefix);

  if(laneID == MICRO_TRI_PER_TASK - 1)
  {
    gl_TaskCountNV = prefix + meshletCount;
#if USE_STATS
    atomicAdd(stats.meshlets, prefix + meshletCount);
#endif
  }
}