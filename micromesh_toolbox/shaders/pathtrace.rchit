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
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_NV_displacement_micromap : require

#include "device_host.h"
#include "dh_bindings.h"
#include "payload.glsl"
#include "dh_scn_desc.h"

hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT HitPayload payload;

layout(buffer_reference, scalar) readonly buffer Vertices { vec4 v[]; };
layout(buffer_reference, scalar) readonly buffer TexCoords { vec2 v[]; };
layout(buffer_reference, scalar) readonly buffer Tangents { vec2 v[]; };
layout(buffer_reference, scalar) readonly buffer Indices { uvec3 i[]; };
layout(buffer_reference, scalar) readonly buffer DeviceMeshInfos { DeviceMeshInfo i[]; };

layout(set = 1, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; };
// clang-format on

#include "nvvkhl/shaders/func.glsl"
#include "get_hit.glsl"

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void main()
{
  // Retrieve the Primitive mesh buffer information
  DeviceMeshInfos pInfo_ = DeviceMeshInfos(sceneDesc.deviceMeshInfoAddress);
  DeviceMeshInfo  pinfo  = pInfo_.i[gl_InstanceCustomIndexEXT];

  payload.hitT = gl_HitTEXT;
  //  payload.instanceIndex = gl_InstanceCustomIndexEXT;
  payload.instanceIndex = gl_InstanceID;

  vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
  bool frontFacing  = gl_HitKindEXT == gl_HitKindFrontFacingTriangleEXT || gl_HitKindEXT == gl_HitKindFrontFacingMicroTriangleNV;
  payload.hit       = getHitState(pinfo, barycentrics, gl_PrimitiveID, gl_ObjectToWorldEXT, gl_WorldToObjectEXT,
                                  gl_WorldRayDirectionEXT, frontFacing);
  payload.micromesh = false;

  // Replace the interpolated base triangle position with the ray hit position for microtriangles
  if(gl_HitKindEXT == gl_HitKindFrontFacingMicroTriangleNV || gl_HitKindEXT == gl_HitKindBackFacingMicroTriangleNV)
  {
    payload.micromesh = true;
    payload.hit.pos   = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    payload.hit.geonrm = normalize(cross(gl_HitMicroTriangleVertexPositionsNV[2] - gl_HitMicroTriangleVertexPositionsNV[0],
                                         gl_HitMicroTriangleVertexPositionsNV[2] - gl_HitMicroTriangleVertexPositionsNV[1]));
    if(!frontFacing)
    {
      payload.hit.geonrm = -payload.hit.geonrm;
    }
  }
}
