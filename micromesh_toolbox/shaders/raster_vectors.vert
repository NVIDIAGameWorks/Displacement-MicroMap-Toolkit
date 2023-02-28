/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_shader_explicit_arithmetic_types : enable


#include "device_host.h"
#include "dh_bindings.h"
#include "octant_encoding.h"
#include "dh_scn_desc.h"

// clang-format off
layout(set = 0, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 0, binding = eSceneDescTools) readonly buffer SceneDesc_ { SceneDescription sceneDesc; };
layout(buffer_reference, scalar) readonly buffer Vertices { vec4 v[]; };
layout(buffer_reference, scalar) readonly buffer DirectionBounds { vec2 v[]; };
layout(buffer_reference, scalar) readonly buffer Directions { f16vec4 v[]; };
layout(buffer_reference, scalar) readonly buffer Indices { uvec3 i[]; };
layout(buffer_reference, scalar) readonly buffer DeviceMeshInfos { DeviceMeshInfo i[]; };
layout(buffer_reference, scalar) readonly buffer InstancesInfo { InstanceInfo i[]; };

layout(push_constant) uniform RasterPushConstant_ { PushConstant pc; };
// clang-format on


layout(location = 0) in vec4 i_pos;

layout(constant_id = 1) const int CONST_DEBUG_MODE = 0;

void main()
{
  // Shell drawing mode
  uint realVertexIndex;

  // Directions from base to tip
  realVertexIndex = gl_VertexIndex / 3;
  float tip       = (((gl_VertexIndex % 3) == 1) ? 1.0 : 0.0);

  // Instances
  InstancesInfo instances = InstancesInfo(sceneDesc.instInfoAddress);
  InstanceInfo  instinfo  = instances.i[pc.instanceID];

  // Primitive meshes
  DeviceMeshInfos pInfos = DeviceMeshInfos(sceneDesc.deviceMeshInfoAddress);
  DeviceMeshInfo  pinfo  = pInfos.i[pc.primMeshID];

  // Getting Position, Normal and default values
  Vertices vertices  = Vertices(pinfo.vertexPositionNormalBuffer);
  vec4     posNorm   = vertices.v[realVertexIndex];
  vec3     position  = posNorm.xyz;
  vec3     direction = oct32_to_vec(floatBitsToUint(posNorm.w));

  if(CONST_DEBUG_MODE == eDbgMethod_normal)
  {
    direction = oct32_to_vec(floatBitsToUint(posNorm.w));
  }
  else if(pinfo.vertexDirectionsBuffer != 0)
  {
    Directions directions = Directions(pinfo.vertexDirectionsBuffer);
    direction             = vec3(directions.v[realVertexIndex]);
  }


  vec3 f_pos = position.xyz + direction.xyz * tip * frameInfo.vectorLength;

  vec4 o_pos  = instinfo.objectToWorld * vec4(f_pos, 1.0);
  gl_Position = frameInfo.proj * frameInfo.view * o_pos;
}
