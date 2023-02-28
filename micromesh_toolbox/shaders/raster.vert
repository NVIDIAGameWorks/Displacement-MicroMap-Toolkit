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

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "device_host.h"
#include "dh_bindings.h"
#include "octant_encoding.h"
#include "dh_scn_desc.h"

// clang-format off
layout(buffer_reference, scalar) buffer  InstancesInfo { InstanceInfo i[]; };

layout(set = 0, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 0, binding = eSceneDescTools) readonly buffer SceneDesc_ { SceneDescription sceneDesc; } ;
// clang-format on

layout(push_constant) uniform RasterPushConstant_
{
  PushConstant pc;
};

layout(location = 0) in vec4 i_pos;


layout(location = 0) out Interpolants
{
  vec3 pos;
}
OUT;

out gl_PerVertex
{
  vec4 gl_Position;
};


void main()
{
  InstancesInfo instances = InstancesInfo(sceneDesc.instInfoAddress);
  InstanceInfo  instinfo  = instances.i[pc.instanceID];

  vec3 origin   = vec3(frameInfo.viewInv * vec4(0, 0, 0, 1));
  vec3 position = i_pos.xyz;

  OUT.pos = vec3(instinfo.objectToWorld * vec4(position, 1.0));

  gl_Position = frameInfo.proj * frameInfo.view * vec4(OUT.pos, 1.0);
}
