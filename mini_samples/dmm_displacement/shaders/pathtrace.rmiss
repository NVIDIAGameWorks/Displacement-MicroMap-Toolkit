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
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "device_host.h"
#include "payload.h"
#include "dh_bindings.h"
#include "nvvkhl/shaders/dh_sky.h"

layout(location = 0) rayPayloadInEXT HitPayload payload;

layout(set = 0, binding = BRtSkyParam) uniform SkyInfo_
{
  ProceduralSkyShaderParameters skyInfo;
};


void main()
{
  vec3 sky_color = proceduralSky(skyInfo, gl_WorldRayDirectionEXT, 0);
  payload.color += sky_color * payload.weight;
  payload.depth = MISS_DEPTH;  // Stop
}
