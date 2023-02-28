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

#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_debug_printf : enable
#extension GL_NV_fragment_shader_barycentric : enable

// Outgoing
layout(location = 0) out vec4 outColor;

#include "device_host.h"
#include "dh_bindings.h"
#include "nvvkhl/shaders/dh_tonemap.h"

layout(set = 0, binding = eFrameInfo) uniform FrameInfo_
{
  FrameInfo frameInfo;
};


layout(location = 0) in Interpolants
{
  float outerShell;
}
IN;


void main()
{
  outColor = vec4(toLinear(mix(vec3(0, 1, 1), vec3(1, 1, 0), IN.outerShell)), 1);
}
