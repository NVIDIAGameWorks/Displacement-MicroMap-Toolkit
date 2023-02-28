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
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require


// Incoming
layout(location = 0) in Interpolants
{
  vec3 pos;
}
IN;

// Outgoing
layout(location = 0) out vec4 outColor;

#include "device_host.h"
#include "dh_bindings.h"
#include "utility.h"

layout(set = 0, binding = eFrameInfo) uniform FrameInfo_
{
  FrameInfo frameInfo;
};


void main()
{
  outColor = intColorToVec4(frameInfo.overlayColor);
}
