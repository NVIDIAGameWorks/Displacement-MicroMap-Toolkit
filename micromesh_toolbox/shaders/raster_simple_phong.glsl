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

#ifndef RASTER_SIMPLE_PHONG
#define RASTER_SIMPLE_PHONG

vec3 simplePhong(in vec3 toEye, in vec3 normal)
{
  vec3 color    = vec3(0.8);
  vec3 wUpDir   = vec3(0, 1, 0);
  vec3 lightDir = toEye;
  vec3 eyeDir   = toEye;

  vec3  reflDir = normalize(-reflect(lightDir, normal));
  float lt      = abs(dot(normal, lightDir)) + pow(max(0, dot(reflDir, eyeDir)), 16.0) * 0.3;  // Diffuse + Specular
  color         = color * (lt);
  color += mix(vec3(0.1, 0.1, 0.4), vec3(0.8, 0.6, 0.2), dot(normal, wUpDir.xyz) * 0.5 + 0.5) * 0.2;  // Ambient term (sky effect)
  return toLinear(color);  // Gamma correction
}

//-----------------------------------------------------------------------
// This darkens the color when the normal is facing away
vec4 simpleShade(in vec3 color, in float NdotL)
{
  color = mix(color, vec3(1), vec3(0.15));  // makes everything brighter (washed out)
  color *= clamp(pow(max(NdotL, 0.2), 1.2) + 0.15, 0, 1);
  return vec4(toLinear(color), 1);
}

#endif
