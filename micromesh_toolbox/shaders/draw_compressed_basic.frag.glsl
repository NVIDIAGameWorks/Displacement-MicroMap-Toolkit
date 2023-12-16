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
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_EXT_buffer_reference2 : require
#extension GL_NV_fragment_shader_barycentric : enable
#extension GL_EXT_debug_printf : enable

#include "common_micromesh_compressed.h"

// Incoming
layout(location = 0) in Interpolants
{
  vec3 pos;
  vec3 bary;
}
IN;

layout(location = 3) in flat uint shadeModeValue;


// Outgoing
layout(location = 0) out vec4 outColor;

#include "raster_common.glsl"

void main()
{
  GltfShadeMaterial gltfMat;
  DeviceMeshInfo    pinfo;
  TriangleAttribute triInfo;
  HitState          hit;  // Interpolated Hit from vertex shader
  vec3              toEye;
  rasterLoad(IN.pos, IN.bary, gl_PrimitiveID, gltfMat, pinfo, triInfo, hit, toEye);

  // Override base triangle position and normal with those for the microtriangle
  hit.pos    = IN.pos;
  hit.geonrm = normalize(-cross(dFdx(IN.pos), dFdy(IN.pos)));

  // Meshes with micromaps usually includes normal maps. Without them, displaced
  // geometry will be shaded flat. If the mesh has none, use microtriangle face
  // normals instead.
  if(CONST_SHADE_MODE == eRenderShading_default && gltfMat.normalTexture == -1)
  {
    hit.nrm = hit.geonrm;
  }

  float NdotL = dot(hit.nrm, toEye);

  if(CONST_SHADE_MODE == eRenderShading_anisotropy)
  {
    float anisotropy = intBitsToFloat(int(shadeModeValue));
    vec3  color      = toLinear(colorMap(frameInfo.colormap, anisotropy));
    outColor         = simpleShade(color, NdotL);
  }
  else if(CONST_SHADE_MODE == eRenderShading_baseTriangleIndex)
  {
    uint id     = shadeModeValue & 0x3FF;
    uint baseID = shadeModeValue >> 10;
    vec3 col1   = uintToColor(id);
    vec3 col2   = uintToColor(baseID);
    vec3 color  = toLinear(mix(col1, col2, 0.8));  // Emphasis on Base triangle
    outColor    = simpleShade(color, NdotL);
  }
  else if(CONST_SHADE_MODE == eRenderShading_subdivLevel)
  {
    vec3 color = subdDecFlagsToColor(shadeModeValue, MAX_BASE_SUBDIV, triInfo.primitiveFlags, IN.bary);
    outColor   = simpleShade(color, NdotL);
  }
  else if(CONST_SHADE_MODE == eRenderShading_minMax)
  {
    vec3 color = toLinear(colorMap(frameInfo.colormap, intBitsToFloat(int(shadeModeValue))));
    outColor   = simpleShade(color, NdotL);
  }
  else if(CONST_SHADE_MODE == eRenderShading_compressionFormat)
  {
    vec3 color = toLinear(colorMap(frameInfo.colormap, float(shadeModeValue) / float(MICRO_MAX_FORMATS)));
    outColor   = simpleShade(color, NdotL);
  }
  else
  {  // Normal shading and other debug information
    outColor = rasterShade(IN.bary, gltfMat, pinfo, triInfo, hit, toEye, gl_PrimitiveID);
  }
}
