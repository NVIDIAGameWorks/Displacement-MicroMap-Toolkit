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
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_debug_printf : enable
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_ray_query : enable
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_GOOGLE_include_directive : enable

#include "host_device.h"

// Buffers
layout(buffer_reference, scalar) readonly buffer PrimMeshInfos
{
  PrimMeshInfo i[];
};
layout(buffer_reference, scalar) readonly buffer Vertices
{
  CompressedVertex cv[];
};
layout(buffer_reference, scalar) readonly buffer Indices
{
  uvec3 i[];
};
layout(buffer_reference, scalar) readonly buffer DirectionBounds
{
  vec2 b[];
};
layout(set = 0, binding = eTlas) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = eSceneDesc) readonly buffer SceneDesc_
{
  SceneDescription sceneDesc;
};

layout(push_constant) uniform PushContrive_
{
  PushHighLow pc;
};

layout(location = 0) out Interpolants
{
  Vertex     vert;
  vec3       traceStart;  // Bounds fitting minimum, corresponding to zero displacement
  vec3       traceDir;    // Bounds fitting direction vector, one unit of displacement
  vec3       traceBase;   // Interpolated original bounds minimum (before fitting)
  vec3       traceMax;    // Interpolated original bounds maximum (before fitting)
  flat int   instance;
  flat uvec3 triIndices;
}
OUT;

void main()
{
  PrimMeshInfos   pInfo_              = PrimMeshInfos(sceneDesc.primLoInfoAddress);
  PrimMeshInfo    pinfo               = pInfo_.i[pc.primMeshID];
  Indices         indices             = Indices(pinfo.indexAddress);
  Vertices        vertices            = Vertices(pinfo.vertexAddress);
  DirectionBounds directionBounds     = DirectionBounds(pinfo.vertexDirectionBoundsAddress);
  DirectionBounds directionBoundsOrig = DirectionBounds(pinfo.vertexDirectionBoundsOrigAddress);

  uvec3  triIndices = indices.i[gl_VertexIndex / 3];
  uint   vertIndex  = triIndices[gl_VertexIndex % 3];
  Vertex v          = decompressVertex(vertices.cv[vertIndex]);
  vec2   texCoord   = v.texCoord;
  OUT.vert          = v;
  OUT.triIndices    = triIndices;

  float boundsScaleEpsilon = 1e-5;

  // Adjust the vertex position and direction based on direction vector bounds
  vec2 bounds    = pc.hasDirectionBounds != 0 ? directionBounds.b[vertIndex] : vec2(0, 1);
  bounds.y       = max(bounds.y, boundsScaleEpsilon);
  OUT.traceStart = v.position + v.displacementDirection * bounds.x;
  OUT.traceDir   = v.displacementDirection * bounds.y;

  vec2 boundsOrig = pc.hasDirectionBounds != 0 ? directionBoundsOrig.b[vertIndex] : vec2(0, 1);
  OUT.traceBase   = v.position + v.displacementDirection * boundsOrig.x;
  OUT.traceMax    = OUT.traceBase + v.displacementDirection * boundsOrig.y;

  // Handle UDIMs by subtracting the UDIM coordinate of the triangle's center
  vec2 a                = decompressVertex(vertices.cv[triIndices.x]).texCoord;
  vec2 b                = decompressVertex(vertices.cv[triIndices.y]).texCoord;
  vec2 c                = decompressVertex(vertices.cv[triIndices.z]).texCoord;
  vec2 triangleCenterUV = (a + b + c) / 3.0;
  texCoord -= floor(triangleCenterUV);

  vec2 maxRes    = vec2(unpackUint2x16(pc.resampleMaxResolution));
  vec2 targetRes = vec2(unpackUint2x16(pc.resampleInstanceResolutions[gl_InstanceIndex]));
  OUT.instance   = gl_InstanceIndex;
  texCoord *= targetRes / maxRes;

  gl_Position = vec4(texCoord * 2.0 - 1.0, 0, 1);
}
