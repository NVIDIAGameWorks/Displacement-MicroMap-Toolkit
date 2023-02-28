// This function returns the geometric information at hit point
// Note: depends on the buffer layout PrimMeshInfo

#ifndef GETHIT_GLSL
#define GETHIT_GLSL

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "octant_encoding.h"
#include "nvvkhl/shaders/ray_util.glsl"
#include "hit_state.h"

precision highp float;

HitState getHitState(in DeviceMeshInfo pinfo, in vec3 barycentrics, in int triID, in mat4x3 objectToWorld, in mat4x3 worldToObject, in vec3 worldRayDirection, bool frontFacing)
{
  HitState hit;

  // Vextex and indices of the primitive
  Vertices vertices = Vertices(pinfo.vertexPositionNormalBuffer);
  Indices  indices  = Indices(pinfo.triangleVertexIndexBuffer);

  // Getting the 3 indices of the triangle (local)
  uvec3 triangleIndex = indices.i[triID];

  // Positions.
  const vec3 pos0 = vertices.v[triangleIndex.x].xyz;
  const vec3 pos1 = vertices.v[triangleIndex.y].xyz;
  const vec3 pos2 = vertices.v[triangleIndex.z].xyz;

  // Normals
  const vec3 nrm0 = oct32_to_vec(floatBitsToInt(vertices.v[triangleIndex.x].w));
  const vec3 nrm1 = oct32_to_vec(floatBitsToInt(vertices.v[triangleIndex.y].w));
  const vec3 nrm2 = oct32_to_vec(floatBitsToInt(vertices.v[triangleIndex.z].w));

  // Texture coordinates if exist
  vec2 uv0 = vec2(0);
  vec2 uv1 = vec2(0);
  vec2 uv2 = vec2(0);
  if(pinfo.vertexTexcoordBuffer != 0)
  {
    TexCoords texcoords = TexCoords(pinfo.vertexTexcoordBuffer);
    uv0                 = texcoords.v[triangleIndex.x];
    uv1                 = texcoords.v[triangleIndex.y];
    uv2                 = texcoords.v[triangleIndex.z];
  }

  // Tangents coordinates if exist
  vec4 tng0 = vec4(1, 0, 0, 1);
  vec4 tng1 = vec4(1, 0, 0, 1);
  vec4 tng2 = vec4(1, 0, 0, 1);
  if(pinfo.vertexTangentSpaceBuffer != 0)
  {
    Tangents tangents = Tangents(pinfo.vertexTangentSpaceBuffer);
    tng0 = vec4(oct32_to_vec(floatBitsToInt(tangents.v[triangleIndex.x].x)), tangents.v[triangleIndex.x].y);
    tng1 = vec4(oct32_to_vec(floatBitsToInt(tangents.v[triangleIndex.y].x)), tangents.v[triangleIndex.y].y);
    tng2 = vec4(oct32_to_vec(floatBitsToInt(tangents.v[triangleIndex.z].x)), tangents.v[triangleIndex.z].y);
  }

  // Position
  hit.pos = mixBary(pos0, pos1, pos2, barycentrics);
  hit.pos = pointOffset(hit.pos, pos0, pos1, pos2, nrm0, nrm1, nrm2, barycentrics);  // Shadow offset position - hacking shadow terminator
  hit.pos = vec3(objectToWorld * vec4(hit.pos, 1.0));

  // Normal
  hit.nrm    = normalize(mixBary(nrm0, nrm1, nrm2, barycentrics));
  hit.nrm    = normalize(vec3(hit.nrm * worldToObject));
  hit.geonrm = normalize(cross(pos1 - pos0, pos2 - pos0));
  hit.geonrm = normalize(vec3(hit.geonrm * worldToObject));

  // TexCoord
  hit.uv = mixBary(uv0, uv1, uv2, barycentrics);

  // Tangent - Bitangent
  hit.tangent   = normalize(mixBary(tng0.xyz, tng1.xyz, tng2.xyz, barycentrics));
  hit.tangent   = vec3(objectToWorld * vec4(hit.tangent, 0.0));
  hit.tangent   = normalize(hit.tangent - hit.nrm * dot(hit.nrm, hit.tangent));
  hit.bitangent = cross(hit.nrm, hit.tangent) * tng0.w;

  // Adjusting normal
  const vec3 V = -worldRayDirection;
  if(dot(hit.geonrm, V) < 0)  // Flip if back facing
    hit.geonrm = -hit.geonrm;

  // Flip direction vectors for backfaces
  if(!frontFacing)
  {
    hit.nrm       = -hit.nrm;
    hit.tangent   = -hit.tangent;
    hit.bitangent = -hit.bitangent;
  }

  return hit;
}


#endif
