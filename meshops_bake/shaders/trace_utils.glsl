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

const float FLT_MAX = 3.402823466e+38F;

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  mat4x3 objectToWorld;
  mat4x3 worldToObject;
  vec2   texCoord;
  vec3   normal;
  vec4   tangent;
  float  distance;
};

bool keepNearest(inout float value, float compare, float target)
{
  bool closer = abs(compare - target) < abs(value - target);
  value = closer ? compare : value;
  return closer;
}

vec2 mixBary(vec2 a, vec2 b, vec2 c, vec3 bary)
{
  return a * bary.x + b * bary.y + c * bary.z;
}

vec3 mixBary(vec3 a, vec3 b, vec3 c, vec3 bary)
{
  return a * bary.x + b * bary.y + c * bary.z;
}

vec4 mixBary(vec4 a, vec4 b, vec4 c, vec3 bary)
{
  return a * bary.x + b * bary.y + c * bary.z;
}

// Intersects a triangle defined by three points a, b, c. Returns true if the
// ray is not near-parallel with the plane and is within the triangle, plus the
// expandEdge factor.
bool intersectTriangle(vec3 aToRayStart, vec3 rayDir, vec3 a, vec3 b, vec3 c, out float t)
{
  const float expandEdge = 0.1;
  vec3  planeNorm = cross(b - a, c - a);
  float det       = 1.0 / dot(rayDir, planeNorm);
  t               = dot(aToRayStart, -planeNorm) * det;

  // Return false if the ray is outside the triangle
  vec3  q = cross(aToRayStart, rayDir);
  float u = dot(q, b - a) * det;
  float v = dot(-q, c - a) * det;
  if(u < -expandEdge || v < -expandEdge || (u + v) > 1.0 + expandEdge)
    return false;

  // abs() to allow back faces from converging direction vectors
  return abs(det) < 1e+12;
}

// Returns the scalar projection of (point - start) onto dir in units of dir.
float rayNearestT(vec3 start, vec3 dir, vec3 point)
{
  return dot(dir, point - start) / dot(dir, dir);
}

// Computes a range to trace relative to the direction vectors length.
// wDirLen is the direction vector's length in world space
// baseRelDist is the relative position of the ray start, before applying fitted direction bounds
// maxRelDist is the relative position of the max trace distance, before applying fitted direction bounds
vec2 computeTraceRange(float wDirLen, float minT, float maxT)
{
  // Apply maxDistanceFactor to the trace bounds to correct for bounds that are
  // too tight or intersect the reference geometry. This distance is relative to
  // the initial vertex direction bounds. For a constant margin, see epsilon in
  // traceRay().
  float newRangeScale = (maxT - minT) * pc.maxDistanceFactor;
  if(pc.uniDirectional != 0)
  {
    // Center the new range if not tracing backwards too.
    minT -= (newRangeScale - (maxT - minT)) * 0.5;
  }
  maxT = minT + newRangeScale;

  // Optionally replace the trace length with a world space value
  if(pc.replaceDirectionLength != 0)
  {
    maxT = minT + pc.maxDistance / wDirLen;
  }

  // Trace backwards if not uniDirectional
  if(pc.uniDirectional == 0)
  {
    minT -= (maxT - minT);
  }

  // Conservatively adds max heightmap displacement to the max trace distance
  if(pc.highMeshHasDisplacement != 0)
  {
    BakerMeshInfos referenceInfos = BakerMeshInfos(sceneDesc.referenceMeshAddress);
    // Index 0 as we are only tracing against one mesh at a time!
    BakerMeshInfo referenceInfo = referenceInfos.i[0];
    minT -= referenceInfo.maxDisplacementWs / wDirLen;
    maxT += referenceInfo.maxDisplacementWs / wDirLen;
  }

  return vec2(minT, maxT);
}

//--------------------------------------------------------------------------------------------------------
// Tracing to find the nearest intersection either forwards or backwards. Start at the range max and trace back towards
// the origin until range min. Returns true if an intersection was found. The HitState will contain surface properties
// of the closest hit to the origin. hit.t will contain the signed distance from the origin.
// traceRange is the range to trace relative to the origin and in units of direction
// returns the intersection closest to keepClosestOrigin, relative to the origin and in units of direction
bool traceRay(in rayQueryEXT rayQuery, vec3 origin, vec3 direction, uint flags, vec2 traceRange, float keepClosestOrigin, inout HitState hit)
{
  const float epsilon = 4e-7;

  // Trace from the range minimum to the maximum (units of the direction
  // vector). Add a small offset so we definitely hit high geometry that's
  // exactly on the upper bound.
  float traceStart = traceRange.x - epsilon;
  origin += direction * traceStart;

  // Compute the total trace distance and add a small offset so we definitely
  // hit high geometry that's exactly on the lower bound.
  float rayDistance = traceRange.y - traceStart + epsilon;
  rayQueryInitializeEXT(rayQuery, topLevelAS, flags, 0xFF, origin, 0.0, direction, rayDistance);

  // Check all intersections and keep the one closest to origin
  float nearestT = FLT_MAX;
  while(rayQueryProceedEXT(rayQuery))
  {
    // Add traceStart as this relative distance was also added to the origin
    // before tracing
    float t = rayQueryGetIntersectionTEXT(rayQuery, false) + traceStart;

    if(keepNearest(nearestT, t, keepClosestOrigin))
    {
      hit.distance = t;

      int  primitiveID = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false);
      vec2 baryCoord2  = rayQueryGetIntersectionBarycentricsEXT(rayQuery, false);
      vec3 baryCoord3  = vec3(1.0 - baryCoord2.x - baryCoord2.y, baryCoord2);

      BakerMeshInfos pInfo_ = BakerMeshInfos(sceneDesc.referenceMeshAddress);

      // Only tracing against at one mesh at a time!
      BakerMeshInfo pinfo = pInfo_.i[0];

      Indices  indices  = Indices(pinfo.indexAddress);
      Vertices vertices = Vertices(pinfo.vertexAddress);

      uvec3  tri = indices.i[primitiveID];
      Vertex a   = decompressVertex(vertices.cv[tri.x]);
      Vertex b   = decompressVertex(vertices.cv[tri.y]);
      Vertex c   = decompressVertex(vertices.cv[tri.z]);

      hit.normal   = mixBary(a.normal, b.normal, c.normal, baryCoord3);
      hit.tangent  = mixBary(a.tangent, b.tangent, c.tangent, baryCoord3);
      hit.texCoord = mixBary(a.texCoord, b.texCoord, c.texCoord, baryCoord3);

      hit.objectToWorld = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, false);
      hit.worldToObject = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, false);
    }
  }

  return nearestT != FLT_MAX;
}

// Creates a matrix whose columns are an orthonormal basis with a world-space
// tangent, bitangent, and normal, from an object-space normal, tangent,
// tangent w component as defined by glTF, and object-to-world and
// world-to-object matrices.
// This will use the input normal converted to world-space, but it may move
// the tangent in the plane span{normal, tangent} so that the basis
// is orthonormal. Note that the TBN will be right-handed if
// bitangentSign == 1.0f, and left-handed if bitangentSign == -1.0f.
mat3 makeTangentSpace(in mat3x3 objectToWorld, in mat3x3 worldToObject, in vec3 normal, in vec3 tangent, in float bitangentSign)
{
  // Positions are transformed from object-space to world-space as
  // (objectToWorld * v). Normals transform using matrices proportional to
  // the conjugate transpose, while tangents and bitangents use the same matrix
  // as positions.
  vec3 wsNormal = normalize(normal * worldToObject);  // == normalize(objectToWorld^(-T) * normal)
  // Please see MICROSDK-191 for tangent/bitangent conventions.
  vec3 wsTangent   = normalize(objectToWorld * tangent.xyz);
  vec3 wsBitangent = cross(wsNormal, wsTangent) * bitangentSign;
  return mat3(wsTangent, wsBitangent, wsNormal);
}
