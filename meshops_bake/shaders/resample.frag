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
#extension GL_EXT_shader_image_load_formatted : require
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

layout(set = 0, binding = eTexturesIn) uniform sampler2D[] texturesIn;
layout(set = 0, binding = eTexturesOut) uniform image2D[] texturesOut;
layout(set = 0, binding = eTexturesDist, r32f) uniform image2D[] texturesDist;

layout(push_constant) uniform PushContrive_
{
  PushHighLow pc;
};

layout(location = 0) in Interpolants
{
  Vertex     vert;
  vec3       traceStart;  // Bounds fitting minimum, corresponding to zero displacement
  vec3       traceDir;    // Bounds fitting direction vector, one unit of displacement
  vec3       traceBase;   // Interpolated original bounds minimum (before fitting)
  vec3       traceMax;    // Interpolated original bounds maximum (before fitting)
  flat int   instance;
  flat uvec3 triIndices;
}
IN;

#include "trace_utils.glsl"

float sinc(float x)
{
  const float M_PI = 3.14159265359;
  return sin(x * M_PI) / (x * M_PI);
}


float lanczos_value(int width, float x)
{
  float abs_x = abs(x);

  const float epsilon = 1.0e-6;
  if(abs_x >= float(width))
  {
    return 0.0;
  }
  if(abs_x < epsilon)
  {
    return 1.0;
  }

  return sinc(x) * sinc(x / width);
}

vec4 lanczos(int width, sampler2D s, vec2 normCoord)
{
  const int lw           = width - 1;
  ivec2     sizeI        = textureSize(s, 0);
  vec2      size         = vec2(sizeI);
  vec2      coord        = fract(normCoord - vec2(0.5) / size) * size;
  ivec2     icoord       = ivec2(coord);
  vec4      total        = vec4(0.0);
  float     kernel_total = 0.0;

  // Calling texelFetch outside the bounds of an image is undefined
  // behavior unless robust buffer access is enabled. For now, we give such
  // samples weight 0. TODO: Use texture wrap mode from glTF.
  ivec2 windowMin = max(ivec2(0), icoord - lw);
  ivec2 windowMax = min(icoord + lw, sizeI - 1);
  for(int x = windowMin.x; x <= windowMax.x; x++)
  {
    for(int y = windowMin.y; y <= windowMax.y; y++)
    {
      ivec2 samplePos = ivec2(x, y);
      vec2  delta     = coord - vec2(samplePos);
      float lz        = lanczos_value(width, delta.x) * lanczos_value(width, delta.y);
      kernel_total += lz;
      total += lz * texelFetch(s, samplePos, 0);
    }
  }

  return total / kernel_total;
}

// Quaternion logic
// Here, we store a quaternion xi + yj + zk + w as a vec4(x, y, z, w).

// Multiplies two quaternions; returns qr.
vec4 quatMultiply(vec4 q, vec4 r)
{
  return vec4(cross(q.xyz, r.xyz) + q.w * r.xyz + r.w * q.xyz, q.w * r.w - dot(q.xyz, r.xyz));
}

// Rotates a unit quaternion q by the 3D vector v (i.e. returns q v q^*, where
// q^* denotes the conjugate of q). This optimized version is from
// Fabian "ryg" Giesen, "Rotating a single vector using a quaternion":
// https://fgiesen.wordpress.com/2019/02/09/rotating-a-single-vector-using-a-quaternion/
vec3 quatRotate(vec4 q, vec3 v)
{
  const vec3 t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

// Given unit vectors n1 ("normal 1"), n2 ("normal 2"), t1 ("tangent 1"), and
// t2 ("tangent 2"), returns a unit quaternion q such that:
// 1.   q rotates n1 to n2 (i.e. q n1 q^* == n2)
// 2.   q rotates t1 as close as possible under the L2 norm to t2
//      (i.e. q minimizes distance(q t1 q^*, t2)).
vec4 computeQuaternionTexture(vec3 n1, vec3 n2, vec3 t1, vec3 t2)
{
  // The idea behind this algorithm is that after rotating n1 to n2, we have
  // one extra degree of freedom: q == q2 q1, where q1 is a unit quaternion
  // that rotates n1 to n2, and q2 is a unit quaternion that preserves n2
  // (that is, q2 is a rotation by some angle about n2).

  // Without loss of generality, we can choose any quaternion that rotates n1
  // to n2. Let's use a 180-degree rotation about the bisector of n1 and n2,
  // normalize(n1+n2), because it's fast to write out.
  const float SINGULARITY_EPS = 1e-6f; // Ad-hoc detection threshold for singularities
  const vec3 n1_plus_n2 = n1 + n2;
  const float n1_plus_n2_squared_len = dot(n1_plus_n2, n1_plus_n2);
  vec4 q1;
  if(n1_plus_n2_squared_len > SINGULARITY_EPS)
  {
    q1 = vec4(n1_plus_n2 / sqrt(n1_plus_n2_squared_len), 0.);
  }
  else
  {
    // We have a singularity when n1 == -n2. In this case, a 180-degree
    // rotation about any vector perpendicular to n1 will work.
    // We use the Duff et al. branchless algorith from "Building an Orthonormal
    // Basis, Revisited", Journal of Computer Graphics Techniques, 2017.
    const float sign = (n1.z >= 0.0 ? 1.0 : -1.0); // Since GLSL sign(0) == 0
    const float a    = -1.0 / (sign + n1.z);
    const float b    = n1.x * n1.y * a;
    q1 = vec4(b, sign + n1.y * n1.y * a, -n1.y, 0.);
  }

  // Now we know that our answer will be q1 followed by q2, a rotation by an
  // angle t about n2. What's that angle? First, determine what q1 rotates
  // t1 to:
  const vec3 it1 = quatRotate(q1, t1);
  // q2 will spin it1 in a circle about n2. Build a local coordinate system
  // for that spin.
  const vec3 center = n2 * dot(n2, it1); // Center of the circle
  // Our local x coordinate axis will be perpendicular to n2 and aligned to it1.
  // We don't need to normalize it.
  vec3 vx = it1 - center;
  // Local y coordinate axis; vx, vy, n2 is a right-handed orthonormal system.
  // vy has the same length as vx.
  const vec3 vy = cross(n2, vx);
  // Now we can read the cosine and sine of t (times a common factor), which is
  // going to be t2 projected to the plane of the circle.
  const float t2_projected_x = dot(t2, vx);
  const float t2_projected_y = dot(t2, vy);
  // To compute the quaternion q2, we need the cosine and sine of t/2. We can
  // do this algebraically using the half-angle formulas - but it's long enough
  // that atan2 + cos + sin should give similar performance.
  vec4 q2;
  if(abs(t2_projected_x) > SINGULARITY_EPS)
  {
    const float t = atan(t2_projected_y, t2_projected_x);
    const float t_over_2 = t * 0.5;
    q2 = vec4(sin(t_over_2) * n2, cos(t_over_2));
  }
  else
  {
    // GLSL atan(t2_projected_y, t2_projected_x) isn't guaranteed to be defined
    // if t2_projected_x == 0, and we have a singularity if
    // t2_projected_y == t2_projected_x == 0.
    // In the former case, we know t == sign(t2_projected_y) * pi/2, so we can
    // write out the result. In the singularity case, we know that the image of
    // t1 after rotation by q2 q1 will be the same distance away from t2 no
    // matter what angle we choose. So we may as well choose t == +- pi/2.
    q2 = vec4((t2_projected_y >= 0.0 ? 1.0 : -1.0) * sqrt(0.5) * n2, sqrt(0.5));
  }
  // That gives us q!
  const vec4 q = quatMultiply(q2, q1);
  // Because unit quaternions double-cover the space of rotations, there are
  // two possible answers (multiply the quaternion by -1 to get the other one;
  // this is equivalent to adding or subtracting 360 degrees from the
  // axis-angle rotation). We choose the quaternion with less rotation when
  // possible (i.e. q.w >= 0).
  return (q.w >= 0) ? q : -q;
}

void main()
{
  // Ray start/direction comes from interpolated vertex shader values. These have already had direction bounds applied.
  vec3       start  = IN.traceStart;
  vec3       dir    = IN.traceDir;
  const vec3 wStart = vec3(pc.objectToWorld * vec4(start, 1.0));
  const vec3 wDir   = mat3(pc.objectToWorld) * dir;

  PrimMeshInfos   pInfo_              = PrimMeshInfos(sceneDesc.primLoInfoAddress);
  PrimMeshInfo    pinfo               = pInfo_.i[pc.primMeshID];
  Indices         indices             = Indices(pinfo.indexAddress);
  Vertices        vertices            = Vertices(pinfo.vertexAddress);
  DirectionBounds directionBoundsOrig = DirectionBounds(pinfo.vertexDirectionBoundsOrigAddress);

  Vertex v0 = decompressVertex(vertices.cv[IN.triIndices.x]);
  Vertex v1 = decompressVertex(vertices.cv[IN.triIndices.y]);
  Vertex v2 = decompressVertex(vertices.cv[IN.triIndices.z]);

  float boundsScaleEpsilon = 1e-5;

  // Get original bounds, before any bounds fitting iterations
  vec2 boundsOrig0 = pc.hasDirectionBounds != 0? directionBoundsOrig.b[IN.triIndices.x] : vec2(0, 1);
  vec2 boundsOrig1 = pc.hasDirectionBounds != 0? directionBoundsOrig.b[IN.triIndices.y] : vec2(0, 1);
  vec2 boundsOrig2 = pc.hasDirectionBounds != 0? directionBoundsOrig.b[IN.triIndices.z] : vec2(0, 1);
  boundsOrig0.y = max(boundsOrig0.y, boundsScaleEpsilon);
  boundsOrig1.y = max(boundsOrig1.y, boundsScaleEpsilon);
  boundsOrig2.y = max(boundsOrig2.y, boundsScaleEpsilon);
  const vec3 traceBase0  = v0.position + v0.displacementDirection * boundsOrig0.x;
  const vec3 traceBase1  = v1.position + v1.displacementDirection * boundsOrig1.x;
  const vec3 traceBase2  = v2.position + v2.displacementDirection * boundsOrig2.x;
  const vec3 traceMax0   = traceBase0 + v0.displacementDirection * boundsOrig0.y;
  const vec3 traceMax1   = traceBase1 + v1.displacementDirection * boundsOrig1.y;
  const vec3 traceMax2   = traceBase2 + v2.displacementDirection * boundsOrig2.y;

  // Compute surface origin and min/max trace distances relative to the current ray
  float traceOriginT = 0.0;
  float traceMinT    = 0.0;
  float traceMaxT    = 0.0;
  if(!intersectTriangle(start - v0.position, dir, v0.position, v1.position, v2.position, traceOriginT))
  {
    traceOriginT = rayNearestT(start, dir, IN.vert.position.xyz);
  }
  if(!intersectTriangle(start - traceBase0, dir, traceBase0, traceBase1, traceBase2, traceMinT))
  {
    traceMinT = rayNearestT(start, dir, IN.traceBase);
  }
  if(!intersectTriangle(start - traceMax0, dir, traceMax0, traceMax1, traceMax2, traceMaxT))
  {
    traceMaxT = rayNearestT(start, dir, IN.traceMax);
  }

  // Command line arguments may override the trace distance and/or trace both forwards and backwards
  vec2 traceRange = computeTraceRange(length(wDir), traceMinT, traceMaxT);

  //------------------------------------------------------------
  // Trace ray
  rayQueryEXT rayQuery;
  HitState    hit;
  if(!traceRay(rayQuery, wStart, wDir, gl_RayFlagsNoneNV, traceRange, traceOriginT, hit))
  {
    // Leave this pixel empty if there is no hit
    discard;
    return;
  }

  // Note: the instance transforms from the hit should all be the same because meshes are processed one at a time and there's only one instance
  mat3 highTBN =
      makeTangentSpace(mat3(hit.objectToWorld), mat3(hit.worldToObject), hit.normal, hit.tangent.xyz, hit.tangent.w);
  mat3 lowTBN = makeTangentSpace(mat3(pc.objectToWorld), mat3(pc.worldToObject), IN.vert.normal.xyz,
                                 IN.vert.tangent.xyz, IN.vert.tangent.w);

  ivec2 targetRes = ivec2(unpackUint2x16(pc.resampleInstanceResolutions[IN.instance]));
  ivec2 outPixel  = clamp(ivec2(gl_FragCoord.xy), ivec2(0), targetRes);
  for(int i = 0; i < pc.numResampleTextures && i < MAX_RESAMPLE_TEXTURES; ++i)
  {
    // One instance is rendered per resolution. Only resample the textures for the current instance.
    if(targetRes != imageSize(texturesOut[i]))
      continue;

    // Keep samples less than or equal to the current distance
    // NOTE: texturesDist may contain duplicate textures
    float       currentDist     = imageLoad(texturesDist[i], outPixel).r;
    const float absHitDistance  = abs(hit.distance);
    const bool  passedDepthTest = absHitDistance <= currentDist;
    if(passedDepthTest)
      imageStore(texturesDist[i], outPixel, vec4(abs(hit.distance)));
    else
      continue;

    vec4 texSample = vec4(0.0f);
    const uint textureType = bitfieldExtract(pc.textureInfo[i].bits, TEXINFO_TYPE_OFFSET, TEXINFO_TYPE_BITS);
    const uint inputIndex = bitfieldExtract(pc.textureInfo[i].bits, TEXINFO_INDEX_OFFSET, TEXINFO_INDEX_BITS);
    if(textureType == eGeneric)
    {
      texSample = lanczos(4, texturesIn[i], hit.texCoord);
    }
    else if(textureType == eNormalMap)
    {
      texSample = lanczos(4, texturesIn[i], hit.texCoord);
      // Read from the normal map, and transform it to world-space.
      vec3 highNormal = texSample.xyz * 2.0 - 1.0;
      vec3 wsNormal   = highTBN * highNormal;
      // Transform the world-space normal to be relative to the micromesh's
      // interpolated tangent space. Because lowTBN is orthonormal,
      // transpose(lowTBN) is a faster way to compute inverse(lowTBN).
      vec3 lowNormal = transpose(lowTBN) * wsNormal;
      texSample.xyz  = lowNormal * 0.5 + 0.5;
    }
    else if(textureType == eQuaternionMap)
    {
      // Compute the high-res normal and tangent, relative to the micromesh's
      // interpolated tangent space. The space in which we compute quaternion
      // transformations is important because rotations don't commute
      // (applying rotation Q in space A is not the same as transforming to
      // space B, applying Q, and transforming back to space A).
      // In this space, choosing the micromesh's tangent space is a good
      // choice: apps usually know how this tangent space changes under
      // deformations, because they likely already implement such logic for
      // normal maps. They can unpack the normal and tangent from the
      // quaternion as if reading a normal from a normal map, apply any
      // deformation adjustments, and then continue as before.
      // As with normal maps, because lowTBN is orthonormal, transpose(lowTBN)
      // is a faster way to compute inverse(lowTBN).
      mat3x3 lowTBNInverse = transpose(lowTBN);
      vec3 relativeHighNormal = lowTBNInverse * highTBN[2];
      vec3 relativeHighTangent = lowTBNInverse * highTBN[0];
      // Compute the quaternion that rotates the Z+ and X+ coordinate axes
      // to the normal (exactly) and the tangent (as close as possible).
      texSample = 0.5 + 0.5 * computeQuaternionTexture(vec3(0., 0., 1.), relativeHighNormal,
                                                       vec3(1., 0., 0.), relativeHighTangent);
    }
    else if(textureType == eOffsetMap)
    {
      // To convert from pixel indices to UV coordinates, we add 0.5f, because
      // pixel centers are located at half-integers.
      const vec2 outUV = (vec2(outPixel) + vec2(0.5f)) / vec2(targetRes);
      texSample = vec4(0.5 * (hit.texCoord - outUV) + 0.5, 0.0, 1.0);
    }
    else if(textureType == eHeightMap)
    {
      texSample = vec4((hit.distance - pc.globalMinMax.x) / (pc.globalMinMax.y - pc.globalMinMax.x), 0.0, 0.0, 1.0);
    }

    imageStore(texturesOut[i], outPixel, texSample);
  }
}
