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

#ifndef COMPRESS_DECOMPRESS_TYPES_H
#define COMPRESS_DECOMPRESS_TYPES_H

#ifdef __cplusplus  // GLSL Type
#include <glm/glm.hpp>
namespace shaders {
using namespace glm;
#define INLINE inline
#else
#define INLINE
#endif

// Common vertex types used by both micromesh_viewer and meshops_bake.
// Note that these are not used by the remesher as of yet.
struct Vertex
{
  vec3 position;  // Ray tracing attribute layout depends on position.xyz coming first.
  vec3 normal;
  vec4 tangent;  // w == bitangentSign
  vec3 displacementDirection;
  vec2 texCoord;
};

// `Vertex` lossily compressed into 36 bytes.
struct CompressedVertex
{
  // v[0], v[1], and v[2] are `vec3 position` stored losslessly.
  // Micromesh_viewer and meshops_bake's ray tracing attribute layout
  // depend on these coming first.
  // v[3] and v[4] store the 3 components of displacementDirection as half
  // floats, followed by 1 bit of a 16-bit field used for bitangentSign.
  // v[5] and v[6] store the normal in octahedral format.
  // v[7] and v[8] store texture coordinates losslessly.
  uint v[9];
};

#define C_Stack_Max 3.402823466e+38f
INLINE uint compress_unit_vec(vec3 nv)
{
  // map to octahedron and then flatten to 2D (see 'Octahedron Environment Maps' by Engelhardt & Dachsbacher)
  if((nv.x >= C_Stack_Max) || isinf(nv.x))
    return ~0u;

  const float d = 32767.0f / (abs(nv.x) + abs(nv.y) + abs(nv.z));
  int         x = int(roundEven(nv.x * d));
  int         y = int(roundEven(nv.y * d));

  if(nv.z < 0.0f)
  {
    const int maskx = x >> 31;
    const int masky = y >> 31;
    const int tmp   = 32767 + maskx + masky;
    const int tmpx  = x;
    x               = (tmp - (y ^ masky)) ^ maskx;
    y               = (tmp - (tmpx ^ maskx)) ^ masky;
  }

  uint packed = (uint(y + 32767) << 16) | uint(x + 32767);
  if(packed == ~0u)
    return ~0x1u;

  return packed;
}

INLINE float short_to_floatm11(const int v)  // linearly maps a short 32767-32768 to a float -1-+1 //!! opt.?
{
  return (v >= 0) ? (uintBitsToFloat(0x3F800000u | (uint(v) << 8)) - 1.0f) :
                    (uintBitsToFloat((0x80000000u | 0x3F800000u) | (uint(-v) << 8)) + 1.0f);
}

INLINE vec3 decompress_unit_vec(uint packed)
{
  if(packed == ~0u)  // sanity check, not needed as isvalid_unit_vec is called earlier
    return vec3(C_Stack_Max);


  int x = int(packed & 0xFFFFu) - 32767;
  int y = int(packed >> 16) - 32767;

  const int maskx = x >> 31;
  const int masky = y >> 31;
  const int tmp0  = 32767 + maskx + masky;
  const int ymask = y ^ masky;
  const int tmp1  = tmp0 - (x ^ maskx);
  const int z     = tmp1 - ymask;
  float     zf;
  if(z < 0)
  {
    x  = (tmp0 - ymask) ^ maskx;
    y  = tmp1 ^ masky;
    zf = uintBitsToFloat((0x80000000u | 0x3F800000u) | (uint(-z) << 8)) + 1.0f;
  }
  else
  {
    zf = uintBitsToFloat(0x3F800000u | (uint(z) << 8)) - 1.0f;
  }

  return normalize(vec3(short_to_floatm11(x), short_to_floatm11(y), zf));
}

INLINE CompressedVertex compressVertex(Vertex v)
{
  CompressedVertex cv;
  // Position
  cv.v[0] = floatBitsToUint(v.position.x);
  cv.v[1] = floatBitsToUint(v.position.y);
  cv.v[2] = floatBitsToUint(v.position.z);

  // Displacement direction
  cv.v[3] = packHalf2x16(vec2(v.displacementDirection));
  cv.v[4] = packHalf2x16(vec2(v.displacementDirection.z, 0.0));

  // Tangent sign in what would be displacementDirection.w
  cv.v[4] = (cv.v[4] & ~(0x1)) | ((v.tangent.w < 0.f) ? 0x0 : 0x1);

  // Normal
  cv.v[5] = compress_unit_vec(v.normal);

  // Tangent
  cv.v[6] = compress_unit_vec(vec3(v.tangent));

  // Texture coordinates
  cv.v[7] = floatBitsToUint(v.texCoord.x);
  cv.v[8] = floatBitsToUint(v.texCoord.y);

  return cv;
}

INLINE Vertex decompressVertex(CompressedVertex cv)
{
  Vertex v;
  v.position.x = uintBitsToFloat(cv.v[0]);
  v.position.y = uintBitsToFloat(cv.v[1]);
  v.position.z = uintBitsToFloat(cv.v[2]);

  v.displacementDirection = vec3(unpackHalf2x16(cv.v[3]), unpackHalf2x16(cv.v[4]).x);

  v.normal = decompress_unit_vec(cv.v[5]);

  {
    // Tangent sign bit is packed into what would be displacementDirection.w
    float tSign = (cv.v[4] & 0x1) == 1 ? 1.f : -1.f;
    vec3  t     = decompress_unit_vec(cv.v[6]);
    v.tangent   = vec4(t, tSign);
  }

  v.texCoord.x = uintBitsToFloat(cv.v[7]);
  v.texCoord.y = uintBitsToFloat(cv.v[8]);

  return v;
}

#ifdef __cplusplus
}  // namespace shaders
#endif

#endif