/*
 * Copyright (c) 2014-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


uint wangHash(uint seed)
{
  seed = (seed ^ 61) ^ (seed >> 16);
  seed *= 9;
  seed = seed ^ (seed >> 4);
  seed *= 0x27d4eb2d;
  seed = seed ^ (seed >> 15);
  return seed;
}

uint xorshift32(uint x64)
{
  x64 ^= x64 << 13;
  x64 ^= x64 >> 7;
  x64 ^= x64 << 17;
  return x64;
}


#define ADD_HASH(type_, components_, conversion_)                                                                      \
  uvec2 addHash(uvec2 h, type_ val)                                                                                    \
  {                                                                                                                    \
    for(uint i = 0; i < components_; i++)                                                                              \
    {                                                                                                                  \
      h = uvec2(wangHash(h.x + conversion_(val[i])), xorshift32(h.y + conversion_(val[i])));                           \
    }                                                                                                                  \
    return h;                                                                                                          \
  }
#define ADD_HASH1(type_, conversion_)                                                                                  \
  uvec2 addHash(uvec2 h, type_ val)                                                                                    \
  {                                                                                                                    \
    return uvec2(wangHash(h.x + conversion_(val)), xorshift32(h.y + conversion_(val)));                                \
  }

ADD_HASH(uvec2, 2, uint)
ADD_HASH1(float, floatBitsToUint)
ADD_HASH1(uint, uint)
ADD_HASH(vec2, 2, floatBitsToUint)
ADD_HASH(vec3, 3, floatBitsToUint)
ADD_HASH(vec4, 4, floatBitsToUint)
