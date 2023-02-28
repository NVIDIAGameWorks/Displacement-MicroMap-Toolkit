/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// this file is included by C++ and GLSL

#ifndef _COMMON_H_
#define _COMMON_H_

#include "config.h"

// clang-format off


/////////////////////////////////////////////////
// general options
// 
#ifndef __cplusplus

    // see RendererVK::getShaderPrepend()
    // ------------------------------
    // must respect edgeDecimateFlags
    #ifndef USE_NON_UNIFORM_SUBDIV
    #define USE_NON_UNIFORM_SUBDIV 1
    #endif

    // Specifies if have per-microvertex normals within the per-microvertex data.
    #ifndef USE_MICROVERTEX_NORMALS
    #define USE_MICROVERTEX_NORMALS 0
    #endif

    #ifndef USE_TEXTURE_NORMALS
    #define USE_TEXTURE_NORMALS 0
    #endif
    
    #ifndef USE_DIRECTION_BOUNDS
    #define USE_DIRECTION_BOUNDS 1
    #endif

    // see Sample::getShaderPrepend()
    // ------------------------------

    // for micromesh, use triangle lod metric, otherwise sphere
    #ifndef USE_TRI_LOD
    #define USE_TRI_LOD 0
    #endif
    
    #ifndef USE_INSTANCE_LOD
    #define USE_INSTANCE_LOD 1
    #endif

    // compute additional statistics (costs perf)
    #ifndef USE_STATS
    #define USE_STATS 0
    #endif

    // use pixel derivatives to compute a flat shading normal
    #ifndef USE_FACET_SHADING
    #define USE_FACET_SHADING 0
    #endif
    
    // use proper fp16 types in micromesh displacement
    #ifndef USE_FP16_DISPLACEMENT_MATH
    #define USE_FP16_DISPLACEMENT_MATH 1
    #endif

    // applies additional restrictions in hlsl mesh shading
    // no read/write access to outputs
    // no 8/16 bit addressable task outputs
    #ifndef USE_MESH_HLSL_RESTRICTIONS
    #define USE_MESH_HLSL_RESTRICTIONS 0
    #endif
    
    // triangle culling
    #ifndef USE_PRIMITIVE_CULLING
    #define USE_PRIMITIVE_CULLING 0
    #endif
    
    // occlusion culling
    #ifndef USE_OCCLUSION_CULLING
    #define USE_OCCLUSION_CULLING 1
    #endif

    // Surface visualization. Affects the value of gl_PrimitiveID and
    // additional attributes sent to the fragment shader, and produces flat
    // colors instead of shading if not SURFACEVIS_SHADING.
    // See the SURFACEVIS_* definitions for more info.
    #ifndef SURFACEVIS
    #define SURFACEVIS SURFACEVIS_SHADING
    #endif

#endif
/////////////////////////////////////////////////

#define DSET_RENDERER 0
#define DSET_TEXTURES 1

// binding information for descriptor sets

#define DRAWSTD_UBO_VIEW      0
#define DRAWSTD_SSBO_STATS    1
#define DRAWSTD_UBO_MESH      2

#define DRAWRAY_UBO_VIEW      0
#define DRAWRAY_SSBO_STATS    1
#define DRAWRAY_IMG_OUT       2
#define DRAWRAY_UBO_MESH      3
#define DRAWRAY_ACC           4

#define SBT_ENTRY_RGEN        0
#define SBT_ENTRY_RMISS       1
#define SBT_ENTRY_RHIT        2
#define SBT_ENTRIES           3

//////////////////////////////////////////////////////////////////////////
// BUFFER_REF macro
// in GLSL buffer_reference is a typed global memory pointer provided as 64-bit VA
// in C++ we expose it as u64 to store the VA

#ifdef __cplusplus
#define BUFFER_REF(typ) uint64_t
#else
#define BUFFER_REF(typ) typ

layout(buffer_reference, buffer_reference_align = 4, scalar) restrict readonly buffer uint8s_in
{
    uint8_t d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict readonly buffer int8s_in
{
    int8_t d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict readonly buffer uint16s_in
{
    uint16_t d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict readonly buffer int16s_in
{
    int16_t d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict readonly buffer uints_in
{
    uint d[];
};
layout(buffer_reference, buffer_reference_align = 8, scalar) restrict readonly buffer uvec2s_in
{
    uvec2 d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict readonly buffer f16vec2s_in
{
    f16vec2 d[];
};
layout(buffer_reference, buffer_reference_align = 8, scalar) restrict readonly buffer f16vec4s_in
{
    f16vec4 d[];
};
layout(buffer_reference, buffer_reference_align = 8, scalar) restrict readonly buffer uint64s_in
{
    uint64_t d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict readonly buffer uvec3s_in
{
    uvec3 d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict readonly buffer floats_in
{
    float d[];
};
layout(buffer_reference, buffer_reference_align = 8, scalar) restrict readonly buffer vec2s_in
{
    vec2 d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict readonly buffer vec3s_in
{
    vec3 d[];
};
layout(buffer_reference, buffer_reference_align = 16, scalar) restrict readonly buffer vec4s_in
{
    vec4 d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict readonly buffer u8vec2s_in
{
    u8vec2 d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict writeonly buffer uints_out
{
    uint d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict writeonly buffer floats_out
{
    float d[];
};
layout(buffer_reference, buffer_reference_align = 8, scalar) restrict writeonly buffer vec2s_out
{
    vec2 d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict writeonly buffer vec3s_out
{
    vec3 d[];
};
layout(buffer_reference, buffer_reference_align = 8, scalar) restrict writeonly buffer f16vec4s_out
{
    f16vec4 d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict writeonly buffer f16vec2s_out
{
    f16vec2 d[];
};
layout(buffer_reference, buffer_reference_align = 8, scalar) restrict buffer f16vec4s_inout
{
    f16vec4 d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict buffer f16vec2s_inout
{
    f16vec2 d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict buffer vec3s_inout
{
    vec3 d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict buffer floats_inout
{
    float d[];
};
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict buffer uints_inout
{
    uint d[];
};
#endif

#if BOUNDS_AS_FP32
  #define boundsVec2      vec2
  #define boundsVec2s_in  vec2s_in
  #define boundsVec2s_out vec2s_out
#else
  #define boundsVec2      f16vec2
  #define boundsVec2s_in  f16vec2s_in
  #define boundsVec2s_out f16vec2s_out
#endif

//////////////////////////////////////////////////////////////////////////

#ifndef __cplusplus

// GLSL functions
// Uvec = encoded unit vector

#define unpackUvec   oct32_to_vec
#define packUvec     vec_to_oct32

#define getInterpolatedArrayUvec(arr, inds, uvw)  ( unpackUvec((arr)[inds.x]) * (uvw).x + unpackUvec((arr)[inds.y]) * (uvw).y + unpackUvec((arr)[inds.z]) * (uvw).z)
#define getInterpolatedArray(arr, inds, uvw)      ((arr)[inds.x] * (uvw).x + (arr)[inds.y] * (uvw).y + (arr)[inds.z] * (uvw).z)
#define getInterpolated(a, b, c, uvw)             ((a) * (uvw).x + (b) * (uvw).y + (c) * (uvw).z)

// Possibly sets a primitive ID based on the surface visualization mode.
// Renderers that don't have particular inputs should set them to 0.
// At the moment, SURFACEVIS_ANISOTROPY must be handled separately.
void chooseSurfaceVisOutput(inout int primitiveID,
  uint baseTriangleIndex,
  uint localMicrotriangleIndex,
  uint format,
  uint lodBias,
  float valueRange,
  uint baseSubdiv)
{
#if SURFACEVIS == SURFACEVIS_BASETRI
  primitiveID = int(baseTriangleIndex);
#elif SURFACEVIS == SURFACEVIS_MICROTRI
  primitiveID = int(localMicrotriangleIndex | (baseTriangleIndex << 10));
#elif SURFACEVIS == SURFACEVIS_LOCALTRI
  primitiveID = int(localMicrotriangleIndex);
#elif SURFACEVIS == SURFACEVIS_FORMAT
  primitiveID = int(format);
#elif SURFACEVIS == SURFACEVIS_LODBIAS
  primitiveID = int(lodBias);
#elif SURFACEVIS == SURFACEVIS_VALUERANGE
  primitiveID = floatBitsToInt(valueRange);
#elif SURFACEVIS == SURFACEVIS_BASESUBDIV
  primitiveID = int(baseSubdiv);
#endif
}

uint murmurHash(uint idx)
{
    uint m = 0x5bd1e995;
    uint r = 24;

    uint h = 64684;
    uint k = idx;

    k *= m;
    k ^= (k >> r);
    k *= m;
    h *= m;
    h ^= k;

    return h;
}

float getSignedTriangleVolume(vec3 a, vec3 b, vec3 c)
{
  // http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf
  return (1.0f/6.0f)*(
    -(c.x*b.y*a.z) + (b.x*c.y*a.z) + (c.x*a.y*b.z) 
    -(a.x*c.y*b.z) - (b.x*a.y*c.z) + (a.x*b.y*c.z));
}

float getBoundsVolume(vec3 min0, vec3 min1, vec3 min2, vec3 max0, vec3 max1, vec3 max2)
{
  return (
      // bottom (reverse winding)
      getSignedTriangleVolume(min2, min1, min0) +
      
      // sides
      getSignedTriangleVolume(min0, min1, max0) +
      getSignedTriangleVolume(max0, min1, max1) +
      
      getSignedTriangleVolume(min2, min0, max2) +
      getSignedTriangleVolume(max2, min0, max0) +
      
      getSignedTriangleVolume(min1, min2, max1) +
      getSignedTriangleVolume(max1, min2, max2) +
      
      // top
      getSignedTriangleVolume(max0, max1, max2));
}

float getValueRange(float triMin, float triMax)
{
  return (triMax - triMin);
}

#include "octant_encoding.h"

// clang-format on
#endif

#endif
