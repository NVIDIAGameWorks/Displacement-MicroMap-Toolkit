/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// this file is included by C++ and GLSL

#ifndef _COMMON_BARYMAP_H_
#define _COMMON_BARYMAP_H_

//////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
namespace microdisp {
using namespace nvmath;
#endif


////////////////////////////////////////////////////////////////////
// BaryMapData

// meshlet configuration for pregenerated meshlets for each level
#define MAX_BARYMAP_VERTICES 64
#define MAX_BARYMAP_PRIMITIVES 64

// Limit the current app to a sane upper bound.
// Raising this can have consequences in number of bits required for offsets
// in task shader output in dynamic lod scenario.
// But also the data structure required for precalculated uv coords,
// as well as other upper bounds.
#define MAX_BARYMAP_LEVELS 8
// 3 edge bits, up to 8 permutations
#define MAX_BARYMAP_TOPOS 8

////////////////////////////////////////////////////////////////////
// BaryMapMeshlet
//
// pre-computed meshlets, we render a single base-triangle
// as sequence of these.

struct BaryMapMeshlet
{
  uint16_t numVertices;
  uint16_t numPrimitives;
  uint16_t offsetPrims;
  uint16_t offsetVertices;
};
#ifndef __cplusplus
layout(buffer_reference, buffer_reference_align = 4, scalar) restrict readonly buffer BaryMapMeshlet_in
{
  BaryMapMeshlet d[];
};
#endif

////////////////////////////////////////////////////////////////////
// BaryMapLevel
//
// pre-computed data for each subdivision level

struct BaryMapLevel
{
  BUFFER_REF(uints_in) coords;                   // micro vertex barycentric coords
  BUFFER_REF(BaryMapMeshlet_in) meshletHeaders;  // micro triangles meshlet headers
  BUFFER_REF(uints_in) meshletData;              // micro triangles meshlet data
  uint meshletCount;
  uint triangleCount;
};
#ifndef __cplusplus
layout(buffer_reference, buffer_reference_align = 16, scalar) restrict readonly buffer BaryMapLevel_in
{
  BaryMapLevel d[];
};
#endif

////////////////////////////////////////////////////////////////////
// BaryMapData
//
// contains multiple BaryMapLevels and accessors for uniform or
// divergent access.

struct BaryMapData
{
  // barycentric lookup maps
  BaryMapLevel levelsUni[MAX_BARYMAP_LEVELS * MAX_BARYMAP_TOPOS];  // subgroup-uniform access
  BUFFER_REF(BaryMapLevel_in) levels;                              // divergent access (points to same data as above)
};


#ifdef __cplusplus
}
#endif
#endif
