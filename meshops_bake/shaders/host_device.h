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

#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#include "vertex_types.h"

#define TRIANGLE_BLOCK_SIZE 256  // Grid size used by compute shaders

#define MAX_RESAMPLE_TEXTURES 8

// clang-format off
#ifdef __cplusplus // GLSL Type
using vec2 = nvmath::vec2f;
using vec3 = nvmath::vec3f;
using vec4 = nvmath::vec4f;
using mat4 = nvmath::mat4f;
using uint = uint32_t;
#endif

#ifdef __cplusplus // Common enumeration helper for C++ and GLSL
 #define START_ENUM(a) enum class a {
 #define END_ENUM(a) }; \
 inline constexpr uint32_t operator+(const a& v){return static_cast<uint32_t>(v);} // Implements cast-to-base-type using unary +
#else
 #define START_ENUM(a)  const uint
 #define END_ENUM(a) 
#endif

#define BITFIELD_MASK(bits) (~(0xffffffffU << bits))
#define BITFIELD_SET(bitfield, offset, bits, value) \
  ((bitfield & (~(BITFIELD_MASK(uint32_t(bits)) << uint32_t(offset)))) | ((value & BITFIELD_MASK(bits)) << uint32_t(offset)))
#define BITFIELD_GET(bitfield, offset, bits) \
  ((bitfield >> uint32_t(offset)) & BITFIELD_MASK(uint32_t(bits)))

START_ENUM(SceneBindings)
  eFrameInfo    = 0,
  eSceneDesc    = 1,
  eDistances    = 2,
  eBaryCoords   = 3,
  eTlas         = 4,
  eTexturesIn   = 5,
  eTexturesOut  = 6,
  eTexturesDist = 7
END_ENUM(SceneBindings);

// Must match TextureType in meshops_operations.h.
#ifndef __cplusplus
const uint
  eGeneric       = 0, 
  eNormalMap     = 1, 
  eQuaternionMap = 2, 
  eOffsetMap     = 3,
  eHeightMap     = 4;
#endif

// clang-format on

struct BakerMeshInfo
{
  uint64_t vertexAddress;
  uint64_t indexAddress;
  uint64_t vertexDirectionBoundsAddress;
  uint64_t vertexDirectionBoundsOrigAddress;
  uint     padding_;
  uint     numTriangles;

  // Low+high mesh primitives are baked in pairs. When tracing from the low res
  // mesh, we need to adjust the max trace distance to account for any
  // additional heightmap displacement.
  float maxDisplacementWs;
};

#define BAKER_MAX_SUBDIV_LEVEL 5

// Max. subdiv level + 1. 6 means handle levels 0 to 5 inclusive.
#define BAKER_NUM_SUBDIV_LEVEL_MAPS (BAKER_MAX_SUBDIV_LEVEL + 1)

struct SceneDescription
{
  uint64_t baseMeshAddress;
  uint64_t referenceMeshAddress;
  uint64_t distancesAddress;
  uint64_t trianglesAddress;
  uint64_t triangleMinMaxsAddress;
  uint64_t baryCoordsAddress[BAKER_NUM_SUBDIV_LEVEL_MAPS];
};

struct FrameInfo
{
  mat4 view;
  mat4 proj;
  mat4 viewInv;
  mat4 projInv;
};

#define TEXINFO_TYPE_OFFSET (0x0)
#define TEXINFO_TYPE_BITS (0x3)
#define TEXINFO_INDEX_OFFSET (0x3)
#define TEXINFO_INDEX_BITS (0x3)

struct ResampleTextureInfo
{
#ifdef __cplusplus
  void setTextureType(uint type)
  {
    assert(type <= BITFIELD_MASK(uint32_t(TEXINFO_TYPE_BITS)));
    bits = BITFIELD_SET(bits, TEXINFO_TYPE_OFFSET, TEXINFO_TYPE_BITS, type);
  }
  void setInputIndex(uint index)
  {
    assert(index <= BITFIELD_MASK(uint32_t(TEXINFO_TYPE_BITS)));
    bits = BITFIELD_SET(bits, TEXINFO_INDEX_OFFSET, TEXINFO_INDEX_BITS, index);
  }
#if 1
  uint bits = 0;
#else
  union
  {
    struct
    {
      uint textureType : 3;  // Stores meshops::TextureType
      uint inputIndex : 3;   // Stores up to MAX_RESAMPLE_TEXTURES
    };
    uint bits = 0;
  };
#endif
#else
  uint bits;
#endif
};

#ifdef __cplusplus
static_assert(sizeof(ResampleTextureInfo) == 4, "Minimum push constant size on NV GPUs exceeded");
#endif

struct BakerPushConstants
{
  mat4 objectToWorld;
  mat4 worldToObject;
  int  padding1_;
  int  padding2_;

  // Tracing rays is based on direction vectors.
  // - Bidirectional                                   (!uniDirectional)
  //       |<  (hit)   o---(hit)-->|
  // - Unidirectional                                  ( uniDirectional)
  //           (miss)  o---(hit)-->|
  // - Max distance matches direction vector magnitude (!replaceDirectionLength)
  //                   o---(hit)-->|  (miss)
  // - Max distance overridden                         ( replaceDirectionLength, maxDistance=...)
  //                   o---(hit)--> (hit)      |
  // - Max distance increased for heightmaps           ( highMeshHasDisplacement)
  //   In addition to other options, min/max distance is extended by BakerMeshInfo::maxDisplacementWs
  //               |<  o---(hit)--> (hit)         |
  float maxDistance;              // Baking ray distances in world space
  uint  replaceDirectionLength;   // bool, use maxDistance instead of direction vector magnitude
  uint  highMeshHasDisplacement;  // bool, conservatively extend ray distance by BakerMeshInfo::maxDisplacementWs
  uint  uniDirectional;           // bool, only trace forwards, along the direction vector
  uint  hasDirectionBounds;       // bool, use per-vertex direction bounds if true, otherwise assume bias/scale of 0/1
  uint  lastBatch;                // bool, true to do disatance post-processing if this is the final batch for baking
  uint  numResampleTextures;
  uint  resampleMaxResolution;
  uint  baryTraceBatchOffset;

  // Factor applied to the maximum tracing distance, useful when the displacement bounds define a tight
  // shell around the original geometry, where floating-point approximations may create false misses.
  // A value of 1.02 typically provides satisfying results without resulting in performance/accuracy loss.
  float maxDistanceFactor;

  // Used to normalize distances when generating a heightmap during resampling.
  vec2 globalMinMax;

  ResampleTextureInfo textureInfo[MAX_RESAMPLE_TEXTURES];
  uint resampleInstanceResolutions[MAX_RESAMPLE_TEXTURES];  // The mesh is rendered once for each unique output resolution
};
#ifdef __cplusplus
static_assert(sizeof(BakerPushConstants) < 256, "Minimum push constant size on NV GPUs exceeded");
#endif

struct Triangle
{
  uint subdivLevel;
  uint valueFirst;  // offset (= valueFirst * valueByteSize) into valueData section
  uint valueCount;
  uint meshTriangle;
};


#endif  // HOST_DEVICE_H
