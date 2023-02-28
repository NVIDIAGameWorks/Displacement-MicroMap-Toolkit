/*
 * Copyright (c) 2014-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef __cplusplus
#include "octant_encoding.h"
#endif

#define GENERATE_IMPORTANCE_BLOCK_SIZE 256

// clang-format off
#ifdef __cplusplus // Descriptor binding helper for C++ and GLSL
 #define START_BINDING(a) enum a {
 #define END_BINDING() }
 #define INLINE inline
#else
#define START_BINDING(a)  const uint
 #define END_BINDING()
 #define INLINE
#endif

#define START_ENUM(a) START_BINDING(a)
#define END_ENUM() END_BINDING()
// clang-format on

struct GenerateImportanceConstants
{
  mat4  worldToObject;
  mat4  objectToWorld;
  uint  vertexCount;
  float curvatureMaxDist;
  float curvaturePower;
  uint  hasImportanceMap;

  uint texCoordCount;
  uint texCoordIndex;
};
// clang-format off
START_BINDING(GenerateImportanceBindings)

    // fp32 x 3 + octant normal (snorm16x2)
    eModifiedVertexPositionNormalBuffer = 0,

    // 2 x octant normal (snorm16x2)
    eModifiedVertexTangentSpaceBuffer = 1,

    // n x fp32 x 2
    eModifiedVertexTexcoordBuffer = 2,

    // fp16 x 4
    eModifiedVertexDirectionsBuffer = 3,

    // 1 x fp16
    // used by remesher
    eModifiedVertexImportanceBuffer = 5,
    eMeshAccel = 6,
    eInputImportanceMap = 7
END_BINDING();
// clang-format on
