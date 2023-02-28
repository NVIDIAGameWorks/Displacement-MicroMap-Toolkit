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

#define REMESHER_BLOCK_SIZE 256

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

#ifndef __cplusplus
#define CopyDirection uint
#endif

struct VertexCopyConstants
{
  uint itemCount;

  uint useTexCoord;
  uint useTangent;
  uint useNormal;
  uint useDirection;

  uint texcoordCount;
  uint texcoordIndex;

};

struct VertexMergeConstants
{
  uint useTexCoord;
  uint useTangent;
  uint useNormal;
  uint useDirection;
  uint texcoordCount;
  uint texcoordIndex;
  uint fitToOriginalSurface;
};


START_BINDING(VertexKernelBindings)


    // fp32 x 3 + octant normal (snorm16x2)
    eModifiedVertexPositionNormalBuffer = 0,

    // 2 x octant normal (snorm16x2)
    eModifiedVertexTangentSpaceBuffer = 1,

    // n x fp32 x 2
    eModifiedVertexTexcoordBuffer = 2,

    // fp16 x 4
    eModifiedVertexDirectionsBuffer = 3,
    // fp32 x 2
    eModifiedVertexDirectionBoundsBuffer = 4,

    // 1 x fp16
    // used by remesher
    eModifiedVertexImportanceBuffer = 5,

    // 2 x uint per-vertex
    eGpuRemeshingMeshVertexHashBuffer = 6,

    eGpuRemeshingMeshVertexMergeBuffer = 7,
    eGpuRemeshingCurrentStateBuffer = 8

END_BINDING();


#define COMPACTION_BLOCK_SIZE 1024
#define COMPACTION_ENTRIES_PER_THREAD_0 4
#define COMPACTION_ENTRIES_PER_THREAD_1 4
#define COMPACTION_ENTRIES_PER_THREAD_2 4

START_BINDING(CompactionBindings)
    eData = 0,
    eInvalidEntry = 1,
    eBlockState = 2,
    eAuxBuffer = 3,
    eGlobalCounter = 4
END_BINDING();
// clang-format on
struct CompactionConstants
{
  uint entryCount;
  uint entrySize;
  uint mode;
};
struct GlobalCounters
{
  uint validEntries;
  uint currentInvalidEntry;
  uint currentValidEntry;
};
