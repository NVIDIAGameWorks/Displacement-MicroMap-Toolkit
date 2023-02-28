/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

// PLEASE READ:
// The rasterization of micromeshes, especially compressed is a bit of
// a more complex topic on its own. Therefore there will be a future
// dedicated sample that goes into details how it works
// and showcases more features, such as dynamic level-of-detail.
// We recommend to wait for this, rather than attempt to
// embed the code from the toolkit. The future sample will also
// provide more performant solutions and address compute-based
// rasterization as well.

///////////////////////////////////////////////////////////////
// public api

struct MicroDecoderConfig
{
  // baseTriID or subTriID
  // depending on decoder type
  uint microID;

  // only for base-triangle based decoder
  uint subTriangleIndex;

  // a micromesh may need multiple parts to be decoded, each part has a maximum of
  // MICRO_PART_MAX_PRIMITIVES triangles (64) and
  // MICRO_PART_MAX_VERTICES   vertices  (45)
  uint partID;
  // subdivision resolution of the part being decoded [0,3]
  uint partSubdiv;

  // target subdivision [0, microSubdiv]
  uint targetSubdiv;

  // When multiple decoder states are packed within a subgroup
  // packID is the unique identifier for each such state.
  // Each pack will use packThreads many threads, and packThreadID
  // specifies which thread is calling the function.
  // packThreads * packID + packThreadID < SUBGROUP_SIZE

  uint packID;
  uint packThreads;
  uint packThreadID;

  // decoder could be invalid
  // if subgroup contains some that are not
  // supposed to participate
  bool valid;
};
