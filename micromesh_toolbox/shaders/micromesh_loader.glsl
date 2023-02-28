/*
* Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

// separated to ease integration with other renderers/apis

// flat renderers need to pull per-mesh data from local pointers

#ifndef mesh_microdata
#define mesh_microdata microdata
#endif

uint microdata_loadDistance(uint idx)
{
  return mesh_microdata.distances.d[idx];
}

uvec2 microdata_loadDistance2(uint idx)
{
  uvec2s_in distances64 = uvec2s_in(mesh_microdata.distances);
  return distances64.d[idx];
}

uint microdata_loadMipDistance(uint idx)
{
  return mesh_microdata.mipDistances.d[idx];
}

uint microdata_loadFormatInfo(uint formatIdx, uint decodeSubdiv)
{
  return uint(mesh_microdata.formats.d[formatIdx].width_start[decodeSubdiv]);
}

#if MICRO_DECODER == MICRO_DECODER_BASETRI_MIP_SHUFFLE
MicromeshBTriVertex
#elif MICRO_DECODER == MICRO_DECODER_MICROTRI_THREAD
MicromeshMTriVertex
#else
MicromeshSTriVertex
#endif
microdata_loadMicromeshVertex(uint idx)
{
  return mesh_microdata.vertices.d[idx];
}
#if MICRO_DECODER == MICRO_DECODER_BASETRI_MIP_SHUFFLE
MicromeshBTriDescend
#elif MICRO_DECODER == MICRO_DECODER_MICROTRI_THREAD
MicromeshMTriDescend
#else
MicromeshSTriDescend
#endif
microdata_loadMicromeshDescend(uint idx)
{
  return mesh_microdata.descendInfos.d[idx];
}

uint microdata_loadTriangleIndices(uint idx)
{
  return mesh_microdata.triangleIndices.d[idx];
}