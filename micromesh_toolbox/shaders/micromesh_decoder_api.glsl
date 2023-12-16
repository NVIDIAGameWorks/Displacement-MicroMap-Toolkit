/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

/*
 * NOTE: This file's content is based on older versions of code in
 *       https://github.com/nvpro-samples/vk_displacement_micromaps. Please
 *       follow micromesh rasterization examples there instead.
 */

///////////////////////////////////////////////////////////////
// public api

// how many threads are used for each part, partSubdiv [0,3]
uint smicrodec_getThreadCount(uint partSubdiv);

// sets up the decoder state
// must be called in subgroup-uniform branch
//
// arguments can be divergent in subgroup, but must
// be uniform for threads with equal packID.
void smicrodec_subgroupInit(inout SubgroupMicromeshDecoder sdec,
                            MicroDecoderConfig             cfg,
#if MICRO_DECODER == MICRO_DECODER_SUBTRI_SHUFFLE
                            MicromeshSubTri microSubTri,
#else
                            MicromeshBaseTri microBaseTri,
#endif
                            uint firstMicro,
                            uint firstTriangle,
                            uint firstData,
                            uint firstMipData);

uint smicrodec_getNumTriangles(inout SubgroupMicromeshDecoder sdec);
uint smicrodec_getNumVertices(inout SubgroupMicromeshDecoder sdec);
uint smicrodec_getMeshTriangle(inout SubgroupMicromeshDecoder sdec);
uint smicrodec_getBaseSubdiv(inout SubgroupMicromeshDecoder sdec);
uint smicrodec_getMicroSubdiv(inout SubgroupMicromeshDecoder sdec);
uint smicrodec_getFormatIdx(inout SubgroupMicromeshDecoder sdec);

// the decoding process of vertices and triangles may need multiple iterations
// within the subgroup (known at compile-time)
uint smicrodec_getIterationCount();

// retrieve the triangle information for the part being decoded
struct MicroDecodedTriangle
{
  bool valid;       // this thread has work to do
  uint localIndex;  // local index relative to current part
  uint outIndex;    // output index across parts in all decoders within subgroup

  uvec3 indices;  // is adjusted for vertex output index
};

MicroDecodedTriangle smicrodec_getTriangle(inout SubgroupMicromeshDecoder sdec, uint iterationIndex);

// retrieve the vertex information for the part being decoded
// must be called in subgroup-uniform branch
struct MicroDecodedVertex
{
  bool valid;       // this thread has work to do
  uint localIndex;  // local index relative to current part
  uint outIndex;    // output index across parts in all decoders within subgroup

  ivec2 uv;            // relative to base triangle
  int   displacement;  // raw displacement
  vec3  bary;          // relative to base triangle
};

MicroDecodedVertex smicrodec_subgroupGetVertex(inout SubgroupMicromeshDecoder sdec, uint iterationIndex);
