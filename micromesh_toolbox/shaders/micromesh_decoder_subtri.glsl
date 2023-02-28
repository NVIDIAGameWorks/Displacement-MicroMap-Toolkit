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
// embed the code from the toolkit.

//////////////////////////////////////////////////////////////
// SubgroupMicromeshDecoder
//
// This decoder makes use of shuffle to gather the displacement values of the two
// vertices whose edge was split to create a new vertex. That new
// vertex computes its displacement using a signed correction value that
// is applied on the average displacement of the edge vertices.
//
// The decode process iteratively adds more vertices that are computed
// for each subdivision level, relying on the results of the
// previous level.

struct SubgroupMicromeshDecoder
{
  MicroDecoder       dec;
  MicroDecoderConfig cfg;

  MicromeshSubTri microSubTri;

  // offset into threads for shuffle access
  // influenced by packID when multiple decoders live in same
  // workgroup
  uint threadOffset;

  // index buffer offset based on lod and topology
  uint primOffset;

  // offset into vertices array
  // influenced by partID state (if micromesh split into multiple meshlets)
  uint vertexOffset;

  // to account for scaling uv when lod mismatches fetched microVertex UVs
  int uvMultiplier;

  // from when vertices are updated in the current decode iteration
  // previous vertex thtreads required active for shuffle access to their results
  uint decodeVertStart;

  // per-thread micro vertex info used during decoding
  // vtx info is accurate for first 2 levels, 3rd level is partial, and needs extra warp iteration
  // to load vertices 32...44
  MicromeshSTriVertex localVtx;
  // encoded correction value (signed delta or flat)
  // in flat case this matches displacement value
  int localCorrection;
  // final displacement value, used to apply delta to, fetched via shuffle
  int displacement;
};


#include "micromesh_decoder_api.glsl"

// MicromeshSTriVertex
ivec2 microvertex_getUV(MicromeshSTriVertex vtx)
{
  return ivec2(bitfieldExtract(vtx.packed, MICRO_STRI_VTX_U_SHIFT, MICRO_STRI_VTX_UV_WIDTH),
               bitfieldExtract(vtx.packed, MICRO_STRI_VTX_V_SHIFT, MICRO_STRI_VTX_UV_WIDTH));
}

uvec2 microvertex_getAB(MicromeshSTriVertex vtx)
{
  return uvec2(bitfieldExtract(vtx.packed, MICRO_STRI_VTX_A_SHIFT, MICRO_STRI_VTX_AB_WIDTH),
               bitfieldExtract(vtx.packed, MICRO_STRI_VTX_B_SHIFT, MICRO_STRI_VTX_AB_WIDTH));
}

uint microvertex_getLevel(MicromeshSTriVertex vtx)
{
  return bitfieldExtract(vtx.packed, MICRO_STRI_VTX_LVL_SHIFT, MICRO_STRI_VTX_LVL_WIDTH);
}

uint microvertex_getIndex(MicromeshSTriVertex vtx)
{
  return vtx.packed >> MICRO_STRI_VTX_IDX_SHIFT;
}

int microdec_readCorrectionBits(inout MicroDecoder dec, MicromeshSTriVertex vtx, bool isUnsigned)
{
  uint lvl      = microvertex_getLevel(vtx);
  uint idx      = microvertex_getIndex(vtx);
  uint corrBits = microdec_getNumCorrBits(dec, lvl);
  uint bitPos   = microdec_getStartPos(dec, lvl) + idx * corrBits;
  uint raw      = microdata_readDataBits(dec.dataOffset, bitPos, corrBits);
  return (isUnsigned || lvl == 0) ? int(raw) : microdata_convertSigned(raw, corrBits);
}

int smicrodec_subgroupGetVertexDisplacement(inout SubgroupMicromeshDecoder sdec, uint vert, MicromeshSTriVertex vtx, ivec2 uv, int correction)
{
  // safe to use shuffle for accessing first vertices

  uvec2 abIdx = microvertex_getAB(vtx);
  int   a     = subgroupShuffle(sdec.displacement, abIdx.x + sdec.threadOffset);
  int   b     = subgroupShuffle(sdec.displacement, abIdx.y + sdec.threadOffset);

  // if this vertex was already decoded (< decodeVertStart) then use existing displacement
  // otherwise compute new displacement value using averages and correction.

  int disp = vert < sdec.decodeVertStart ?
                 sdec.displacement :
                 (microdec_compute(sdec.dec, a, b,
                                   microdec_decodePredictionCorrection(sdec.dec, microdec_getVertexType(sdec.dec, uv), correction)));

  return disp;
}

bool splitidx_isFlippped(uint splitIdx)
{
  return splitIdx == 1 || splitIdx == 3;
}

bool partmicro_isFlipped(uint partID)
{
  return splitidx_isFlippped(partID & 3) BOOL_XOR splitidx_isFlippped((partID / 4) & 3);
}

uint splitidx_getTopo(uint splitIdx, uint topo, inout bool reversed)
{
  // Topo contains 3 bits (value 1,2,4) that are set for each base edge with
  // decimated resolution.
  // Because meshlet rendering causes multiple meshlets per micromesh
  // we need to figure out which edges are relevant, and if they change location.
  //
  //               V
  //   edge       / \      edge
  //   value     / 3 \     value
  //      4     x_____x    2
  //           / \ 1 / \ 
  //          / 0 \ / 2 \ 
  //         W ___ x ___ U
  //
  //         edge value 1
  //

  // split 0 keeps 1 and 4 and same order
  if(splitIdx == 0)
    return topo & 5;
  // split 1 looses all, purely internal
  else if(splitIdx == 1)
    return 0;
  // split 2 keeps 1 and 2 and same order
  else if(splitIdx == 2)
    return topo & 3;
  // split 3 keeps 2 and 4 reversed
  else /*if (splitIdx == 3)*/
  {
    reversed = !reversed;
    return ((topo & 2) << 1) | ((topo & 4) >> 1);
  }
}

uint partmicro_getTopo(uint partID, uint topo, uint subdivLevel, inout bool reversed)
{
  // subdivLevel 5 uses two hierarchical steps
  // subdivLevel 4 one
  // all others leave value as is
  topo = subdivLevel == 5 ? splitidx_getTopo((partID / 4) & 3, topo, reversed) : topo;
  return subdivLevel >= 4 ? splitidx_getTopo(partID & 3, topo, reversed) : topo;
}

uint micromesh_getTopoPrimOffset(in MicromeshSubTri subTri, uint targetSubdiv, uint partID, uint partOffset)
{
  // This function adjust the index buffer permutation of the meshlet based
  // on edge decimate flags (topo bits) to create watertight triangle skirts.

  uint topo        = micromesh_getBaseTopo(subTri);
  uint baseSubdiv  = micromesh_getBaseSubdiv(subTri);
  uint microSubdiv = micromesh_getSubdiv(subTri);
  uint subTopo     = partOffset;

  if(targetSubdiv < microSubdiv && targetSubdiv < 3)
  {
    topo = 0;
  }

  if(baseSubdiv > microSubdiv && topo != 0)
  {
    // adjust topo situation based on subTri itself
    // singel base triangle can be composed of many micromeshes / sub triangles
    bool isReversedU = false;

    // adjust topo bits based on partID (due to meshlet splitting)
    topo = partmicro_getTopo(partID, topo, microSubdiv, isReversedU);

    // must pick different topology variant based on which corner we are in
    // we cannot be in the inner split quadrant (1)

    isReversedU = isReversedU BOOL_XOR micromesh_isReversedU(subTri);

    ivec2 baseUV    = micromesh_getBaseUV(subTri, ivec2(0, 0));
    int   halfCoord = (1 << (baseSubdiv - 1));

    //
    //               V
    //   topo       / \      topo
    //   value     / 3 \     value
    //      4     x_____x    2
    //           / \ 1 / \ 
    //          / 0 \ / 2 \ 
    //         W ___ x ___ U
    //
    //         topo value 1
    //

    // if the current meshlet runs reversed, swap U/W
    uint uCorner = isReversedU ? 0 : 2;

    if(topo == 0)
      subTopo = 0;  // could become 0 due to partmicro_getTopo adjustments
    else if(baseUV.y >= halfCoord)
      subTopo = 3 + 1;  // V corner
    else if(baseUV.x >= halfCoord)
      subTopo = uCorner + 1;  // U corner
    else
      subTopo = 2 - uCorner + 1;  // W corner

    if(subTopo == 4)
    {
      // top quadrand flips top edges
      topo = ((topo & 2) << 1) | ((topo & 4) >> 1);
    }
  }

  return subTopo * MICRO_PART_MAX_PRIMITIVES + topo * MICRO_MESHLET_PRIMS;
}

#if MICRO_DECODER == MICRO_DECODER_SUBTRI_BASE_SHUFFLE

MicromeshSubTri micromesh_getSubTri(in MicromeshBaseTri baseTri, uint baseTriangleIdx, uint subTriangleIndex)
{
  MicromeshSubTri subTri;

  subTri.baseTriangleIdx = baseTriangleIdx;
  subTri.packedBits      = baseTri.packedBits;
  subTri.dataOffset      = baseTri.dataOffset + subTriangleIndex * micromesh_getDataSize(baseTri);

  uint level    = micromesh_getBaseSubdiv(subTri);
  uint levelFmt = micromesh_getSubdiv(subTri);
  bool flipped  = (bitCount(bird_extractEvenBits(subTriangleIndex)) & 1) != 0;

  uvec3 iuvw;
  iuvw    = bird_getUVW(subTriangleIndex);  // Logic
  uint iu = iuvw.x;
  uint iv = iuvw.y;
  uint iw = iuvw.z;

  uint levelBird = level - levelFmt + 1;
  uint edge      = 1 << (levelBird - 1);

  // we need to only look at "level" bits
  iu = iu & ((1u << levelBird) - 1);
  iv = iv & ((1u << levelBird) - 1);
  iw = iw & ((1u << levelBird) - 1);

  bool upright = ((iu & 1) ^ (iv & 1) ^ (iw & 1)) != 0;

  uint subSegments = subdiv_getNumSegments(micromesh_getSubdiv(baseTri));

  uint u = flipped ? 1 : 0;

  if(upright)
  {
    iu                = iu + u;
    subTri.baseOffset = u16vec2(uint16_t(iu * subSegments), uint16_t(iv * subSegments));

    iw = edge - iu - iv;

    // only upright triangles can be affected by edge decimate flags

    uint topoBits = micromesh_getBaseTopo(subTri);
    uint topoOrig = topoBits;
    uint topoMask = 0;

    //     v
    //   4/ \2
    //   w___u
    //     1

    //if (level <= 3)                   topoMask |= 7;
    if(iv == 0)
      topoMask |= 1;
    if(iu == 0 || (iu - u) == 0)
      topoMask |= flipped ? 2 : 4;
    if(iw == 0 || (iw - 1 - u) == 0)
      topoMask |= flipped ? 4 : 2;

    if(flipped)
    {
      // swap second and third bit
      topoBits = (topoBits & 1) | ((topoBits << 1) & 4) | ((topoBits >> 1) & 2);
    }

    topoBits = topoBits & topoMask;

    // assign new topo
    subTri.packedBits = uint((micromesh_getBaseSubdiv(baseTri) << MICRO_SUB_LVL_SHIFT) | (topoBits << MICRO_SUB_TOPO_SHIFT)
                             | (micromesh_getFormat(baseTri) << MICRO_SUB_FMT_SHIFT) | MICRO_SUB_SIGN_V_POSITIVE
                             | (flipped ? 0 : MICRO_SUB_SIGN_U_POSITIVE) | (flipped ? MICRO_SUB_FLIP : 0));
    // if flipped -u, else +u, always +v
  }
  else
  {
    iu = iu + 1;
    iv = iv + 1;

    subTri.baseOffset = u16vec2(uint16_t((iu - u) * subSegments), uint16_t(iv * subSegments));

    // remove topo
    subTri.packedBits = uint((micromesh_getBaseSubdiv(baseTri) << MICRO_BASE_LVL_SHIFT)
                             | (micromesh_getFormat(baseTri) << MICRO_SUB_FMT_SHIFT)
                             | (flipped ? MICRO_SUB_SIGN_U_POSITIVE : 0) | (flipped ? MICRO_SUB_FLIP : 0));
    // if flipped +u, else -u, always -v
  }

  return subTri;
}

#endif

//////////////////////////////////////
// public api

// the principle operations of the decoding process is illustrated in the rasterization pdf document

uint smicrodec_getThreadCount(uint partSubdiv)
{
#if SUBGROUP_SIZE == 32
  return partSubdiv == 3 ? 32u : (1u << (1 + partSubdiv));
#elif SUBGROUP_SIZE == 64
  return partSubdiv == 3 ? 64u : (2u << (1 + partSubdiv));
#else
#error "unspported SUBGROUP_SIZE"
#endif
}

void smicrodec_subgroupInit(inout SubgroupMicromeshDecoder sdec,
                            MicroDecoderConfig             cfg,
#if MICRO_DECODER == MICRO_DECODER_SUBTRI_BASE_SHUFFLE
                            MicromeshBaseTri microBaseTri,
#elif MICRO_DECODER == MICRO_DECODER_SUBTRI_SHUFFLE
                            MicromeshSubTri microSubTri,
#endif
                            uint firstMicro,
                            uint firstData,
                            uint firstMipData)
{
#if MICRO_DECODER == MICRO_DECODER_SUBTRI_BASE_SHUFFLE
  // Init microSubTri
  MicromeshSubTri microSubTri = micromesh_getSubTri(microBaseTri, cfg.microID, cfg.subTriangleIndex);
#endif

  uint packThreadID = cfg.packThreadID;
  uint packThreads  = cfg.packThreads;
  uint packID       = cfg.packID;
  uint microID      = cfg.microID;
  uint partID       = cfg.partID;
  uint subdivTarget = cfg.targetSubdiv;

  sdec.microSubTri = microSubTri;
  sdec.cfg         = cfg;

  microdec_init(sdec.dec, micromesh_getFormat(microSubTri), firstData + microSubTri.dataOffset, 0);
  microdec_setCurrentSubdivisionLevel(sdec.dec, 0, 0);

  uint subdivEncoded = microdec_getFormatSubdiv(sdec.dec);

  // subdivEncoded   : the subdivision level for the current micromesh encoding
  // subdivTarget    : the target subdivision level for the displaced patch
  //                   subdivDescend + subdivMerge + subdivVertex
  // the hierarchy is as follows:
  //                  0: always base micro mesh displacements
  // subdivDescend    x: number of levels to get to current sub micro mesh displacements (caused by meshlet splitting)
  // subdivMerge      x: number of levels resolved by computing displacement values through merging
  //                     maximum is 2
  // subdivVertex     1: last level is computed at vertex time based on previous
  //

  uint subdivLocal  = MICRO_PART_MAX_SUBDIV;  // maximum levels we can do within workgroup
  uint subdivVertex = 1;                      // last level is done here, reads from sdec.displacement

  uint subdivDescend = uint(max(0, int(subdivTarget) - int(subdivLocal)));
  uint subdivMerge   = uint(max(0, int(subdivTarget) - int(subdivDescend) - 1));
  sdec.uvMultiplier  = 1 << (subdivEncoded - max(subdivTarget, 3));

  // offset into meshlet configurations
  // influenced by partID state (if micromesh split into multiple meshlets)
  // there are 3 partMicro configrations
  // 1 meshlet  for level 3 or less
  // 4 meshlets for level 4
  // 5 meshlets for level 5
  uint partOffset = subdiv_getPartOffset(subdivTarget, partID);

  sdec.threadOffset = packID * packThreads;
  sdec.vertexOffset = partOffset * MICRO_PART_VERTICES_STRIDE;

  bool isFlatOrLevelZero = microdec_isFlat(sdec.dec) || subdivTarget == 0;

  // always read initial values
  sdec.localVtx        = microdata_loadMicromeshVertex(packThreadID + sdec.vertexOffset);
  sdec.localCorrection = microdec_readCorrectionBits(sdec.dec, sdec.localVtx, isFlatOrLevelZero);

  sdec.primOffset = MICRO_MESHLET_LOD_PRIMS * cfg.partSubdiv;
#if USE_NON_UNIFORM_SUBDIV
  sdec.primOffset += micromesh_getTopoPrimOffset(microSubTri, subdivTarget, partID, partOffset);
#else
  // sdec.primOffset += partOffset * MICRO_PART_MAX_PRIMITIVES;
#endif

  // debug
  //if (gl_SubgroupInvocationID == 0)
  //{
  //  stats.debugA[gl_WorkGroupID.x] = sdec.partOffset;
  //  stats.debugB[gl_WorkGroupID.x] = sdec.vertexOffset;
  //  stats.debugC[gl_WorkGroupID.x] = firstData + microSubTri.dataOffset;
  //}

#if MICRO_SUPPORTED_FORMAT_BITS != (1 << MICRO_FORMAT_64T_512B)
  if(isFlatOrLevelZero)
  {
    return;
  }

  // setup local anchor displacements
  if(packThreadID < 3)
  {
    // initial lossless micromesh anchors
    sdec.displacement = int(microdata_readDataBits(sdec.dec.dataOffset, packThreadID * MICRO_UNORM_BITS, MICRO_UNORM_BITS));

    // the micromesh was split into multiple meshlets
    // each meshlet now needs to find its local anchors, which are within the micromesh
    // we need to descend the hierarchy up to 2 times (subdiv 5 and 4 are lowered to 3)
    if(subdivDescend > 0)
    {

      int                  uvMul       = 1 << (subdivEncoded - subdivTarget);
      MicromeshSTriDescend descendInfo = microdata_loadMicromeshDescend((subdivDescend - 1) * 4 + partID);
      // Subdivide down to the target micro-triangle
      for(uint i = 0; i < subdivDescend; i++)
      {
        microdec_setCurrentSubdivisionLevel(sdec.dec, i + 1, 0);

        MicromeshSTriVertex mvtx  = descendInfo.vertices[packThreadID + 3 * i];
        ivec2               uv    = microvertex_getUV(mvtx) * uvMul;
        uvec2               abIdx = microvertex_getAB(mvtx);

        int a = subgroupShuffle(sdec.displacement, abIdx.x + sdec.threadOffset);
        int b = subgroupShuffle(sdec.displacement, abIdx.y + sdec.threadOffset);

        int correction = microdec_readCorrectionBits(sdec.dec, mvtx, false);

        sdec.displacement =
            abIdx.x == abIdx.y ?
                a :
                microdec_compute(sdec.dec, a, b,
                                 microdec_decodePredictionCorrection(sdec.dec, microdec_getVertexType(sdec.dec, uv), correction));
      }
    }
  }

  // add merge levels
  // subd  | # vertices |
  // subd 0| 3          | anchor/ post descend level
  // subd 1| 6          | mergeLevel 0
  // subd 2| 15         | mergeLevel 1
  // subd 3| 45         | vtx level

  sdec.decodeVertStart = 3;
  for(uint32_t mergeLevel = 0; mergeLevel < subdivMerge; mergeLevel++)
  {
    uint subdiv          = mergeLevel + 1;
    uint subdivDec       = subdivDescend + subdiv;
    uint numVertsPerEdge = subdiv_getNumVertsPerEdge(subdiv);
    uint numVerts        = 6 + mergeLevel * 9;

    microdec_setCurrentSubdivisionLevel(sdec.dec, subdivDec, 0);

    if(packThreadID < numVerts)
    {
      ivec2 localUV = microvertex_getUV(sdec.localVtx) * sdec.uvMultiplier;
      sdec.displacement = smicrodec_subgroupGetVertexDisplacement(sdec, packThreadID, sdec.localVtx, localUV, sdec.localCorrection);
    }

    sdec.decodeVertStart = numVerts;
  }
  microdec_setCurrentSubdivisionLevel(sdec.dec, subdivDescend + subdivMerge + 1, 0);
#endif
}

uint smicrodec_getIterationCount()
{
#if SUBGROUP_SIZE == 32
  return 2;
#elif SUBGROUP_SIZE == 64
  return 1;
#else
#error "unspported SUBGROUP_SIZE"
#endif
}

uint smicrodec_getPackID(inout SubgroupMicromeshDecoder sdec)
{
  return sdec.cfg.packID;
}

uint smicrodec_getDataIndex(inout SubgroupMicromeshDecoder sdec, uint iterationIndex)
{
  return sdec.cfg.packThreadID + iterationIndex * sdec.cfg.packThreads;
}

uint smicrodec_getNumTriangles(inout SubgroupMicromeshDecoder sdec)
{
  return subdiv_getNumTriangles(sdec.cfg.partSubdiv);
}
uint smicrodec_getNumVertices(inout SubgroupMicromeshDecoder sdec)
{
  return subdiv_getNumVerts(sdec.cfg.partSubdiv);
}
uint smicrodec_getMeshTriangle(inout SubgroupMicromeshDecoder sdec)
{
  return sdec.microSubTri.baseTriangleIdx;
}
uint smicrodec_getBaseSubdiv(inout SubgroupMicromeshDecoder sdec)
{
  return micromesh_getBaseSubdiv(sdec.microSubTri);
}
uint smicrodec_getFormatIdx(inout SubgroupMicromeshDecoder sdec)
{
  return microdec_getFormatIdx(sdec.dec);
}
uint smicrodec_getMicroSubdiv(inout SubgroupMicromeshDecoder sdec)
{
  return micromesh_getSubdiv(sdec.microSubTri);
}

MicroDecodedVertex smicrodec_subgroupGetVertex(inout SubgroupMicromeshDecoder sdec, uint iterationIndex)
{
  uint numVerts = subdiv_getNumVerts(sdec.cfg.partSubdiv);
  uint packID   = sdec.cfg.packID;

  uint vert      = smicrodec_getDataIndex(sdec, iterationIndex);
  bool vertValid = vert < numVerts && sdec.cfg.valid;
  bool isFlat    = microdec_isFlat(sdec.dec) || sdec.cfg.targetSubdiv == 0;

#if SUBGROUP_SIZE == 32
#if 0
    MicromeshSTriVertex localVtx = iterationIndex > 0 ? microdata_loadMicromeshVertex(vert + sdec.vertexOffset) : sdec.localVtx;
#else
  // compiler bug WAR
  // https://github.com/KhronosGroup/glslang/issues/2843
  MicromeshSTriVertex localVtx;
  if(iterationIndex > 0)
  {
    localVtx = microdata_loadMicromeshVertex(vert + sdec.vertexOffset);
  }
  else
  {
    localVtx = sdec.localVtx;
  }
#endif
  int localCorrection = iterationIndex > 0 ? microdec_readCorrectionBits(sdec.dec, localVtx, isFlat) : sdec.localCorrection;
#else
  MicromeshSTriVertex localVtx = sdec.localVtx;
  int localCorrection = sdec.localCorrection;
#endif
  ivec2 localUV = microvertex_getUV(localVtx) * sdec.uvMultiplier;

  MicroDecodedVertex outVertex;

  if(isFlat)
  {
    outVertex.displacement = localCorrection;
  }
  else
  {
    outVertex.displacement = smicrodec_subgroupGetVertexDisplacement(sdec, vert, localVtx, localUV, localCorrection);
  }

  outVertex.valid      = vertValid;
  outVertex.localIndex = vert;
  outVertex.outIndex   = vert + packID * numVerts;

  if(vertValid)
  {
    // adjust uv if microSubdiv is actually smaller than 3
    localUV = localUV >> (3 - min(3, micromesh_getSubdiv(sdec.microSubTri)));

    // Compute barycentrics
    outVertex.uv      = micromesh_getBaseUV(sdec.microSubTri, localUV);
    outVertex.bary.yz = vec2(outVertex.uv) / float(subdiv_getNumSegments(micromesh_getBaseSubdiv(sdec.microSubTri)));
    outVertex.bary.x  = 1.0f - outVertex.bary.y - outVertex.bary.z;
  }

  return outVertex;
}

MicroDecodedTriangle smicrodec_getTriangle(inout SubgroupMicromeshDecoder sdec, uint iterationIndex)
{
  uint numVerts     = subdiv_getNumVerts(sdec.cfg.partSubdiv);
  uint numTriangles = subdiv_getNumTriangles(sdec.cfg.partSubdiv);
  uint packID       = sdec.cfg.packID;

  uint prim      = smicrodec_getDataIndex(sdec, iterationIndex);
  bool primValid = prim < numTriangles && sdec.cfg.valid;

  MicroDecodedTriangle outPrim;
  outPrim.valid      = primValid;
  outPrim.localIndex = prim;
  outPrim.outIndex   = prim + packID * numTriangles;

  if(primValid)
  {
    uint  localTri = microdata_loadTriangleIndices(prim + sdec.primOffset);
    bool  isFlipped;
    uvec3 indices;
    indices.x = (localTri >> 0) & 0xFF;
    indices.y = (localTri >> 8) & 0xFF;
    indices.z = (localTri >> 16) & 0xFF;

    isFlipped = partmicro_isFlipped(sdec.cfg.partID);
    isFlipped = isFlipped BOOL_XOR micromesh_isFlipped(sdec.microSubTri);

    if(isFlipped)
      indices = indices.xzy;

    outPrim.indices = indices + packID * numVerts;
  }

  return outPrim;
}
