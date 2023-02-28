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

#include "micromesh_decoder_config.glsl"
#include "micromesh_loader.glsl"

/////////////////////////////////////////////////////
// Internal details


//////////////////////////////////////////////////////////////
// data access

// numBits must be < 32
uint microdata_readDataBits(uint dataOffset, uint bitOffset, uint numBits)
{
  uint idx = dataOffset + (bitOffset / 32u);

  // assumes data was padded so out-of-bounds access is not fatal
  // actual displacement values should always be safe, as the shiftbits
  // act as terminator bound of the block.

  uint rawLo = microdata_loadDistance(idx);
  uint rawHi = microdata_loadDistance(idx + 1);

  uint64_t raw64  = packUint2x32(uvec2(rawLo, rawHi));
  uint     shift  = bitOffset % 32;
  uint     mask   = ((1u << numBits) - 1);
  uint     result = uint(raw64 >> shift) & mask;

  return result;
}

// numBits must be < 32
uint microdata_readMipBits(uint mipDataOffset, uint bitOffset, uint numBits)
{
  uint idx = mipDataOffset + (bitOffset / 32u);

  // assumes data was padded so out-of-bounds access is not fatal
  // actual displacement values should always be safe, as the shiftbits
  // act as terminator bound of the block.

  uint rawLo = microdata_loadMipDistance(idx);
  uint rawHi = microdata_loadMipDistance(idx + 1);

  uint64_t raw64  = packUint2x32(uvec2(rawLo, rawHi));
  uint     shift  = bitOffset % 32;
  uint     mask   = ((1u << numBits) - 1);
  uint     result = uint(raw64 >> shift) & mask;

  return result;
}

int microdata_convertSigned(uint result, uint numBits)
{
  // Convert from unsigned to signed by stickying the sign bit
  uint shift        = 32u - numBits;
  int  signedResult = (int(result << shift)) >> shift;
  return signedResult;
}

uint microdata_getVertexType(ivec3 wuv)
{
  uint interior = 0;
  uint edge0    = 1;
  uint edge1    = 2;
  uint edge2    = 3;

  //      V
  //     /\ 
  // e2 /__\ e1
  //   /\  /\ 
  //  /  \/  \ 
  // W __e0__ U

  if(wuv.x == 0)
    return edge1;
  else if(wuv.y == 0)
    return edge2;
  else if(wuv.z == 0)
    return edge0;
  else
    return interior;
}

//////////////////////////////////////////////////////////////
// MicroDecoder
//
// Primary class that handles fetching and adjusting the vertex displacement
// correction values from the compressed bit block.
// Different block compression formats exist and as the values
// are stored hierachical, the correction bitwidths vary
// and how they are shifted to adjust content-dependent magnitudes.

struct MicroDecoder
{
  // Fixed micromesh state:
  uint formatIdx;

  // where the encoded distance block starts (in u32 units)
  uint dataOffset;
  uint mipOffset;

  // number of segments along edge
  uint blockSegments;

  // Current decoding state:
  // Decoding is done iteratively, we decode displacements level after level
  // because subsequent levels depend on previous results.

  // Each decoding iteration potentially uses less bits for the correction delta values.
  // These shifts allow the encoded deltas to regain a larger magnitude (albeit less precision).
  // We encode four shifts depending on vertex type: vertex sits on one of the three edges, or is interior
  // (anchor is always lossless).
  uint decodeShiftBits;
};

uint microdec_getVertexType(inout MicroDecoder dec, ivec2 uv)
{
  int w = int(dec.blockSegments) - uv.x - uv.y;

  return microdata_getVertexType(ivec3(w, uv.x, uv.y));
}

bool microdec_is1024bit(inout MicroDecoder dec)
{
  return dec.formatIdx > MICRO_FORMAT_64T_512B;
}

uint microdec_getBitSize(inout MicroDecoder dec)
{
  return dec.formatIdx > MICRO_FORMAT_64T_512B ? 1024 : 512;
}

uint microdec_getBitShift(inout MicroDecoder dec)
{
  return dec.formatIdx > MICRO_FORMAT_64T_512B ? 10 : 9;
}

bool microdec_isFlat(inout MicroDecoder dec)
{

#if MICRO_UNORM_BITS == 11
#if MICRO_SUPPORTED_FORMAT_BITS == (1 << MICRO_FORMAT_64T_512B)
  return true;
#else
  return dec.formatIdx == MICRO_FORMAT_64T_512B;
#endif
#else
  return false;
#endif
}

// replacing these lookups by hardcoding the logic was slower
uint microdec_getNumCorrBits(inout MicroDecoder dec, uint decodeSubdiv)
{
  return microdata_loadFormatInfo(dec.formatIdx, decodeSubdiv) & MICRO_FORMATINFO_CORR_MASK;
}

uint microdec_getStartPos(inout MicroDecoder dec, uint decodeSubdiv)
{
  return microdata_loadFormatInfo(dec.formatIdx, decodeSubdiv) >> MICRO_FORMATINFO_START_SHIFT;
}

/*

#define MICRO_FORMAT_64T_512B     0
#define MICRO_FORMAT_256T_1024B   1
#define MICRO_FORMAT_1024T_1024B  2

  correction bits per subdiv level

  fmtidx | type                   | 0   | 1   | 2   | 3   | 4   | 5   |
  =======|========================|=====|=====|=====|=====|=====|=====|
      2  | 1024 triangles - 1024b | 11  | 11  | 8   |  4  |  2  |  1  |
      1  |  256 triangles - 1024b | 11  | 11  | 11  | 10  |  5  |     |
      0  |   64 triangles - 512b  | 11  | 11  | 11  | 11  |     |     |

  startpos per subdiv level

  fmtidx | type                   | 0   | 1   | 2   | 3   | 4   | 5   |
  =======|========================|=====|=====|=====|=====|=====|=====|
      2  | 1024 triangles - 1024b | 0   | 33  | 66  | 138 | 258 | 474 |
      1  |  256 triangles - 1024b | 0   | 33  | 66  | 165 | 465 |     |
      0  |   64 triangles - 512b  | 0   | 33  | 66  | 165 |     |     |

  correction shift bit groups (packed 4 values) per subdiv level
                  
  fmtidx | type                   | 0   | 1   | 2   | 3   | 4   | 5   |
  =======|========================|=====|=====|=====|=====|=====|=====|
      2  | 1024 triangles - 1024b |  0  |  0  | 8   | 12  | 16  | 16  |
      1  |  256 triangles - 1024b |  0  |  0  | 0   | 4   | 12  |     |
      0  |   64 triangles - 512b  |  0  |  0  | 0   | 0   |     |     |

*/

#if MICRO_DECODER != MICRO_DECODER_BASETRI_MIP_SHUFFLE
// blockOffset can be used if we multiple blocks of same format are pointed
// to by dec.dataOffset and we want to encode a specific sub-block
void microdec_setCurrentSubdivisionLevel(inout MicroDecoder dec, uint decodeSubdiv, uint blockOffset)
{
  dec.decodeShiftBits = 0;

  if((dec.formatIdx == MICRO_FORMAT_64T_512B)
#if MICRO_FORMAT_SUPPORTED(MICRO_FORMAT_256T_1024B)
     || (dec.formatIdx == MICRO_FORMAT_256T_1024B && decodeSubdiv < 3)
#endif
  )
  {
    return;
  }
// any compressed formats enabled
#if MICRO_SUPPORTED_FORMAT_BITS != (1 << MICRO_FORMAT_64T_512B)
  else if(decodeSubdiv >= 2)
  {
    uint  block64s = true ? (1024 / 64) : (512 / 64);
    uvec2 bitsRead = microdata_loadDistance2((dec.dataOffset / 2) + ((1 + blockOffset) * block64s) - 1);

// all compressed enabled
#if MICRO_FORMAT_SUPPORTED(MICRO_FORMAT_256T_1024B) && MICRO_FORMAT_SUPPORTED(MICRO_FORMAT_1024T_1024B)
    if(dec.formatIdx == MICRO_FORMAT_256T_1024B)
#endif
#if MICRO_FORMAT_SUPPORTED(MICRO_FORMAT_256T_1024B)
    {
      if(decodeSubdiv >= 4)
      {
        int b = 14;  // 46 - 32
        dec.decodeShiftBits = (bitfieldExtract(bitsRead.y, b + 0, 3) << 0) | (bitfieldExtract(bitsRead.y, b + 3, 3) << 4)
                              | (bitfieldExtract(bitsRead.y, b + 6, 3) << 8) | (bitfieldExtract(bitsRead.y, b + 9, 3) << 12);
      }
      else if(decodeSubdiv >= 3)
      {
        int b = 26;  // 58 - 32
        dec.decodeShiftBits = (bitfieldExtract(bitsRead.y, b + 0, 1) << 0) | (bitfieldExtract(bitsRead.y, b + 1, 1) << 4)
                              | (bitfieldExtract(bitsRead.y, b + 2, 1) << 8) | (bitfieldExtract(bitsRead.y, b + 3, 1) << 12);
      }
      return;
    }
#endif
// all compressed enabled
#if MICRO_FORMAT_SUPPORTED(MICRO_FORMAT_256T_1024B) && MICRO_FORMAT_SUPPORTED(MICRO_FORMAT_1024T_1024B)
    else
#endif
#if MICRO_FORMAT_SUPPORTED(MICRO_FORMAT_1024T_1024B)
    {
      if(decodeSubdiv >= 4)
      {
        uint64_t bits64     = packUint2x32(bitsRead);
        uint     bits32     = uint(bits64 >> (10 + (5 - decodeSubdiv) * 16));
        dec.decodeShiftBits = bits32;
      }
      else if(decodeSubdiv >= 3)
      {
        int b = 10;  // 42 - 32
        dec.decodeShiftBits = (bitfieldExtract(bitsRead.y, b + 0, 3) << 0) | (bitfieldExtract(bitsRead.y, b + 3, 3) << 4)
                              | (bitfieldExtract(bitsRead.y, b + 6, 3) << 8) | (bitfieldExtract(bitsRead.y, b + 9, 3) << 12);
      }
      else if(decodeSubdiv >= 2)
      {
        int b = 22;  // 54 - 32
        dec.decodeShiftBits = (bitfieldExtract(bitsRead.y, b + 0, 2) << 0) | (bitfieldExtract(bitsRead.y, b + 2, 2) << 4)
                              | (bitfieldExtract(bitsRead.y, b + 4, 2) << 8) | (bitfieldExtract(bitsRead.y, b + 6, 2) << 12);
      }
    }
#endif
  }
#endif
}

int microdec_decodePredictionCorrection(inout MicroDecoder dec, uint vertexType, int correction)
{
  uint shift = (dec.decodeShiftBits >> (vertexType * 4)) & 0xF;
  return correction << shift;
}
#endif

int microdec_predict(inout MicroDecoder dec, int a, int b)
{
  return (a + b + 1) >> 1;
}

int microdec_compute(inout MicroDecoder dec, int a, int b, int correction)
{
  // mask to handle wrap around
  const int mask = (1 << MICRO_UNORM_BITS) - 1;
  return mask & (microdec_predict(dec, a, b) + correction);
}

uint microdec_getFormatSubdiv(inout MicroDecoder dec)
{
  return micromesh_getFormatSubdiv(dec.formatIdx);
}

uint microdec_getFormatIdx(inout MicroDecoder dec)
{
  return (dec.formatIdx);
}

void microdec_setDataOffset(inout MicroDecoder dec, uint dataOffset)
{
  dec.dataOffset = dataOffset;
}

void microdec_init(inout MicroDecoder dec, uint formatIdx, uint dataOffset, uint mipOffset)
{
  dec.formatIdx     = formatIdx;
  dec.dataOffset    = dataOffset;
  dec.mipOffset     = mipOffset;
  dec.blockSegments = subdiv_getNumSegments(micromesh_getFormatSubdiv(formatIdx));
}

#if MICRO_DECODER == MICRO_DECODER_BASETRI_MIP_SHUFFLE
#include "micromesh_decoder_basetri.glsl"
#elif MICRO_DECODER == MICRO_DECODER_MICROTRI_THREAD
#include "micromesh_decoder_microtri.glsl"
#else
#include "micromesh_decoder_subtri.glsl"
#endif
