/*
* Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

//////////////////////////////////////////////////////////////////////////
// this file is included by C++ and GLSL

#ifndef _CONFIG_H_
#define _CONFIG_H_

/////////////////////////////////////////////////////

#define API_SUPPORTED_SETUP_ONLY 1

// warning raising this beyond 5 has consequences on storage bits
// and needs manual changes in code
#define MAX_BASE_SUBDIV 5

#define BOUNDS_AS_FP32 1

#define ATOMIC_LAYERS 2

// must not change
#define SUBGROUP_SIZE 32

// Different surface visualization modes.
#define SURFACEVIS_SHADING 0     // Default shading.
#define SURFACEVIS_ANISOTROPY 1  // gl_PrimitiveID is not used; additional pervertexNV attributes; batlow coloring.
#define SURFACEVIS_BASETRI 2     // gl_PrimitiveID holds base triangle index; colorizePrimitive coloring.
#define SURFACEVIS_MICROTRI                                                                                            \
  3  // gl_PrimitiveID holds unique index per microtriangle; colorizePrimitive coloring. 0 for standard renderer.
#define SURFACEVIS_LOCALTRI                                                                                            \
  4  // gl_PrimitiveID holds local index of meshlet triangle; colorizePrimitive coloring. 0 for standard renderer.
#define SURFACEVIS_FORMAT                                                                                              \
  5  // gl_PrimitiveID holds index of encoding format used; colorizePrimitive coloring. 0 for non-umesh renderers.
#define SURFACEVIS_LODBIAS                                                                                             \
  6  // gl_PrimitiveID holds lod bias. custom hue2rgb coloring; valid for umesh-lod-renderers only.
#define SURFACEVIS_VALUERANGE 7  // gl_PrimitiveID holds effective base triangle range compared to mesh value range.
#define SURFACEVIS_BASESUBDIV 8  // gl_PrimitiveID holds base triangle subdiv level

#define CLEAR_COLOR 0.1, 0.13, 0.15, 0

///////////////////////////////////////////////////
#if defined(__cplusplus)

#include "../src/micromap/microdisp_shim.hpp"
#include <stddef.h>

// Verify that the SURFACEVIS values match the command-line documentation
static_assert(SURFACEVIS_SHADING == 0, "SURFACEVIS values must match docs!");
static_assert(SURFACEVIS_ANISOTROPY == 1, "SURFACEVIS values must match docs!");
static_assert(SURFACEVIS_BASETRI == 2, "SURFACEVIS values must match docs!");
static_assert(SURFACEVIS_MICROTRI == 3, "SURFACEVIS values must match docs!");
static_assert(SURFACEVIS_LOCALTRI == 4, "SURFACEVIS values must match docs!");
static_assert(SURFACEVIS_FORMAT == 5, "SURFACEVIS values must match docs!");
static_assert(SURFACEVIS_LODBIAS == 6, "SURFACEVIS values must match docs!");
static_assert(SURFACEVIS_VALUERANGE == 7, "SURFACEVIS values must match docs!");
static_assert(SURFACEVIS_BASESUBDIV == 8, "SURFACEVIS values must match docs!");

enum ModelType
{
  MODEL_LO,
  MODEL_DISPLACED,
  NUM_MODELTYPES,
  MODEL_SHELL,
};

// few more status prints
extern bool g_verbose;
// allow enabling raytracing extension for micromesh
// if true then codepaths assume native extension exists and rely on it
// if false we still do some fake setup work but the image will be the basemesh alone
extern bool g_enableMicromeshRTExtensions;
// number of default processing threads
extern uint32_t g_numThreads;

class float16_t
{
private:
  glm::detail::hdata h = 0;

public:
  float16_t() {}
  float16_t(float f) { h = glm::detail::toFloat16(f); }

  operator float() const { return glm::detail::toFloat32(h); }
};

struct f16vec2
{
  float16_t x;
  float16_t y;
};

struct f16vec4
{
  float16_t x;
  float16_t y;
  float16_t z;
  float16_t w;
};

struct u16vec2
{
  uint16_t x;
  uint16_t y;
};
struct u16vec4
{
  uint16_t x;
  uint16_t y;
  uint16_t z;
  uint16_t w;

  uint16_t& operator[](size_t i) { return (&x)[i]; }
};
struct u8vec2
{
  uint8_t x;
  uint8_t y;

  uint8_t& operator[](size_t i) { return (&x)[i]; }
};
struct u8vec4
{
  uint8_t x;
  uint8_t y;
  uint8_t z;
  uint8_t w;

  uint8_t& operator[](size_t i) { return (&x)[i]; }
};
#else

uint encodeMinMaxFp32(float val)
{
  uint bits = floatBitsToUint(val);
  bits ^= (int(bits) >> 31) | 0x80000000u;
  return bits;
}

float decodeMinMaxFp32(uint bits)
{
  bits ^= ~(int(bits) >> 31) | 0x80000000u;
  return uintBitsToFloat(bits);
}

const float FLT_MAX     = 3.402823466e+38F;
const float FLT_EPSILON = 1.192092896e-07F;

#endif

#endif
