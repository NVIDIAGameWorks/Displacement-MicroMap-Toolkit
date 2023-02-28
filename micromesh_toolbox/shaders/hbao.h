/*
*  Copyright (c) 2014-2023, NVIDIA CORPORATION.All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met :
*  * Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and / or other materials provided with the distribution.
*  * Neither the name of NVIDIA CORPORATION nor the names of its
*    contributors may be used to endorse or promote products derived
*    from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
*  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
*  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
*  PURPOSE ARE DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT OWNER OR
*  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
*  EXEMPLARY, OR CONSEQUENTIAL DAMAGES( INCLUDING, BUT NOT LIMITED TO,
*  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
*  PROFITS; OR BUSINESS INTERRUPTION ) HOWEVER CAUSED AND ON ANY THEORY
*  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
*  ( INCLUDING NEGLIGENCE OR OTHERWISE ) ARISING IN ANY WAY OUT OF THE USE
*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef NVHBAO_H_
#define NVHBAO_H_

#define NVHBAO_RANDOMTEX_SIZE 4
#define NVHBAO_NUM_DIRECTIONS 8

#define NVHBAO_MAIN_UBO 0
#define NVHBAO_MAIN_TEX_DEPTH 1
#define NVHBAO_MAIN_TEX_LINDEPTH 2
#define NVHBAO_MAIN_TEX_VIEWNORMAL 3
#define NVHBAO_MAIN_TEX_DEPTHARRAY 4
#define NVHBAO_MAIN_TEX_RESULTARRAY 5
#define NVHBAO_MAIN_TEX_RESULT 6
#define NVHBAO_MAIN_TEX_BLUR 7
#define NVHBAO_MAIN_IMG_LINDEPTH 8
#define NVHBAO_MAIN_IMG_VIEWNORMAL 9
#define NVHBAO_MAIN_IMG_DEPTHARRAY 10
#define NVHBAO_MAIN_IMG_RESULTARRAY 11
#define NVHBAO_MAIN_IMG_RESULT 12
#define NVHBAO_MAIN_IMG_BLUR 13
#define NVHBAO_MAIN_IMG_OUT 14

#ifndef NVHBAO_BLUR
#define NVHBAO_BLUR 1
#endif

// 1 is slower
#ifndef NVHBAO_SKIP_INTERPASS
#define NVHBAO_SKIP_INTERPASS 0
#endif

#ifdef __cplusplus
namespace glsl {
using namespace nvmath;
#endif

struct NVHBAOData
{
  float RadiusToScreen;  // radius
  float R2;              // 1/radius
  float NegInvR2;        // radius * radius
  float NDotVBias;

  vec2 InvFullResolution;
  vec2 InvQuarterResolution;

  ivec2 SourceResolutionScale;
  float AOMultiplier;
  float PowExponent;

  vec4  projReconstruct;
  vec4  projInfo;
  int   projOrtho;
  int   _pad0;
  ivec2 _pad1;

  ivec2 FullResolution;
  ivec2 QuarterResolution;

  mat4 InvProjMatrix;

  vec4 float2Offsets[NVHBAO_RANDOMTEX_SIZE * NVHBAO_RANDOMTEX_SIZE];
  vec4 jitters[NVHBAO_RANDOMTEX_SIZE * NVHBAO_RANDOMTEX_SIZE];
};

// keep all these equal size
struct NVHBAOMainPush
{
  int   layer;
  int   _pad0;
  ivec2 _pad1;
};

struct NVHBAOBlurPush
{
  vec2  invResolutionDirection;
  float sharpness;
  float _pad;
};

#ifdef __cplusplus
}
#else

layout(std140, binding = NVHBAO_MAIN_UBO) uniform controlBuffer
{
  NVHBAOData control;
};

#ifndef NVHABO_GFX

layout(local_size_x = 32, local_size_y = 2) in;

bool setupCoord(inout ivec2 coord, inout vec2 texCoord, ivec2 res, vec2 invRes)
{
  ivec2 base   = ivec2(gl_WorkGroupID.xy) * 8;
  ivec2 subset = ivec2(int(gl_LocalInvocationID.x) & 1, int(gl_LocalInvocationID.x) / 2);
  subset += gl_LocalInvocationID.x >= 16 ? ivec2(2, -8) : ivec2(0, 0);
  subset += ivec2(gl_LocalInvocationID.y * 4, 0);

  coord = base + subset;

  if(coord.x >= res.x || coord.y >= res.y)
    return true;

  texCoord = (vec2(coord) + vec2(0.5)) * invRes;

  return false;
}

bool setupCoordFull(inout ivec2 coord, inout vec2 texCoord)
{
  return setupCoord(coord, texCoord, control.FullResolution, control.InvFullResolution);
}

bool setupCoordQuarter(inout ivec2 coord, inout vec2 texCoord)
{
  return setupCoord(coord, texCoord, control.QuarterResolution, control.InvQuarterResolution);
}

#endif

#endif
#endif
