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

#ifdef __cplusplus
#include <nvmath/nvmath.h>
namespace shaders {
using mat4 = nvmath::mat4f;
using vec4 = nvmath::vec4f;
using vec3 = nvmath::vec3f;
using vec2 = nvmath::vec2f;
#endif  // __cplusplus


struct PushConstant
{
  float metallic;
  float roughness;
  float intensity;
  int   maxDepth;
  int   numBaseTriangles;
};


struct FrameInfo
{
  mat4 proj;
  mat4 view;
  mat4 projInv;
  mat4 viewInv;
  vec3 camPos;
};


struct Light
{
  vec3  position;
  float intensity;
  vec3  color;
  int   type;
};

// From primitive
struct Vertex
{
  vec3 position;
  vec3 normal;
  vec2 t;
};

struct PrimMeshInfo
{
  uint64_t vertexAddress;
  uint64_t indexAddress;
};

struct InstanceInfo
{
  mat4 transform;
  int  materialID;
};

struct SceneDescription
{
  uint64_t materialAddress;
  uint64_t instInfoAddress;
  uint64_t primInfoAddress;
};

struct GltfShadeMaterial
{
  vec4 pbrBaseColorFactor;
  vec3 emissiveFactor;
  int  pbrBaseColorTexture;

  int   normalTexture;
  float normalTextureScale;
  int   shadingModel;
  float pbrRoughnessFactor;

  float pbrMetallicFactor;
  int   pbrMetallicRoughnessTexture;
  int   khrSpecularGlossinessTexture;
  int   khrDiffuseTexture;

  vec4  khrDiffuseFactor;
  vec3  khrSpecularFactor;
  float khrGlossinessFactor;

  int   emissiveTexture;
  int   alphaMode;
  float alphaCutoff;
};

#ifdef __cplusplus
}  // namespace shaders
#endif

#endif  // HOST_DEVICE_H
