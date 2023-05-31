/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "device_host.h"
#include "dh_bindings.h"
#include "dh_scn_desc.h"
#include "octant_encoding.h"

#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/shaders/pbr_eval.glsl"
#include "nvvkhl/shaders/light_contrib.glsl"
#include "nvvkhl/shaders/dh_tonemap.h"
#include "nvvkhl/shaders/ray_util.glsl"

struct TriangleAttribute
{
  uint16_t subdLevel;
  uint8_t  primitiveFlags;
  uint8_t  openEdgeFlags;
};


// Buffers
layout(buffer_reference, scalar) readonly buffer GltfMaterials
{
  GltfShadeMaterial m[];
};
layout(buffer_reference, scalar) readonly buffer Vertices
{
  vec4 v[];
};
layout(buffer_reference, scalar) readonly buffer TexCoords
{
  vec2 v[];
};
layout(buffer_reference, scalar) readonly buffer Tangents
{
  vec2 v[];
};
layout(buffer_reference, scalar) readonly buffer Indices
{
  uvec3 i[];
};
layout(buffer_reference, scalar) readonly buffer Directions
{
  f16vec4 v[];
};

layout(buffer_reference, scalar) readonly buffer DeviceMeshInfos
{
  DeviceMeshInfo i[];
};
layout(buffer_reference, scalar) readonly buffer InstancesInfo
{
  InstanceInfo i[];
};
layout(buffer_reference, scalar) readonly buffer TriangleAttributes
{
  TriangleAttribute bt[];
};

layout(set = 0, binding = eFrameInfo) uniform FrameInfo_
{
  FrameInfo frameInfo;
};
layout(set = 0, binding = eSceneDesc) readonly buffer SceneDesc_
{
  SceneDescription sceneDesc;
};
layout(set = 0, binding = eTextures) uniform sampler2D[] texturesMap;

layout(set = 1, binding = 0) uniform sampler2D u_GGXLUT;                  // lookup table
layout(set = 1, binding = 1) uniform samplerCube u_LambertianEnvSampler;  //
layout(set = 1, binding = 2) uniform samplerCube u_GGXEnvSampler;         //

layout(set = 2, binding = eSkyParam) uniform SkyInfo_
{
  ProceduralSkyShaderParameters skyInfo;
};

  // clang-format on

#include "colormap.glsl"
#include "hit_state.h"

#include "nvvkhl/shaders/mat_eval.glsl"
#include "raster_simple_phong.glsl"
#include "raster_anisotropy.glsl"
#include "get_hit.glsl"
#include "config.h"


layout(push_constant) uniform RasterPushConstant_
{
  PushConstant pc;
};

layout(constant_id = 0) const int CONST_SHADE_MODE = 0;
layout(constant_id = 1) const int CONST_DEBUG_MODE = 0;


struct Vertex
{
  vec3 position;
};

vec3 getDiffuseLight(vec3 n)
{
  vec3 dir = rotate(n, vec3(0, 1, 0), -frameInfo.envRotation);
  return texture(u_LambertianEnvSampler, dir).rgb * frameInfo.envColor.rgb;
}

vec4 getSpecularSample(vec3 reflection, float lod)
{
  vec3 dir = rotate(reflection, vec3(0, 1, 0), -frameInfo.envRotation);
  return textureLod(u_GGXEnvSampler, dir, lod) * frameInfo.envColor;
}

// Calculation of the lighting contribution
vec3 getIBLContribution(vec3 n, vec3 v, float roughness, vec3 diffuseColor, vec3 specularColor)
{
  int   u_MipCount = textureQueryLevels(u_GGXEnvSampler);
  float lod        = (roughness * float(u_MipCount - 1));
  vec3  reflection = normalize(reflect(-v, n));
  float NdotV      = clampedDot(n, v);

  // retrieve a scale and bias to F0. See [1], Figure 3
  vec3 brdf          = (texture(u_GGXLUT, vec2(NdotV, 1.0 - roughness))).rgb;
  vec3 diffuseLight  = getDiffuseLight(n);
  vec3 specularLight = getSpecularSample(reflection, lod).xyz;

  vec3 diffuse  = diffuseLight * diffuseColor;
  vec3 specular = specularLight * (specularColor * brdf.x + brdf.y);

  return diffuse + specular;
}

void getTriangleVertex(in uint64_t vertexAddress, in uint64_t indexAddress, int triangle, out Vertex v[3])
{
  // Vextex and indices of the primitive
  Vertices vertices = Vertices(vertexAddress);
  Indices  indices  = Indices(indexAddress);

  // Getting the 3 indices of the triangle (local)
  uvec3 triangleIndex = indices.i[triangle];

  // All vertex attributes of the triangle.
  v[0].position = vertices.v[triangleIndex.x].xyz;
  v[1].position = vertices.v[triangleIndex.y].xyz;
  v[2].position = vertices.v[triangleIndex.z].xyz;
}

void getDirectionVertex(in uint64_t directionAddress, in uint64_t indexAddress, int triangle, out vec3 d[3])
{
  if(directionAddress == 0)
    return;

  // Vextex and indices of the primitive
  Directions directions = Directions(directionAddress);
  Indices    indices    = Indices(indexAddress);

  // Getting the 3 indices of the triangle (local)
  uvec3 triangleIndex = indices.i[triangle];

  // All vertex attributes of the triangle.
  d[0] = directions.v[triangleIndex.x].xyz;
  d[1] = directions.v[triangleIndex.y].xyz;
  d[2] = directions.v[triangleIndex.z].xyz;
}

// Returns the subdivision level for a UV edge that matches the frequency of
// heightmap texels. C++ reference: computeSubdivisionLevelsMatchingHeightmap()
float edgeSubdivLevel(vec2 edge, vec2 heightmapSize)
{
  edge            = abs(edge * heightmapSize);
  float maxTexels = max(1.0, max(edge.x, edge.y));
  return log2(maxTexels);
}

// Return the minimum subdiv level that will match the heightmap texel
// frequency. This is a fractional value on a log scale; take the ceiling for
// the integer subdiv level.
float triangleIdealSubdivLevel(in uint64_t texCoordsAddress, in uint64_t indexAddress, int triangle, vec2 heightmapSize)
{
  TexCoords texCoords     = TexCoords(texCoordsAddress);
  Indices   indices       = Indices(indexAddress);
  uvec3     vertexIndices = indices.i[triangle];
  vec2      uvEdge0       = texCoords.v[vertexIndices.y] - texCoords.v[vertexIndices.x];
  vec2      uvEdge1       = texCoords.v[vertexIndices.z] - texCoords.v[vertexIndices.y];
  vec2      uvEdge2       = texCoords.v[vertexIndices.x] - texCoords.v[vertexIndices.z];
  float     result        = 0.0;
  result                  = max(result, edgeSubdivLevel(uvEdge0, heightmapSize));
  result                  = max(result, edgeSubdivLevel(uvEdge1, heightmapSize));
  result                  = max(result, edgeSubdivLevel(uvEdge2, heightmapSize));
  return result;
}

//-----------------------------------------------------------------------
// Hashing function
//-----------------------------------------------------------------------
uint wangHash(uint seed)
{
  seed = (seed ^ 61) ^ (seed >> 16);
  seed *= 9;
  seed = seed ^ (seed >> 4);
  seed *= 0x27d4eb2d;
  seed = seed ^ (seed >> 15);
  return seed;
}


//-----------------------------------------------------------------------
// Debugging
//-----------------------------------------------------------------------

// Printing the information ONLY when the mouse is down and when the
// mouse coordinate equal the current fragment coordinate. Information
// is logged with the Debug level.
void debug(in vec3 val)
{
  ivec2 fragCoord = ivec2(floor(gl_FragCoord.xy));
  if(fragCoord == ivec2(frameInfo.mouseCoord))
    debugPrintfEXT("[%d, %d, %f] Val: %f, %f, %f\n", fragCoord.x, fragCoord.y, gl_FragCoord.z, val.x, val.y, val.z);
}

void debug(in int val)
{
  ivec2 fragCoord = ivec2(floor(gl_FragCoord.xy));
  if(fragCoord == ivec2(frameInfo.mouseCoord))
    debugPrintfEXT("[%d, %d, %f] Val: %d\n", fragCoord.x, fragCoord.y, gl_FragCoord.z, val);
}

vec3 uintToColor(uint val)
{
  val    = wangHash(val);
  uint r = val & 0x7FF;
  uint g = (val >> 11) & 0x7FF;
  uint b = (val >> 22) & 0x3FF;
  return vec3(float(r) / 2047.0, float(g) / 2047.0, float(b) / 1023.0);
}


//-----------------------------------------------------------------------
// This function finds the material and hi information
//
void rasterLoad(vec3 position, vec3 baryCoord, int triangle, out GltfShadeMaterial gltfMat, out DeviceMeshInfo pinfo, out TriangleAttribute triInfo, out HitState hit, out vec3 toEye)
{
  // Material of the object
  GltfMaterials gltfMats = GltfMaterials(sceneDesc.materialAddress);
  gltfMat                = gltfMats.m[pc.materialID];

  // Instances
  InstancesInfo instances = InstancesInfo(sceneDesc.instInfoAddress);
  InstanceInfo  instinfo  = instances.i[pc.instanceID];

  // Primitive meshes
  DeviceMeshInfos pInfos = DeviceMeshInfos(sceneDesc.deviceMeshInfoAddress);
  pinfo                  = pInfos.i[pc.primMeshID];


  // Triangle information
  triInfo   = TriangleAttribute(uint16_t(0), uint8_t(0), uint8_t(0));

  if(pinfo.triangleAttributesBuffer != 0)
  {
    TriangleAttributes tris = TriangleAttributes(pinfo.triangleAttributesBuffer);
    triInfo                 = tris.bt[triangle];
  }

  const vec3 eyePos            = vec3(frameInfo.viewInv[3].x, frameInfo.viewInv[3].y, frameInfo.viewInv[3].z);
  const vec3 worldRayDirection = normalize(position - eyePos);
  toEye                        = -worldRayDirection;

  hit = getHitState(pinfo, baryCoord, gl_PrimitiveID, mat4x3(instinfo.objectToWorld), mat4x3(instinfo.worldToObject),
                    worldRayDirection, gl_FrontFacing);
}


//-------------------------------------------------------------------------------------------------
// Return the color of the subdivision level, and the edge has the color to the level it should
// be subdivided
//
//       bary.z
//      /     \ 
//     2       1
//    /         \ 
// bary.x __0_ bary.y
//
vec3 subdDecFlagsToColor(uint subdLevel, int subdMax, uint primitiveFlags, vec3 bary)
{
  float threshold = 0.1f;  // Edge size

  vec3 coreColor = colorMap(frameInfo.colormap, float(subdLevel) / float(subdMax));

  if(subdLevel == 0)
    return coreColor;

  vec3 decColor = colorMap(frameInfo.colormap, float(subdLevel - 1) / float(subdMax));

  if(bary.x < threshold || bary.y < threshold || bary.z < threshold)
  {
    if(bary.x < bary.y && bary.x < bary.z)
    {
      if(((primitiveFlags & (1 << 1)) != 0))
      {
        float w = 1.f - bary.x / threshold;
        return coreColor * (1.f - w) + decColor * w;
      }
      return coreColor;
    }

    if(bary.y < bary.z)
    {
      if(((primitiveFlags & (1 << 2)) != 0))
      {
        float w = 1.f - bary.y / threshold;
        return coreColor * (1.f - w) + decColor * w;
      }
      return coreColor;
    }

    {
      if(((primitiveFlags & (1 << 0)) != 0))
      {
        float w = 1.f - bary.z / threshold;
        return coreColor * (1.f - w) + decColor * w;
      }
      return coreColor;
    }
  }
  return coreColor;
}


//-----------------------------------------------------------------------
// This returns the shading for the wanted sahded mode
//
vec4 rasterShade(in vec3 baryCoord, in GltfShadeMaterial gltfMat, in DeviceMeshInfo pinfo, in TriangleAttribute triInfo, in HitState hit, in vec3 toEye, int triangle)
{
  float NdotL = dot(hit.nrm, toEye);

  // All various information that can be rendered
  switch(CONST_SHADE_MODE)
  {
    case eRenderShading_phong:
      return vec4(simplePhong(toEye, hit.geonrm), 1);

    case eRenderShading_anisotropy: {
      Vertex v[3];
      getTriangleVertex(pinfo.vertexPositionNormalBuffer, pinfo.triangleVertexIndexBuffer, triangle, v);

      float t = anisotropyMetric(v[0].position.xyz, v[1].position.xyz, v[2].position.xyz);
      return simpleShade(colorMap(frameInfo.colormap, t), NdotL);
    }

    case eRenderShading_baseTriangleIndex:
      return simpleShade(uintToColor(gl_PrimitiveID), NdotL);

    case eRenderShading_subdivLevel: {
      // Visualize subdiv level that a triangle would be subdivided to.
      // draw_compressed_basic.frag.glsl has its own path when rendering
      // microtriangles. Draw magenta if there is no data rather than default 0.
      bool hasSubdivLevels = (pinfo.sourceAttribFlags & eMeshAttributeTriangleSubdivLevelsBit) != 0;
      vec3 color = hasSubdivLevels ? subdDecFlagsToColor(triInfo.subdLevel, MAX_BASE_SUBDIV, triInfo.primitiveFlags, baryCoord) : vec3(1, 0, 1);
      return simpleShade(color, NdotL);
    }

    case eRenderShading_minMax:
      return simpleShade(colorMap(frameInfo.colormap, 0 /*TODO*/), NdotL);

    case eRenderShading_heightmapTexelFrequency: {
      if(gltfMat.khrDisplacementTexture == -1)
      {
        // Display error with magenta. There is no heightmap texture loaded.
        return vec4(1, 0, 1, 1);
      }
      else
      {
        vec2  heightmapSize    = vec2(textureSize(texturesMap[gltfMat.khrDisplacementTexture], 0));
        float idealSubdivLevel = triangleIdealSubdivLevel(pinfo.vertexTexcoordBuffer, pinfo.triangleVertexIndexBuffer,
                                                          gl_PrimitiveID, heightmapSize);

        // Handle per-triangle subdiv levels from the remesher (or gltf
        // NV_micromap_tooling), falling back to the global target subdiv level
        // if none exist.
        bool hasSubdivLevels = (pinfo.sourceAttribFlags & eMeshAttributeTriangleSubdivLevelsBit) != 0;
        int  subdivLevel     = hasSubdivLevels ? triInfo.subdLevel : pc.bakeSubdivLevel;

        // Display difference to ideal subdiv level in fifths (matching HW max.
        // subdiv level)
        float t = max(0.0, (idealSubdivLevel - float(subdivLevel)) / 5.0);
        return simpleShade(colorMap(frameInfo.colormap, t), NdotL);
      }
    }
    case eRenderShading_opposingDirections: {
      float r = 0;
      if(pinfo.vertexDirectionsBuffer != 0)
      {
        vec3 d[3];
        getDirectionVertex(pinfo.vertexDirectionsBuffer, pinfo.triangleVertexIndexBuffer, triangle, d);
        r = (dot(d[0], d[1]) < 0 || dot(d[1], d[2]) < 0 || dot(d[2], d[0]) < 0) ? 1 : 0;
      }
      return simpleShade(colorMap(frameInfo.colormap, r), NdotL);
    }
    case eRenderShading_sharedPositions: {
      Vertex v[3];
      getTriangleVertex(pinfo.vertexPositionNormalBuffer, pinfo.triangleVertexIndexBuffer, triangle, v);
      vec3 p;
      if(baryCoord.x > baryCoord.y && baryCoord.x > baryCoord.z)
        p = v[0].position;
      else if(baryCoord.y > baryCoord.z)
        p = v[1].position;
      else
        p = v[2].position;

      uint hashValue = wangHash(floatBitsToUint(p.z) + wangHash(floatBitsToUint(p.y) + wangHash(floatBitsToUint(p.x))));
      vec3 color     = uintToColor(hashValue);

      return simpleShade(color, NdotL);
    }
  }


  if(gl_FrontFacing == false)
  {
    hit.tangent *= -1.0;
    hit.bitangent *= -1.0;
    hit.nrm *= -1.0;
  }

  PbrMaterial pbrMat = evaluateMaterial(gltfMat, hit.nrm, hit.tangent, hit.bitangent, hit.uv);

  // Overriding shading
  if(CONST_SHADE_MODE == eRenderShading_faceted)
  {
    pbrMat.albedo    = vec4(1.0);
    pbrMat.roughness = frameInfo.roughness;
    pbrMat.metallic  = frameInfo.metallic;
    pbrMat.f0        = mix(vec3(g_min_reflectance), vec3(pbrMat.albedo), float(pbrMat.metallic));
    pbrMat.normal    = hit.geonrm;
  }

  switch(CONST_DEBUG_MODE)
  {
    case eDbgMethod_metallic:
      return vec4(vec3(pbrMat.metallic), 1);
    case eDbgMethod_roughness:
      return vec4(vec3(pbrMat.roughness), 1);
    case eDbgMethod_normal:
      return vec4(toLinear(vec3(pbrMat.normal * .5 + .5)), 1);
    case eDbgMethod_basecolor:
      return vec4(vec3(pbrMat.albedo), 1);
    case eDbgMethod_emissive:
      return vec4(vec3(pbrMat.emissive), 1);
    case eDbgMethod_txtcoord:
      return vec4(toLinear(vec3(fract(hit.uv), 0.)), 1);
    case eDbgMethod_direction: {
      vec3 d[3] = vec3[3](vec3(0), vec3(0), vec3(0));
      getDirectionVertex(pinfo.vertexDirectionsBuffer, pinfo.triangleVertexIndexBuffer, triangle, d);
      vec3 result = mixBary(d[0], d[1], d[2], baryCoord);
      return vec4(toLinear(result * .5 + .5), 1);
    }
  }

  if(gltfMat.alphaMode == ALPHA_MASK)
  {
    if(pbrMat.albedo.a < gltfMat.alphaCutoff)
      discard;
  }


  // Result
  vec3 result = vec3(0);

  float ambientFactor = 0.3;
  if(frameInfo.useSky != 0)
  {
    vec3 ambientColor = mix(skyInfo.groundColor.rgb, skyInfo.skyColor.rgb, pbrMat.normal.y * 0.5 + 0.5) * ambientFactor;
    result += ambientColor * pbrMat.albedo.rgb;
    result += ambientColor * pbrMat.f0;
  }
  else
  {
    // Calculate lighting contribution from image based lighting source (IBL)
    vec3 diffuseColor  = pbrMat.albedo.rgb * (vec3(1.0) - pbrMat.f0) * (1.0 - pbrMat.metallic);
    vec3 specularColor = mix(pbrMat.f0, pbrMat.albedo.rgb, pbrMat.metallic);

    result += getIBLContribution(pbrMat.normal, toEye, pbrMat.roughness, diffuseColor, specularColor);
  }


  result += pbrMat.emissive;  // emissive

  // All lights
  for(int i = 0; i < frameInfo.nbLights; i++)
  {
    Light        light        = frameInfo.light[i];
    LightContrib lightContrib = singleLightContribution(light, hit.pos, pbrMat.normal, toEye);

    float pdf      = 0;
    vec3  brdf     = pbrEval(pbrMat, toEye, -lightContrib.incidentVector, pdf);
    vec3  radiance = brdf * lightContrib.intensity;
    result += radiance;
  }

  // This is an example on how data can be debugged
  // debug(result);

  return vec4(result, pbrMat.albedo.a);
}
